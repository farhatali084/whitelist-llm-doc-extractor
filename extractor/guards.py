"""Post-processing guardrails: turn a validated LLMResponse into the final Extraction.

Principle: the model proposes, the document decides. Every non-null value must be
found in the source text (possibly after OCR repair / number-word parsing);
otherwise it is set to null and a warning explains why. Contradictions (two
issue dates, two different totals) are detected here and resolved by fixed rules
documented in README.md.
"""
from __future__ import annotations

import re
from datetime import date

from . import textnorm as tn
from .schema import Extraction, LineItem, LLMResponse

# ----------------------------- patterns ------------------------------------ #

_TITLE = {
    "commercial_offer": r"коммерческое\s+предложение|коммерциялық\s+ұсыныс|commercial\s+offer|quotation",
    "invoice": r"сч[её]т|шот|invoice|төлем\s+шоты",
    "act": r"акт\b|актісі\b|орындалған\s+жұмыстар|act\s+of|certificate\s+of\s+completion",
}
_HEADER_LINES = 6

_CURRENCY = {
    "KZT": r"тенге|теңге|(?<!\w)тг(?!\w)|₸|(?<!\w)KZT(?!\w)",
    "USD": r"\$|(?<!\w)USD(?!\w)|доллар|долл\.",
    "EUR": r"€|(?<!\w)EUR(?!\w)|евро|(?<!\w)euro",
}

_VAT = re.compile(r"ндс|ққс|(?<!\w)vat(?!\w)", re.I)
_VAT_INCLUSIVE = re.compile(r"(?:с|с\s+учетом|с\s+учётом|включая|including)\s+(?:ндс|vat)|ққс-?пен", re.I)
_NO_VAT = re.compile(
    r"без\s+(?:учета\s+|учёта\s+)?ндс|ндс\s+не\s+облага|не\s+облагается\s+ндс|"
    r"ққс-?сыз|ққс\s+салынбайды|ққс\s+есепке\s+алынбаған|"
    r"without\s+vat|vat\s+exempt|not\s+subject\s+to\s+vat",
    re.I,
)
_PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")

# Total labels, by priority: amount payable > grand total > subtotal.
_TOTAL_LABELS = [
    (3, re.compile(r"к\s+оплате|төлеуге|amount\s+due|total\s+due", re.I)),
    (2, re.compile(r"всего|барлығы|grand\s+total", re.I)),
    (1, re.compile(r"итого|жалпы\s+сома|(?<!\w)total(?!\w)", re.I)),
]

_ISSUE_LABEL = re.compile(r"(?:дата|күні|date|(?<!\w)от|составлено|составлен|dated|жасалған)\W*$", re.I)


class _Builder:
    """Collects values, confidence scores and warnings for one document."""

    def __init__(self) -> None:
        self.values: dict = {}
        self.confidence: dict[str, float] = {}
        self.warnings: list[str] = []

    def set(self, field: str, value, confidence: float) -> None:
        self.values[field] = value
        if value is not None:
            self.confidence[field] = round(confidence, 2)

    def warn(self, code: str, message: str) -> None:
        self.warnings.append(f"{code}: {message}")


# ----------------------------- field guards -------------------------------- #

def _guard_doc_type(resp: LLMResponse, text: str, b: _Builder) -> None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:_HEADER_LINES]
    lines += [tn.skeleton(ln) for ln in lines]  # OCR: "Cчeт" written with Latin C/e
    in_title = {
        t for t, pat in _TITLE.items()
        if any(re.match(rf"^\W*(?:{pat})", ln, re.I) for ln in lines)
    }
    llm = resp.doc_type
    if len(in_title) == 1:
        title_type = next(iter(in_title))
        if llm not in (title_type, "unknown"):
            b.warn("doc_type_conflict", f"model said '{llm}', document title says '{title_type}'; using title")
        b.set("doc_type", title_type, 1.0 if llm == title_type else 0.8)
    elif llm in in_title:
        b.set("doc_type", llm, 0.8)
    elif in_title:
        chosen = sorted(in_title)[0]
        b.warn("doc_type_ambiguous", f"title matches {sorted(in_title)}, model said '{llm}'; using '{chosen}'")
        b.set("doc_type", chosen, 0.5)
    elif llm != "unknown" and re.search(_TITLE[llm], text, re.I):
        b.set("doc_type", llm, 0.6)
    else:
        if llm != "unknown":
            b.warn("doc_type_ungrounded", f"model said '{llm}' but no such document title found")
        b.set("doc_type", "unknown", 1.0)


def _guard_supplier_name(resp: LLMResponse, text: str, b: _Builder) -> None:
    name = resp.supplier_name.strip() if resp.supplier_name else None
    if name and not tn.name_is_grounded(name, text):
        b.warn("supplier_name_dropped", f"'{name}' not found in document")
        name = None
    b.set("supplier_name", name, 1.0 if name and name in text else 0.8)


def _guard_bin(resp: LLMResponse, text: str, b: _Builder) -> None:
    bins = tn.find_bins(text)
    value, conf = None, 1.0
    if resp.supplier_bin:
        cleaned = tn.clean_bin(resp.supplier_bin)
        if cleaned in bins:
            value = cleaned
        else:
            b.warn("supplier_bin_dropped", f"'{resp.supplier_bin}' does not occur in document")
    elif len(bins) == 1:
        value, conf = next(iter(bins)), 0.7
        b.warn("supplier_bin_from_text", f"model returned null; the only BIN in the document is {value}")
    if value:
        if bins.get(value):
            conf = min(conf, 0.8)
            b.warn("supplier_bin_ocr_repaired", f"BIN read as {value} after fixing OCR look-alike characters")
        if not tn.bin_checksum_ok(value):
            conf = min(conf, 0.4)
            b.warn("supplier_bin_checksum", f"{value} fails the BIN check digit; kept as written, verify manually")
    b.set("supplier_bin", value, conf)


def _guard_doc_number(resp: LLMResponse, text: str, b: _Builder) -> None:
    num = resp.doc_number
    if num:
        num = re.sub(r"^\s*(?:№|N[o°]?\.?)\s*", "", num).strip()
        # "AKT-09 / 09 қазан" or "1245 от 12.11" -> keep only the number itself
        num = re.split(r"\s+/\s+|\s+(?:от|from|dated)\s+", num, maxsplit=1)[0].strip()
        num = tn.fold_ocr_digits(num)  # "2O7" -> "207"
        if not num or tn.fold_homoglyphs(num) not in tn.fold_homoglyphs(tn.fold_ocr_digits(text)):
            b.warn("doc_number_dropped", f"'{resp.doc_number}' not found in document")
            num = None
    b.set("doc_number", num, 1.0 if num and num in text else 0.8)


def _issue_dates(text: str) -> list[date]:
    """Issue dates: preceded by a label ("от", "Дата", "Күні", "Составлено"),
    or written on the same line as the document number ("Шот-фактура № X / 09 қазан 2026")."""
    folded = tn.fold_ocr_digits(text)
    out: list[date] = []
    for d, pos in tn.find_dates(text):
        line_start = folded.rfind("\n", 0, pos) + 1
        labelled = _ISSUE_LABEL.search(folded[max(line_start, pos - 25):pos])
        on_number_line = re.search(r"№|N°|(?<!\w)No\.?\s*\d", folded[line_start:pos]) is not None
        if (labelled or on_number_line) and d not in out:
            out.append(d)
    return out


def _guard_doc_date(resp: LLMResponse, text: str, b: _Builder) -> None:
    value, conf = resp.doc_date, 1.0
    if value and not tn.date_is_grounded(value, text):
        b.warn("doc_date_dropped", f"{value.isoformat()} does not occur in document")
        value = None
    issue = _issue_dates(text)
    if len(issue) > 1:
        b.warn(
            "conflicting_dates",
            f"several issue dates {[d.isoformat() for d in issue]}; using the first (header) one",
        )
        value, conf = issue[0], 0.5
    elif len(issue) == 1 and value != issue[0]:
        if value is not None:  # model took a validity/due date instead of the issue date
            b.warn("doc_date_relabelled", f"model chose {value.isoformat()}; labelled issue date is {issue[0].isoformat()}")
        value, conf = issue[0], 0.7
    b.set("doc_date", value, conf)


def _guard_currency(resp: LLMResponse, text: str, b: _Builder) -> None:
    seen = [c for c, pat in _CURRENCY.items() if re.search(pat, text, re.I)]
    value = resp.currency
    if value and value not in seen:
        b.warn("currency_dropped", f"model said {value}, but no such currency in document")
        value = None
    if value is None and len(seen) == 1:
        value = seen[0]
    if len(seen) > 1:
        b.warn("multiple_currencies", f"document mentions {seen}")
    b.set("currency", value, 1.0 if len(seen) == 1 else 0.6)


def _line_numbers(line: str) -> list[float]:
    line = _PERCENT.sub(" ", line)
    return [v for v in tn.numbers_in_text(line) if v >= 1]


def _guard_vat(resp: LLMResponse, text: str, b: _Builder) -> None:
    vat_values: set[float] = set()
    for line in text.splitlines():
        if _VAT.search(line) and not _VAT_INCLUSIVE.search(line) and not _NO_VAT.search(line):
            nums = _line_numbers(line[_VAT.search(line).end():])
            if nums:
                vat_values.add(max(nums))
    value = resp.vat_amount
    if value is not None and not any(abs(value - v) < 0.005 for v in vat_values):
        reason = "document says no VAT" if _NO_VAT.search(text) else "no VAT amount written in document"
        b.warn("vat_dropped", f"model returned {value}, but {reason}")
        value = None
    if value is None and len(vat_values) == 1:
        value = next(iter(vat_values))
    b.set("vat_amount", value, 1.0 if value is not None and tn.number_is_verbatim(value, text) else 0.8)


def _total_candidates(text: str) -> list[tuple[int, float]]:
    out = []
    for line in text.splitlines():
        if _VAT.search(line) and not _VAT_INCLUSIVE.search(line):
            continue  # "Итого НДС: ..." is a VAT line, not a total
        prio, end = 0, 0
        for p, pat in _TOTAL_LABELS:
            m = pat.search(line)
            if m and p > prio:
                prio, end = p, m.end()
        if prio:
            nums = _line_numbers(line[end:])
            if nums:
                out.append((prio, max(nums)))
    return out


def _guard_total(resp: LLMResponse, text: str, b: _Builder) -> None:
    value, conf = resp.total_amount, 1.0
    if value is not None and not tn.number_is_grounded(value, text):
        b.warn("total_dropped", f"{value} does not occur in document (computed by model?)")
        value = None
    cands = _total_candidates(text)
    if cands:
        best = max(p for p, _ in cands)
        rule_value = [v for p, v in cands if p == best][-1]
        distinct = sorted({v for _, v in cands})
        vat = b.values.get("vat_amount")
        explained = len(distinct) == 2 and vat is not None and abs(distinct[1] - distinct[0] - vat) < 0.01
        if len(distinct) > 1 and not explained:
            b.warn(
                "conflicting_totals",
                f"different totals {distinct}; using {rule_value} (rule: 'к оплате' > 'всего' > 'итого', last one)",
            )
            conf = 0.5
        elif value is not None and abs(value - rule_value) > 0.005:
            b.warn("total_relabelled", f"model chose {value}; labelled amount payable is {rule_value}")
            conf = 0.8
        value = rule_value
    if value is not None and not tn.number_is_verbatim(value, text):
        conf = min(conf, 0.8)  # came from words or OCR repair
    b.set("total_amount", value, conf)


def _guard_line_items(resp: LLMResponse, text: str, b: _Builder) -> None:
    items: list[LineItem] = []
    for it in resp.line_items:
        if not tn.name_is_grounded(it.name, text, min_ratio=0.5):
            b.warn("line_item_dropped", f"'{it.name}' not found in document")
            continue
        qty, price = it.qty, it.price
        if qty is not None and not tn.number_is_grounded(qty, text):
            b.warn("line_item_qty_dropped", f"qty {qty} of '{it.name}' not in document")
            qty = None
        if price is not None and not tn.number_is_grounded(price, text):
            b.warn("line_item_price_dropped", f"price {price} of '{it.name}' not in document")
            price = None
        items.append(LineItem(name=it.name.strip(), qty=qty, price=price))
    b.values["line_items"] = items


# ----------------------------- entry points -------------------------------- #

def apply_guards(resp: LLMResponse, text: str) -> Extraction:
    b = _Builder()
    _guard_doc_type(resp, text, b)
    _guard_supplier_name(resp, text, b)
    _guard_bin(resp, text, b)
    _guard_doc_number(resp, text, b)
    _guard_doc_date(resp, text, b)
    _guard_currency(resp, text, b)
    _guard_vat(resp, text, b)  # before total: VAT explains subtotal vs total
    _guard_total(resp, text, b)
    _guard_line_items(resp, text, b)
    b.warnings.extend(f"model: {w}" for w in resp.warnings if w.strip())
    return Extraction(**b.values, field_confidence=b.confidence, warnings=b.warnings)


def without_guards(resp: LLMResponse) -> Extraction:
    """Raw model answer mapped to the output schema (baseline for the eval)."""
    data = resp.model_dump(exclude={"warnings"})
    bin_ = tn.clean_bin(data["supplier_bin"]) if data["supplier_bin"] else None
    data["supplier_bin"] = bin_ if bin_ and re.fullmatch(r"\d{12}", bin_) else None
    return Extraction(**data, warnings=[f"model: {w}" for w in resp.warnings])
