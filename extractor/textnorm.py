"""Deterministic text utilities used to *ground* LLM answers in the source text.

Nothing here calls a model. Every function answers the question
"does this value really occur in the document?" (possibly after OCR repair).
"""
from __future__ import annotations

import re
from datetime import date

# --------------------------------------------------------------------------- #
# Generic normalisation
# --------------------------------------------------------------------------- #

# Letters OCR typically confuses with digits. Applied only next to digits.
_OCR_DIGIT_MAP = str.maketrans(
    {"O": "0", "o": "0", "О": "0", "о": "0", "l": "1", "I": "1", "|": "1",
     "З": "3", "з": "3", "S": "5", "б": "6"}
)
_OCR_CHARS = "OoОоlI|ЗзSб"

# Cyrillic letters that look exactly like Latin ones (for doc numbers like "AKT-09").
_HOMOGLYPHS = str.maketrans("АВЕКМНОРСТХУавекмнорстху", "ABEKMHOPCTXYABEKMHOPCTXY")

# Kazakh-specific letters -> closest Russian letter (for fuzzy name matching).
_KK_TO_RU = str.maketrans("әғқңөұүһіё", "агкноуухие")

_LEGAL_FORMS = {"тоо", "жшс", "ип", "жк", "ао", "ак", "llp", "llc", "ltd", "too", "ооо"}


def fold_ocr_digits(text: str) -> str:
    """Replace look-alike letters with digits when they sit inside a digit run.

    "2l064090Ol80" -> "210640900180", "0З.10.2026" -> "03.10.2026".
    Ordinary words are untouched because a letter is replaced only when a
    neighbour (ignoring one space/dot) is a digit or another OCR char.
    """
    chars = list(text)

    def neighbour(i: int, step: int) -> str:
        j = i + step
        if 0 <= j < len(chars) and chars[j] == " ":  # "9O OOO": look across one space
            j += step
        return chars[j] if 0 <= j < len(chars) else ""

    for _ in range(3):  # a few passes so chains like "Ol" next to a digit both fold
        changed = False
        for i, ch in enumerate(chars):
            if ch not in _OCR_CHARS:
                continue
            left = chars[i - 1] if i > 0 else ""
            right = chars[i + 1] if i + 1 < len(chars) else ""
            if neighbour(i, -1).isdigit() or neighbour(i, 1).isdigit():
                # Do not fold the first letter of a real word ("Оказание", "Итого 5").
                if right.isalpha() and right not in _OCR_CHARS:
                    continue
                if left.isalpha() and left not in _OCR_CHARS:
                    continue
                chars[i] = ch.translate(_OCR_DIGIT_MAP)
                changed = True
        if not changed:
            break
    return "".join(chars)


def fold_homoglyphs(s: str) -> str:
    return re.sub(r"\s+", "", s.translate(_HOMOGLYPHS).upper())


_LATIN_TO_CYR = str.maketrans("abekmhopctxy", "авекмнорстху")


def skeleton(s: str) -> str:
    """Comparison key: lower-case, Kazakh -> Russian letters, Latin look-alikes -> Cyrillic.
    Applied to BOTH sides of a comparison, so "TOO \"BOCTOK\"" == "ТОО «Восток»"."""
    return s.lower().translate(_KK_TO_RU).translate(_LATIN_TO_CYR)


_LEGAL_SKELETONS = {skeleton(f) for f in _LEGAL_FORMS}


def normalize_name(s: str) -> str:
    """Comparison form of a name: skeleton, no quotes, no legal forms (ТОО, ЖШС, LLP...)."""
    s = re.sub(r"[«»\"'“”„`()]", " ", skeleton(s))
    return " ".join(t for t in re.split(r"[\s/,.]+", s) if t and t not in _LEGAL_SKELETONS)


def name_tokens(s: str) -> list[str]:
    return [t for t in normalize_name(s).split() if len(t) >= 3]


def name_is_grounded(name: str, text: str, min_ratio: float = 0.6) -> bool:
    """True if most significant words of ``name`` occur in the text."""
    tokens = name_tokens(name)
    if not tokens:
        return False
    hay = normalize_name(text)
    found = sum(1 for t in tokens if t in hay)
    return found / len(tokens) >= min_ratio


# --------------------------------------------------------------------------- #
# BIN (бизнес-идентификационный номер), 12 digits with a check digit
# --------------------------------------------------------------------------- #

_BIN_LABEL = re.compile(r"(?:БИН|БСН|BIN|ИИН|ЖСН)\s*(?:/\s*(?:БИН|БСН|BIN|ИИН|ЖСН))?\s*[:№]?\s*", re.I)
_BIN_BODY = re.compile(r"[0-9OoОоlI|ЗзSб][0-9OoОоlI|ЗзSб \-]{10,20}")


def bin_checksum_ok(bin_: str) -> bool:
    """Kazakhstan IIN/BIN control digit (two-pass weighted mod 11)."""
    if not re.fullmatch(r"\d{12}", bin_):
        return False
    d = [int(c) for c in bin_]
    s = sum(d[i] * (i + 1) for i in range(11)) % 11
    if s == 10:
        weights = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        s = sum(d[i] * weights[i] for i in range(11)) % 11
        if s == 10:
            return False
    return s == d[11]


def clean_bin(raw: str) -> str:
    """Model's BIN string -> digits only, with OCR look-alikes folded."""
    return re.sub(r"[\s\-]", "", raw.translate(_OCR_DIGIT_MAP))


def _clean_bin_candidate(raw: str) -> tuple[str | None, bool]:
    """Return (12-digit BIN or None, was_ocr_repaired)."""
    folded = raw.translate(_OCR_DIGIT_MAP)
    digits = re.sub(r"[ \-]", "", folded)
    m = re.match(r"\d{12}", digits)
    if not m:
        return None, False
    original_digits = re.sub(r"[ \-]", "", raw)[:12]
    return m.group(0), original_digits != m.group(0)


def find_bins(text: str) -> dict[str, bool]:
    """All BINs in the text -> {bin: was_ocr_repaired}.

    Labelled ones (after "БИН"/"БСН") are allowed OCR repair; unlabelled ones
    must be a clean run of exactly 12 digits.
    """
    found: dict[str, bool] = {}
    for label in _BIN_LABEL.finditer(text):
        body = _BIN_BODY.match(text, label.end())
        if body:
            bin_, repaired = _clean_bin_candidate(body.group(0))
            if bin_:
                found[bin_] = found.get(bin_, False) or repaired
    for m in re.finditer(r"(?<!\d)\d{12}(?!\d)", text):
        found.setdefault(m.group(0), False)
    # Unlabelled (label itself OCR-damaged, e.g. "6NH"): a 12-char token made of
    # digits and look-alikes, at least half of them real digits.
    for m in re.finditer(r"(?<![\w])[0-9OoОоlI|Зз]{12}(?![\w])", text):
        token = m.group(0)
        if sum(c.isdigit() for c in token) >= 6:
            folded = token.translate(_OCR_DIGIT_MAP)
            found[folded] = found.get(folded, False) or folded != token
    return found


# --------------------------------------------------------------------------- #
# Numbers: digits and words (Russian + Kazakh)
# --------------------------------------------------------------------------- #

_RU_WORDS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_KK_WORDS = {
    "нөл": 0, "бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7,
    "сегіз": 8, "тоғыз": 9, "он": 10, "жиырма": 20, "отыз": 30, "қырық": 40, "елу": 50,
    "алпыс": 60, "жетпіс": 70, "сексен": 80, "тоқсан": 90,
}
_UNIT_WORDS = {**_RU_WORDS, **_KK_WORDS}
_HUNDRED_MULT = {"жүз"}  # Kazakh: "үш жүз" = 3 * 100
_SCALES = {
    "тысяча": 1_000, "тысячи": 1_000, "тысяч": 1_000, "мың": 1_000,
    "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000,
    "миллиард": 1_000_000_000, "миллиарда": 1_000_000_000, "миллиардов": 1_000_000_000,
}
_NUMBER_VOCAB = set(_UNIT_WORDS) | _HUNDRED_MULT | set(_SCALES)


def words_to_number(tokens: list[str]) -> int | None:
    """Parse a run of number words: "один миллион двести тысяч" -> 1200000."""
    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok in _UNIT_WORDS:
            current += _UNIT_WORDS[tok]
        elif tok in _HUNDRED_MULT:
            current = (current or 1) * 100
        elif tok in _SCALES:
            total += (current or 1) * _SCALES[tok]
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


def word_numbers_in_text(text: str) -> list[int]:
    """Every maximal run of number words in the text, parsed to an int."""
    out: list[int] = []
    run: list[str] = []
    for tok in re.findall(r"[^\W\d_]+", text.lower()) + [""]:
        if tok in _NUMBER_VOCAB:
            run.append(tok)
            continue
        if run:
            value = words_to_number(run)
            # Ignore lone small words ("он" is also the Russian pronoun "he").
            if value is not None and (len(run) > 1 or value >= 100):
                out.append(value)
            run = []
    return out


_SUBUNIT = re.compile(r"\b\d{1,2}\s*(?:тиын\w*|коп\w*|cents?)\b", re.I)
_NUM_SPACED = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d{1,2})?(?![\d])")
_NUM_GROUPED = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?(?![\d])")
_NUM_GROUPED_EU = re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?(?![\d])")
_NUM_PLAIN = re.compile(r"(?<![\d])\d+(?:[.,]\d{1,2})?(?![\d])")
_NUM_ABBR = re.compile(r"(\d+(?:[.,]\d+)?)\s*(тыс|млн)\.?", re.I)


def _to_float(s: str) -> float:
    """'18 500,50' -> 18500.5 (space-grouped, comma or dot as decimal)."""
    return float(re.sub(r"[   ]", "", s).replace(",", "."))


def numbers_in_text(text: str, include_words: bool = True) -> set[float]:
    """All numeric values written in the text (digits, OCR-repaired digits, words)."""
    values: set[float] = set()
    text = _SUBUNIT.sub(" ", text)
    for variant in {text, fold_ocr_digits(text)}:
        for m in _NUM_SPACED.finditer(variant):
            values.add(_to_float(m.group(0)))
        for m in _NUM_GROUPED.finditer(variant):
            values.add(float(m.group(0).replace(",", "")))
        for m in _NUM_GROUPED_EU.finditer(variant):
            values.add(float(m.group(0).replace(".", "").replace(",", ".")))
        # Plain numbers only outside grouped ones: "3 150" must not ground 3 or 150.
        rest = _NUM_GROUPED_EU.sub(" ", _NUM_GROUPED.sub(" ", _NUM_SPACED.sub(" ", variant)))
        for m in _NUM_PLAIN.finditer(rest):
            values.add(float(m.group(0).replace(",", ".")))
        for m in _NUM_ABBR.finditer(variant):
            mult = 1_000 if m.group(2).lower() == "тыс" else 1_000_000
            values.add(float(m.group(1).replace(",", ".")) * mult)
    if include_words:
        values.update(float(v) for v in word_numbers_in_text(text))
    return values


def number_is_grounded(value: float, text: str) -> bool:
    return any(abs(value - v) < 0.005 for v in numbers_in_text(text))


def number_is_verbatim(value: float, text: str) -> bool:
    """Written in plain digits, no OCR repair or word parsing (used for confidence)."""
    text = _SUBUNIT.sub(" ", text)
    plain = {_to_float(m.group(0)) for m in _NUM_SPACED.finditer(text)}
    plain |= {_to_float(m.group(0)) for m in _NUM_PLAIN.finditer(_NUM_SPACED.sub(" ", text))}
    plain |= {float(m.group(0).replace(",", "")) for m in _NUM_GROUPED.finditer(text)}
    return any(abs(value - v) < 0.005 for v in plain)


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_MONTHS = {
    # Russian, genitive + nominative
    "января": 1, "январь": 1, "февраля": 2, "февраль": 2, "марта": 3, "март": 3,
    "апреля": 4, "апрель": 4, "мая": 5, "май": 5, "июня": 6, "июнь": 6,
    "июля": 7, "июль": 7, "августа": 8, "август": 8, "сентября": 9, "сентябрь": 9,
    "октября": 10, "октябрь": 10, "ноября": 11, "ноябрь": 11, "декабря": 12, "декабрь": 12,
    # Kazakh
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5, "маусым": 6,
    "шілде": 7, "тамыз": 8, "қыркүйек": 9, "қазан": 10, "қараша": 11, "желтоқсан": 12,
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DATE_WORDS = re.compile(r"(?<!\d)[«\"]?(\d{1,2})[»\"]?\s+([^\W\d_]+)\s+(\d{4})", re.U)


def _safe_date(y: int, m: int, d: int) -> date | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[date, int]]:
    """All dates in the text with their character offset (in document order)."""
    out: list[tuple[date, int]] = []
    folded = fold_ocr_digits(text)
    for m in _DATE_ISO.finditer(folded):
        if (d := _safe_date(int(m[1]), int(m[2]), int(m[3]))):
            out.append((d, m.start()))
    for m in _DATE_NUMERIC.finditer(folded):
        if (d := _safe_date(int(m[3]), int(m[2]), int(m[1]))):
            out.append((d, m.start()))
    for m in _DATE_WORDS.finditer(folded):
        month = _MONTHS.get(m[2].lower())
        if month and (d := _safe_date(int(m[3]), month, int(m[1]))):
            out.append((d, m.start()))
    out.sort(key=lambda x: x[1])
    return out


def date_is_grounded(value: date, text: str) -> bool:
    return any(d == value for d, _ in find_dates(text))
