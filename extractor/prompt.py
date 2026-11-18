"""Prompt for the extraction model."""
from __future__ import annotations

SYSTEM_PROMPT = """You extract fields from business documents from Kazakhstan \
(invoices, commercial offers, acts). Text may be Russian, Kazakh, English or mixed, \
and may contain OCR errors.

Return ONLY a JSON object with these keys:
doc_type: "invoice" (счёт, счёт на оплату, счёт-фактура, шот, шот-фактура), \
"commercial_offer" (коммерческое предложение, коммерциялық ұсыныс), \
"act" (акт выполненных работ / оказанных услуг, актісі), or "unknown" if it is none of these.
supplier_name: the seller/issuer company name exactly as written, or null.
supplier_bin: the supplier's БИН/БСН (12 digits) copied as written, or null. \
Never the buyer's BIN.
doc_number: document number as written (without "№"), or null.
doc_date: the date the document was issued, YYYY-MM-DD, or null. \
Not a validity date, not a due date.
currency: "KZT" (тенге, теңге, тг, ₸), "USD", "EUR", or null if no currency is written.
total_amount: the final amount payable (всего к оплате; if absent - итого / жалпы сомасы). \
If written only in words, convert to a number. null if no total is written.
vat_amount: the VAT (НДС / ҚҚС) amount ONLY if a VAT amount is written. \
null if VAT is not mentioned, "без НДС", or "не облагается".
line_items: list of {name, qty, price}; price is the unit price. Use null for qty/price not written.
warnings: short notes about contradictions (e.g. two different dates or totals) or unreadable parts.

STRICT RULES:
- If a value is not written in the document, use null. Do not guess.
- Do not derive the BIN from the company name. Do not use today's date.
- Do not calculate VAT or totals yourself. Do not sum line items.
"""


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document:\n<<<\n{text.strip()}\n>>>"},
    ]


def build_retry_messages(messages: list[dict], bad_answer: str, error: str) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": bad_answer},
        {
            "role": "user",
            "content": (
                "Your answer failed schema validation:\n"
                f"{error}\n"
                "Return the corrected JSON object only. Use null for anything not in the document."
            ),
        },
    ]
