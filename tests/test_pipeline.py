"""Pipeline behaviour with a scripted fake LLM: validation, retry, fallback, guards."""
import json
from datetime import date
from pathlib import Path

import pytest

from extractor.llm import LLMError
from extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent
DOC1 = (ROOT / "data/docs/01_invoice_almaty_trade.txt").read_text("utf-8")
DOC2 = (ROOT / "data/docs/02_offer_digital_solutions.txt").read_text("utf-8")


class FakeBackend:
    """Returns pre-scripted answers in order and records the prompts it saw."""

    model = "fake"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen: list[list[dict]] = []

    def complete(self, messages, schema):
        self.seen.append(messages)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)


def answer(**overrides):
    base = {
        "doc_type": "invoice", "supplier_name": 'ТОО "Алматы Трейд Компани"',
        "supplier_bin": "170340021554", "doc_number": "1245", "doc_date": "2026-11-12",
        "currency": "KZT", "total_amount": 186180, "vat_amount": 25680,
        "line_items": [{"name": "Бумага А4 80г/м2", "qty": 50, "price": 2100}],
        "warnings": [],
    }
    base.update(overrides)
    return base


# ---------------------- validation / retry / fallback ---------------------- #

def test_invalid_output_is_retried_with_error_feedback():
    fake = FakeBackend("not json at all", answer())
    res = extract(DOC1, fake)
    assert res.ok and len(res.attempts) == 2
    retry_prompt = fake.seen[1][-1]["content"]
    assert "failed schema validation" in retry_prompt and "invalid JSON" in retry_prompt
    assert res.extraction.total_amount == 186180


def test_schema_violations_are_reported_field_by_field():
    bad = answer(doc_date="12.11.2026", currency="тенге", invented_field=1)
    fake = FakeBackend(bad, answer())
    res = extract(DOC1, fake)
    err = res.attempts[0]["error"]
    assert "doc_date" in err and "currency" in err and "invented_field" in err
    assert res.ok


def test_persistent_invalid_output_falls_back_to_all_null():
    fake = FakeBackend(answer(doc_type="receipt"), answer(total_amount="a lot"))
    res = extract(DOC1, fake)
    out = res.extraction
    assert not res.ok
    assert out.doc_type == "unknown" and out.supplier_bin is None and out.total_amount is None
    assert out.warnings[0].startswith("extraction_failed")


def test_llm_unavailable_falls_back_without_crash():
    res = extract(DOC1, FakeBackend(LLMError("connection refused")))
    assert not res.ok and "LLM unavailable" in res.extraction.warnings[0]


# ---------------------- hallucination guards ------------------------------ #

def test_bin_not_in_document_is_dropped():
    # Model "invents" a BIN for Digital Solutions (not present in document 2).
    res = extract(DOC2, FakeBackend(answer(doc_type="commercial_offer", supplier_name="Digital Solutions",
                                           supplier_bin="180540012343", doc_date="2026-11-05",
                                           total_amount=None, vat_amount=None, line_items=[])))
    assert res.extraction.supplier_bin is None
    assert any(w.startswith("supplier_bin_dropped") for w in res.extraction.warnings)


def test_todays_date_is_not_accepted():
    res = extract(DOC2, FakeBackend(answer(doc_type="commercial_offer", supplier_name="Digital Solutions",
                                           supplier_bin=None, doc_date="2026-11-18",
                                           total_amount=None, vat_amount=None, line_items=[])))
    # 2026-11-18 is not in the text; the labelled issue date is used instead.
    assert res.extraction.doc_date == date(2026, 11, 5)
    assert any(w.startswith("doc_date_dropped") for w in res.extraction.warnings)


def test_computed_vat_and_summed_total_are_dropped():
    # Document 2 has no VAT and no total; model computes both (510 000 and 16% of it).
    res = extract(DOC2, FakeBackend(answer(doc_type="commercial_offer", supplier_name="Digital Solutions",
                                           supplier_bin=None, doc_date="2026-11-05",
                                           total_amount=510000, vat_amount=81600, line_items=[])))
    assert res.extraction.total_amount is None
    assert res.extraction.vat_amount is None


def test_vat_exempt_document_keeps_vat_null():
    doc = "Счёт на оплату № 1 от 01.11.2026\nТОО «Тест»\nИтого: 100 000 тенге\nБез НДС"
    res = extract(doc, FakeBackend(answer(supplier_name="ТОО «Тест»", supplier_bin=None, doc_number="1",
                                          doc_date="2026-11-01", total_amount=100000, vat_amount=12000,
                                          line_items=[])))
    assert res.extraction.vat_amount is None
    assert any("no VAT" in w for w in res.extraction.warnings)


def test_invented_line_item_and_price_are_removed():
    items = [{"name": "Бумага А4 80г/м2", "qty": 50, "price": 2099},   # price not in doc
             {"name": "Доставка курьером", "qty": 1, "price": 5000}]    # item not in doc
    res = extract(DOC1, FakeBackend(answer(line_items=items)))
    li = res.extraction.line_items
    assert len(li) == 1 and li[0].price is None and li[0].qty == 50


def test_model_picking_subtotal_is_corrected_to_amount_payable():
    res = extract(DOC1, FakeBackend(answer(total_amount=160500)))
    assert res.extraction.total_amount == 186180
    assert any(w.startswith("total_relabelled") for w in res.extraction.warnings)


def test_output_always_matches_final_schema():
    res = extract(DOC1, FakeBackend(answer()))
    data = res.extraction.to_json_dict()
    assert set(data) == {"doc_type", "supplier_name", "supplier_bin", "doc_number", "doc_date", "currency",
                         "total_amount", "vat_amount", "line_items", "field_confidence", "warnings"}
    assert all(0 <= v <= 1 for v in data["field_confidence"].values())
