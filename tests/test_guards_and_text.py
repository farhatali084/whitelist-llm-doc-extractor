"""Deterministic parts: text grounding utilities, contradiction rules, eval scoring."""
import json
from datetime import date
from pathlib import Path

import pytest

from eval import FIELDS, classify
from extractor import textnorm as tn
from extractor.guards import apply_guards
from extractor.schema import LLMResponse

ROOT = Path(__file__).resolve().parent.parent
GOLD_FILES = sorted((ROOT / "data/gold").glob("*.json"))


def load(doc_id):
    gold = json.loads((ROOT / f"data/gold/{doc_id}.json").read_text("utf-8"))
    text = (ROOT / f"data/docs/{doc_id}.txt").read_text("utf-8")
    return gold, text


# ---------------------------- text utilities ------------------------------ #

@pytest.mark.parametrize("words,value", [
    ("Үш жүз қырық мың", 340_000),
    ("Один миллион сто шестьдесят тысяч", 1_160_000),
    ("сто шестьдесят пять тысяч пятьсот семнадцать", 165_517),
    ("екі мың жиырма алты", 2026),
])
def test_number_words_ru_kk(words, value):
    assert tn.words_to_number(words.lower().split()) == value


def test_ocr_bin_is_repaired_and_flagged():
    assert tn.find_bins("БСН/БИН: 2l064090Ol80") == {"210640900180": True}
    assert tn.bin_checksum_ok("210640900180")


def test_bin_with_bad_check_digit_is_detected():
    assert not tn.bin_checksum_ok("170340021554")  # BIN from the assignment's document 1


def test_ocr_digits_fold_only_inside_numbers():
    assert tn.fold_ocr_digits("Итого: l22 OOO, дата 0З.11.2О26") == "Итого: 122 000, дата 03.11.2026"
    assert tn.fold_ocr_digits("Оказание услуг") == "Оказание услуг"


def test_dates_in_kazakh_and_russian_words():
    found = [d for d, _ in tn.find_dates("09 қазан 2026 ж. и «31» октября 2026 г.")]
    assert found == [date(2026, 10, 9), date(2026, 10, 31)]


# ---------------------------- contradiction rules ------------------------- #

def test_conflicting_totals_rule_and_warning():
    gold, text = load("08_invoice_conflicting_totals")
    resp = LLMResponse(**{k: gold[k] for k in FIELDS} | {"total_amount": 380000})
    out = apply_guards(resp, text)
    assert out.total_amount == 430800  # rule: 'к оплате' line wins
    assert any(w.startswith("conflicting_totals") for w in out.warnings)
    assert out.field_confidence["total_amount"] <= 0.5


def test_subtotal_plus_vat_is_not_a_contradiction():
    gold, text = load("01_invoice_almaty_trade")
    out = apply_guards(LLMResponse(**{k: gold[k] for k in FIELDS}), text)
    assert not any(w.startswith("conflicting_totals") for w in out.warnings)


def test_conflicting_issue_dates_rule_and_warning():
    gold, text = load("09_offer_kk_ru_eur_two_dates")
    resp = LLMResponse(**{k: gold[k] for k in FIELDS} | {"doc_date": "2026-10-15"})
    out = apply_guards(resp, text)
    assert out.doc_date == date(2026, 10, 14)  # rule: first (header) issue date
    assert any(w.startswith("conflicting_dates") for w in out.warnings)


def test_validity_date_is_replaced_by_issue_date():
    _, text = load("02_offer_digital_solutions")
    # "действительно до 30.11.2026" is a validity date; "Составлено 05.11.2026" is the issue date.
    resp = LLMResponse(doc_type="commercial_offer", doc_date="2026-11-30")
    out = apply_guards(resp, text)
    assert out.doc_date == date(2026, 11, 5)
    assert any(w.startswith("doc_date_relabelled") for w in out.warnings)
    assert not any(w.startswith("conflicting_dates") for w in out.warnings)  # only one *issue* date


# ---------------------------- guards vs gold ------------------------------ #

@pytest.mark.parametrize("gold_path", GOLD_FILES, ids=lambda p: p.stem)
def test_perfect_answer_survives_guards(gold_path):
    """Guards must never destroy a correct answer (otherwise they create omissions)."""
    gold, text = load(gold_path.stem)
    out = apply_guards(LLMResponse(**{k: gold[k] for k in FIELDS}), text).to_json_dict()
    for f in FIELDS:
        assert classify(f, out[f], gold[f], gold.get("accept", {})) == "correct", f
    codes = {w.split(":")[0] for w in out["warnings"]}
    assert set(gold["expected_warnings"]) <= codes


# ---------------------------- eval scoring -------------------------------- #

def test_eval_separates_hallucination_and_omission():
    assert classify("supplier_bin", "123456789012", None, {}) == "hallucination"
    assert classify("supplier_bin", None, "123456789012", {}) == "omission"
    assert classify("supplier_bin", "123456789012", "210640900180", {}) == "wrong"
    assert classify("supplier_bin", None, None, {}) == "correct"
    assert classify("doc_type", "invoice", "unknown", {}) == "hallucination"
    assert classify("line_items", [], [{"name": "x", "qty": None, "price": 1}], {}) == "omission"


def test_eval_name_matching_is_lenient_on_form_not_content():
    assert classify("supplier_name", "ЖШС «Көкжиек Құрылыс»", 'ТОО "Кокжиек Курылыс"', {}) == "correct"
    assert classify("supplier_name", 'TOO "BOCTOK-ЭHEPГO"', 'ТОО "ВОСТОК-ЭНЕРГО"', {}) == "correct"
    assert classify("supplier_name", "ТОО «Береке Маркет»", "ИП Сейткали Б.", {}) == "wrong"
