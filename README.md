# LLM document extractor (Task A — LLM Engineer)

A CLI that takes the text of a business document from Kazakhstan (invoice, commercial
offer, act; Russian / Kazakh / English, possibly OCR-damaged) and returns JSON in the
fixed schema from the assignment. The model is used only to *read* the document; the
code then checks every value against the source text. **A value that is not written in
the document is returned as `null`.**

## Quick start (from a clean folder)

Requirements: Python 3.10+ and git. Ollama is **not** needed for the eval (see "LLM choice").

**Windows (PowerShell)**
```powershell
git clone <repo-url> whitelist-llm-doc-extractor
cd whitelist-llm-doc-extractor
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe eval.py          # eval on recorded real model answers
.\.venv\Scripts\python.exe -m pytest -q     # unit tests
```

**Linux / macOS**
```bash
git clone <repo-url> whitelist-llm-doc-extractor
cd whitelist-llm-doc-extractor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python eval.py          # or: make eval  (with the venv activated)
.venv/bin/python -m pytest -q     # or: make test
```

### Running with the live model

```bash
# 1. install Ollama: https://ollama.com/download  (Windows: winget install Ollama.Ollama)
ollama pull qwen2.5:3b                 # ~1.9 GB, runs on CPU
# 2. eval with the live model (~1 min per document on a laptop CPU)
python eval.py --backend ollama
# 3. a single document
python -m extractor data/docs/03_invoice_kokzhiek_kk_ocr.txt
python -m extractor data/docs/03_invoice_kokzhiek_kk_ocr.txt -v          # + raw model attempts on stderr
python -m extractor data/docs/03_invoice_kokzhiek_kk_ocr.txt --no-guards # raw model output, for comparison
cat some_doc.txt | python -m extractor -                                 # stdin
```

Exit code: `0` success, `2` extraction failed. On failure, the printed JSON is a safe all-null fallback with a warning.

### Environment variables (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `EXTRACTOR_MODEL` | `qwen2.5:3b` | Ollama model name (also selects the recordings file) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

## How it works

```
text ──► prompt ──► LLM (Ollama, JSON-schema constrained, temperature 0)
                         │
                         ▼
         pydantic LLMResponse ── invalid? ──► retry once, with the validation errors fed back
                         │                          └── still invalid ──► all-null output + "extraction_failed" warning, exit 2
                         ▼
         guards (deterministic, no LLM): every value must be grounded in the text
                         │   • not found          → null + warning
                         │   • contradictions     → rule from the table below + warning
                         │   • field_confidence   → computed by code, not reported by the model
                         ▼
         pydantic Extraction (final schema) ── re-validated ──► JSON
```

| File | Role |
|---|---|
| [extractor/schema.py](extractor/schema.py) | `LLMResponse` (contract for the raw model answer, `extra="forbid"`) and `Extraction` (final schema, BIN `^\d{12}$`) |
| [extractor/prompt.py](extractor/prompt.py) | System prompt and retry message |
| [extractor/llm.py](extractor/llm.py) | `OllamaBackend`, `RecordingBackend`, `ReplayBackend` |
| [extractor/pipeline.py](extractor/pipeline.py) | Validate → retry → fallback → guards → final validation |
| [extractor/guards.py](extractor/guards.py) | Per-field grounding and contradiction rules |
| [extractor/textnorm.py](extractor/textnorm.py) | OCR repair, BIN check digit, number words (ru/kk), dates (ru/kk/en), name matching |
| [eval.py](eval.py) | Mini-eval |
| [data/docs](data/docs), [data/gold](data/gold) | 11 documents and their gold JSON |
| [data/recordings](data/recordings) | Recorded real model answers for replay |
| [tests/](tests) | pytest |

### What happens when the output fails validation

This is requirement 3. It is shown in [tests/test_pipeline.py](tests/test_pipeline.py) and is visible with `-v`.

1. The model's answer is parsed as JSON and validated against `LLMResponse`. Wrong enum values (`"currency": "тенге"`), non-ISO dates (`"12.11.2026"`), strings in number fields and extra keys are all rejected.
2. **Retry once.** The bad answer and a field-by-field list of errors are sent back to the model, which is asked to return corrected JSON.
3. **Still invalid → safe fallback.** The output is `doc_type: "unknown"`, all fields `null`, and the warning `extraction_failed: ...` with the last errors. The CLI exits with code `2`, so a calling system can route the document to a human. Ollama being down is handled the same way, without crashing.
4. After the guards run, the final object is **validated again** against `Extraction`. The guards are code too, and code can have bugs.

Ollama's `format: <json schema>` (constrained decoding) makes invalid output rare, but the system does not rely on it. "The model usually answers correctly" is not the safety net; validation is.

## LLM choice

I had no paid API key, so I use **Ollama with `qwen2.5:3b`**, running locally on a laptop CPU. I chose Qwen 2.5 because it handles Russian well and Kazakh tolerably, and the 3B size fits a CPU in about 1 minute per document. Generation uses temperature 0 and seed 42.

**Record and replay.** Reviewers may not want to install Ollama and download 2 GB, so `eval.py --backend ollama --record` saves every real model answer to [data/recordings/qwen2.5_3b.json](data/recordings/qwen2.5_3b.json), keyed by a hash of (model + full prompt). The default `python eval.py` replays these answers through the **same, real pipeline**: validation, retry, guards and scoring. Nothing is mocked except the network call, and the answers are genuine model outputs. If the prompt or a document changes, the key no longer matches and the replay fails loudly instead of reusing a stale answer.

## Rules for ambiguous cases

| Field | Rule |
|---|---|
| any field | Not written in the document → `null`. "Written" may mean after OCR repair (`2l064090Ol80` → `210640900180`) or in words (`Үш жүз қырық мың` → 340000), but never computed, inferred or defaulted. |
| `doc_type` | Decided by the **title** in the first lines (`Счёт`, `Шот-фактура`, `Коммерческое предложение`, `Акт ...`, `...актісі`). If the model disagrees with the title, the title wins and a `doc_type_conflict` warning is added. With no title, the model's answer is kept only if the document contains that type's keyword; otherwise `unknown`. Document 3 (`Шот-фактура № AKT-09`) is therefore an **invoice**: the title is authoritative and "AKT-09" is just a number. |
| `supplier_bin` | Must occur in the text, either exactly or after folding OCR look-alikes (O→0, l/I→1, З→3) next to digits. When a document has two BINs (seller and buyer), the model chooses; when it has exactly one and the model returned null, that one is used (confidence 0.7). The **Kazakhstan check digit** is verified: a failing BIN is **kept as written**, with confidence 0.4 and a `supplier_bin_checksum` warning. Note that the BIN in the assignment's document 1 (`170340021554`) fails the check digit. |
| `doc_date` | The **issue date**: a date labelled `от / Дата / Күні / Составлено`, or on the same line as `№`. Validity and due dates (`действительно до`) are not issue dates; if the model picks one, it is replaced by the labelled issue date. **Two different issue dates → the first one (document header) + a `conflicting_dates` warning, confidence 0.5.** |
| `total_amount` | The final **amount payable**. Labels by priority: `к оплате / төлеуге` > `всего / барлығы` > `итого / жалпы сомасы / total`; within the highest priority, the last occurrence. `Итого 160 500` with `Всего к оплате 186 180` is *not* a contradiction when the difference equals the stated VAT. **Totals that disagree otherwise → the rule above + a `conflicting_totals` warning, confidence 0.5.** If no total is written (document 2), the result is `null`: line items are **not** summed. |
| `vat_amount` | Only a number written on a VAT line (`НДС`, `ҚҚС`, `VAT`), excluding percentages and "с НДС" totals. `Без НДС`, `НДС не облагается`, `Без учета НДС` → `null`, not `0`. `null` means "no VAT amount is written", and the reason is visible in `warnings`. VAT is never calculated. |
| `currency` | Must be supported by a currency word or sign in the text (`тенге/теңге/тг/₸`, `USD/$`, `EUR/€/евро`). No currency mentioned → `null`. |
| `doc_number` | Without `№`. `AKT-09 / 09 қазан` is trimmed to `AKT-09`; OCR digits are folded (`2O7` → `207`); it must occur in the text. |
| `line_items` | `price` is the **unit price**. An item whose name is not in the text is dropped; a qty or price not in the text becomes `null` (document 2: "Разработка сайта — 450 000" has `qty: null`, not 1). |
| `field_confidence` | Computed by code, only for non-null fields: 1.0 = verbatim in the text; 0.8 = after normalisation (OCR repair, words, Latin/Cyrillic look-alikes); 0.7 = filled by a rule when the model returned null; 0.5 = resolved contradiction; 0.4 = BIN with a failing check digit. The model's self-reported confidence is not used; small models are poorly calibrated. |
| `warnings` | Machine-readable `code: message` lines from the guards, plus the model's own notes prefixed with `model:`. |

## Test set

There are 11 documents: the 3 given in the assignment and 8 of my own, 6 of which are hard. Each has a gold file in [data/gold](data/gold), which also lists the warnings the pipeline is expected to raise (`expected_warnings`) and acceptable alternative spellings of names (`accept`).

| # | Document | What it tests |
|---|---|---|
| 01 | given: invoice | subtotal + VAT + total payable; BIN with an invalid check digit |
| 02 | given: commercial offer | no BIN, number, VAT or total; validity date vs. issue date |
| 03 | given: kk/ru invoice | OCR BIN, Kazakh date, total only in Kazakh words, "Шот-фактура" vs "AKT-09", VAT exempt |
| 04 | act | **two BINs** (seller vs buyer), `«31» октября`, VAT included |
| 05 | invoice in English, USD | `2,400.00` number format, `без НДС` |
| 06 | **hard: missing fields** | blank `№ ____ от ____`, no prices, total, currency or BIN; "5 банковских дней" as bait |
| 07 | **hard: amounts in words** | total and VAT only in Russian words |
| 08 | **hard: conflicting totals** | 380 000 + 60 800 ≠ 430 800 |
| 09 | **hard: mixed kk/ru, EUR** | bilingual lines, two different issue dates, `Без учета НДС` |
| 10 | **hard: heavy OCR** | Latin/Cyrillic mix (`BOCTOK`, `Cчeт`), `O/0`, `l/1`, `З/3`, damaged `БИН` label |
| 11 | **hard: not a document** | an email *asking* for an invoice; everything must be null / unknown |

## Eval

```
python eval.py                    # recorded answers (default)
python eval.py --backend ollama   # live
```

For every field, the eval counts **correct** (including null == null), **hallucination** (filled, gold null), **omission** (null, gold has a value) and **wrong** (both filled, different). Each count is shown twice: for the **raw model output** and for the **full pipeline**, so the effect of the guards is visible. It also reports schema-validation retries, fallbacks, and how many expected contradiction/OCR warnings were raised. A JSON report is written to `reports/eval_report.json`.

Comparison is strict on values and lenient only on form. Names are compared after removing quotes and legal forms (ТОО/ЖШС/LLP) and folding Kazakh letters and Latin look-alikes. Amounts use ±0.01. Line items must match in count, order, qty and price.

### Results (qwen2.5:3b, CPU)

Output of `python eval.py`:

```
               |       raw model output       ||    pipeline (with guards)
field          | accuracy halluc omiss wrong || accuracy halluc omiss wrong
---------------------------------------------------------------------------
doc_type       |      91%      1     0     0 ||     100%      0     0     0
supplier_name  |      91%      1     0     0 ||      91%      1     0     0
supplier_bin   |      91%      0     1     0 ||     100%      0     0     0
doc_number     |      91%      0     0     1 ||     100%      0     0     0
doc_date       |      64%      0     3     1 ||     100%      0     0     0
currency       |      82%      2     0     0 ||     100%      0     0     0
total_amount   |     100%      0     0     0 ||     100%      0     0     0
vat_amount     |      91%      0     0     1 ||     100%      0     0     0
line_items     |      73%      0     0     3 ||      73%      0     0     3
---------------------------------------------------------------------------
raw       overall accuracy 85.9% | HALLUCINATIONS 4 | OMISSIONS 4 | wrong values 6
pipeline  overall accuracy 96.0% | HALLUCINATIONS 1 | OMISSIONS 0 | wrong values 3
schema-validation retries: 0 | failed documents (fallback output): 0
expected contradiction/OCR warnings raised: 5/5
```

What the guards caught in real model output:
- On 06 and 11, the model returned `KZT` although the document mentions no currency: dropped.
- On 07, it returned VAT = 60 000 (invented): dropped, and the real 160 000, written in words on the VAT line, was used instead.
- On 03, 07 and 09, it missed the issue date (Kazakh date, word date) or picked the wrong one: fixed by the issue-date rule.
- On 11, it called an email asking for an invoice an `invoice`: now `unknown`.
- On 09, it read `2 шт × 3 150 EUR` as qty 3, price 150: both nulled, because pieces of `3 150` do not count as grounded.

What is still wrong:
- **1 hallucination:** in 11 (the email), `supplier_name` is the sender's company. That name *is* in the text, so grounding cannot catch it; this is a semantic error (buyer vs supplier).
- **3 wrong `line_items`:** the model missed the qty "20 бобин" (06), merged "40 шт." into the name (07), and the 3/150 error above was nulled but not corrected (09).

**Honest caveats:**
- 11 documents is a tiny sample: one error moves a field by 9 percentage points.
- I developed the guard rules while looking at these documents, so the pipeline numbers are optimistic. A held-out set is needed for an unbiased estimate (see REFLECTION.md).
- `qwen2.5:3b` never produced invalid JSON here, because Ollama constrains decoding to the schema. That is why the retry and fallback path shows `0` in the eval; it is exercised by the tests instead.

Example of the fallback output when the model's answer fails validation twice:

```json
{
  "doc_type": "unknown", "supplier_name": null, "supplier_bin": null, "doc_number": null,
  "doc_date": null, "currency": null, "total_amount": null, "vat_amount": null,
  "line_items": [], "field_confidence": {},
  "warnings": ["extraction_failed: model output failed schema validation 2 times; last errors:\n- doc_type: Input should be 'invoice', 'commercial_offer', 'act' or 'unknown'"]
}
```

## Tests

```
python -m pytest -q
```

The 36 tests cover:
- validation failure → retry with error feedback → fallback, including when Ollama is down
- each guard against a specific hallucination: invented BIN, today's date, computed VAT, summed total, invented line item, VAT on a VAT-exempt document
- the contradiction rules
- OCR repair, number words and dates
- the eval's hallucination/omission classification
- a "perfect model" test for every document: correct answers must survive the guards unchanged (guards that destroy correct values would only turn hallucinations into omissions)

## Done / not done

**Done**
- CLI with a fixed output schema, two-layer pydantic validation, retry with error feedback, and a safe fallback
- Real local LLM (Ollama, qwen2.5:3b) plus record/replay, so the eval runs without Ollama
- Deterministic grounding for every field; contradiction rules for dates and totals; computed `field_confidence`
- OCR repair, BIN check digit, Russian/Kazakh number words and dates
- 11-document test set (8 of mine, 6 of them hard) with gold JSON; eval with hallucinations and omissions counted separately, raw vs pipeline
- 38 unit tests

**Not done**
- No HTTP service (CLI only; the assignment allows either).
- No buyer/supplier disambiguation beyond the prompt, which causes the remaining hallucination.
- Line items are only checked, not repaired. A model error inside an item becomes `null`, not the correct value.
- Not tested on real scanned documents or a larger set; no held-out set.
- Only one model was tried. I did not compare `qwen2.5:7b` or other models, because of CPU time.

**Next steps**
1. A held-out test set of 30–50 real documents, written before touching the rules again.
2. Supplier-side detection: label words such as `Поставщик/Исполнитель/Жеткізуші` vs `Покупатель/Заказчик/Сатып алушы` next to names and BINs.
3. A per-field "send to a human" threshold on `field_confidence`, and measuring how often it fires.
4. Compare `qwen2.5:7b`, and a hosted model, on the same eval.
