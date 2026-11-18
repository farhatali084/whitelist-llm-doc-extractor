"""Mini-eval: run the whole test set and print per-field metrics.

    python eval.py                     # replay recorded model answers (no Ollama needed)
    python eval.py --backend ollama    # live local model
    python eval.py --backend ollama --record   # live + refresh recordings

For every field we count separately:
  correct       - matches gold (null == null counts as correct)
  hallucination - field filled, gold is null        (costly: wrong data enters accounting)
  omission      - field null, gold has a value      (cheaper: a human fills it in)
  wrong         - both filled, values differ
Each metric is shown for the raw model output and for the full pipeline (with guards).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from extractor import textnorm as tn
from extractor.llm import DEFAULT_MODEL, LLMError, make_backend
from extractor.pipeline import extract
from extractor.guards import without_guards
from extractor.schema import SCALAR_FIELDS

ROOT = Path(__file__).resolve().parent
DOCS, GOLD = ROOT / "data" / "docs", ROOT / "data" / "gold"
FIELDS = SCALAR_FIELDS + ["line_items"]


# ------------------------------ comparison --------------------------------- #

def _is_empty(fld: str, v) -> bool:
    return v is None or (fld == "doc_type" and v == "unknown") or (fld == "line_items" and not v)


def _names_match(pred: str, gold: str, accepted: list[str] = ()) -> bool:
    p = tn.normalize_name(pred)
    for g in [gold, *accepted]:
        g = tn.normalize_name(g)
        if p == g or SequenceMatcher(None, p, g).ratio() >= 0.8:
            return True
        # Model may keep a bilingual name ("Сорғы / Насос X"): accept if gold words all inside.
        if g and all(t in p for t in g.split()):
            return True
    return False


def _num_eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 0.01


def _values_equal(fld: str, pred, gold, accept: dict) -> bool:
    if fld == "supplier_name":
        return _names_match(pred, gold, accept.get("supplier_name", []))
    if fld == "doc_number":
        key = lambda s: tn.fold_homoglyphs(tn.fold_ocr_digits(s))
        return key(pred) == key(gold)
    if fld in ("total_amount", "vat_amount"):
        return _num_eq(pred, gold)
    if fld == "line_items":
        if len(pred) != len(gold):
            return False
        alt = accept.get("line_items.name", [])
        return all(
            _names_match(p["name"], g["name"], alt) and _num_eq(p["qty"], g["qty"]) and _num_eq(p["price"], g["price"])
            for p, g in zip(pred, gold)
        )
    return pred == gold


def classify(fld: str, pred, gold, accept: dict) -> str:
    pe, ge = _is_empty(fld, pred), _is_empty(fld, gold)
    if pe and ge:
        return "correct"
    if ge:
        return "hallucination"
    if pe:
        return "omission"
    return "correct" if _values_equal(fld, pred, gold, accept) else "wrong"


# ------------------------------ running ------------------------------------ #

@dataclass
class Tally:
    counts: dict = field(default_factory=lambda: {f: {"correct": 0, "hallucination": 0, "omission": 0, "wrong": 0} for f in FIELDS})
    errors: list = field(default_factory=list)  # (doc, field, outcome, pred, gold)

    def add(self, doc: str, pred: dict, gold: dict) -> None:
        for f in FIELDS:
            outcome = classify(f, pred[f], gold[f], gold.get("accept", {}))
            self.counts[f][outcome] += 1
            if outcome != "correct":
                self.errors.append((doc, f, outcome, pred[f], gold[f]))

    def total(self, outcome: str) -> int:
        return sum(c[outcome] for c in self.counts.values())


def run(backend_name: str, model: str, record: bool) -> dict:
    backend = make_backend(backend_name, model, record=record)
    raw, guarded = Tally(), Tally()
    failures, retries, warn_hits, warn_total, per_doc = 0, 0, 0, 0, []
    gold_files = sorted(GOLD.glob("*.json"))
    for gp in gold_files:
        doc_id = gp.stem
        gold = json.loads(gp.read_text("utf-8"))
        text = (DOCS / f"{doc_id}.txt").read_text("utf-8")
        t0 = time.perf_counter()
        res = extract(text, backend)
        dt = time.perf_counter() - t0
        out = res.extraction.to_json_dict()
        raw_out = without_guards(res.raw).to_json_dict() if res.raw else out
        raw.add(doc_id, raw_out, gold)
        guarded.add(doc_id, out, gold)
        failures += not res.ok
        retries += max(0, len(res.attempts) - 1)
        codes = {w.split(":", 1)[0] for w in out["warnings"]}
        for code in gold.get("expected_warnings", []):
            warn_total += 1
            warn_hits += code in codes
        per_doc.append({"doc": doc_id, "ok": res.ok, "attempts": len(res.attempts),
                        "seconds": round(dt, 1), "output": out})
        print(f"  {doc_id:<34} attempts={len(res.attempts)} ok={res.ok} {dt:5.1f}s", file=sys.stderr)
    return {"n": len(gold_files), "raw": raw, "guarded": guarded, "failures": failures,
            "retries": retries, "warn_hits": warn_hits, "warn_total": warn_total, "per_doc": per_doc}


# ------------------------------ report ------------------------------------- #

def print_report(r: dict, backend: str, model: str) -> None:
    n, raw, gd = r["n"], r["raw"], r["guarded"]
    print(f"\nEval: {n} documents | backend={backend} | model={model}\n")
    hdr = f"{'field':<14} | {'accuracy':>8} {'halluc':>6} {'omiss':>5} {'wrong':>5} || {'accuracy':>8} {'halluc':>6} {'omiss':>5} {'wrong':>5}"
    print(f"{'':<14} | {'raw model output':^28} || {'pipeline (with guards)':^28}")
    print(hdr)
    print("-" * len(hdr))
    for f in FIELDS:
        a, b = raw.counts[f], gd.counts[f]
        print(f"{f:<14} | {a['correct'] / n:>8.0%} {a['hallucination']:>6} {a['omission']:>5} {a['wrong']:>5} || "
              f"{b['correct'] / n:>8.0%} {b['hallucination']:>6} {b['omission']:>5} {b['wrong']:>5}")
    print("-" * len(hdr))
    cells = n * len(FIELDS)
    for name, t in (("raw", raw), ("pipeline", gd)):
        print(f"{name:<9} overall accuracy {t.total('correct') / cells:.1%} | "
              f"HALLUCINATIONS {t.total('hallucination')} | OMISSIONS {t.total('omission')} | "
              f"wrong values {t.total('wrong')}")
    print(f"schema-validation retries: {r['retries']} | failed documents (fallback output): {r['failures']}")
    if r["warn_total"]:
        print(f"expected contradiction/OCR warnings raised: {r['warn_hits']}/{r['warn_total']}")
    if gd.errors:
        print("\nPipeline errors by document:")
        for doc, f, outcome, pred, gold in gd.errors:
            print(f"  {doc:<34} {f:<13} {outcome:<13} got={json.dumps(pred, ensure_ascii=False)[:70]}  "
                  f"gold={json.dumps(gold, ensure_ascii=False)[:70]}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Run the extraction eval")
    p.add_argument("--backend", default="replay", choices=["replay", "ollama"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--record", action="store_true", help="with --backend ollama: save answers for replay")
    p.add_argument("--json", type=Path, default=ROOT / "reports" / "eval_report.json")
    args = p.parse_args()
    try:
        r = run(args.backend, args.model, args.record)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print_report(r, args.backend, args.model)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "backend": args.backend, "model": args.model, "n": r["n"],
        "raw": r["raw"].counts, "pipeline": r["guarded"].counts,
        "failures": r["failures"], "retries": r["retries"], "documents": r["per_doc"],
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nfull report: {args.json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
