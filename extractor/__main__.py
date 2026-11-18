"""CLI: python -m extractor path/to/doc.txt [--backend ollama|replay]

Prints the extraction JSON to stdout. Exit code 0 = ok, 2 = extraction failed
(the printed JSON is then the safe all-null fallback with a warning).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import DEFAULT_MODEL, LLMError, make_backend
from .pipeline import extract


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m extractor", description=__doc__)
    p.add_argument("path", help="text file with the document, or '-' for stdin")
    p.add_argument("--backend", default="ollama", choices=["ollama", "replay"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--record", action="store_true", help="save model answers for replay")
    p.add_argument("--no-guards", action="store_true", help="raw model output (for comparison)")
    p.add_argument("-v", "--verbose", action="store_true", help="show raw model attempts on stderr")
    args = p.parse_args(argv)

    if args.path == "-":
        sys.stdin.reconfigure(encoding="utf-8")
        text = sys.stdin.read()
    else:
        text = Path(args.path).read_text(encoding="utf-8")
    try:
        backend = make_backend(args.backend, args.model, record=args.record)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    result = extract(text, backend, guards=not args.no_guards)
    if args.verbose:
        for i, a in enumerate(result.attempts, 1):
            print(f"--- attempt {i}: {'OK' if a['error'] is None else 'FAILED'}", file=sys.stderr)
            print(a["answer"], file=sys.stderr)
            if a["error"]:
                print(a["error"], file=sys.stderr)
    print(json.dumps(result.extraction.to_json_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
