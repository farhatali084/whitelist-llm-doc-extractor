"""Extraction pipeline: prompt -> LLM -> schema validation (+retry) -> guards -> final validation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from .guards import apply_guards, without_guards
from .llm import LLMBackend, LLMError
from .prompt import build_messages, build_retry_messages
from .schema import Extraction, LLMResponse

MAX_ATTEMPTS = 2  # first try + one retry with the validation error fed back


@dataclass
class Result:
    extraction: Extraction            # what the service returns (always schema-valid)
    raw: LLMResponse | None = None    # validated model answer before guards (None if failed)
    ok: bool = True                   # False -> model never produced a valid answer
    attempts: list[dict] = field(default_factory=list)  # log: raw answer + error per try


def _format_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "\n".join(
            f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
        )
    return f"- invalid JSON: {e}"


def _failure(reason: str, attempts: list[dict]) -> Result:
    """Safe fallback: nothing is asserted about the document."""
    return Result(
        extraction=Extraction(doc_type="unknown", warnings=[f"extraction_failed: {reason}"]),
        ok=False,
        attempts=attempts,
    )


def extract(text: str, backend: LLMBackend, guards: bool = True,
            max_attempts: int = MAX_ATTEMPTS) -> Result:
    schema = LLMResponse.model_json_schema()
    messages = build_messages(text)
    attempts: list[dict] = []

    resp: LLMResponse | None = None
    for _ in range(max_attempts):
        try:
            answer = backend.complete(messages, schema)
        except LLMError as e:
            attempts.append({"answer": None, "error": str(e)})
            return _failure(f"LLM unavailable: {e}", attempts)
        try:
            resp = LLMResponse.model_validate(json.loads(answer))
            attempts.append({"answer": answer, "error": None})
            break
        except (json.JSONDecodeError, ValidationError) as e:
            error = _format_error(e)
            attempts.append({"answer": answer, "error": error})
            messages = build_retry_messages(messages, answer, error)

    if resp is None:
        return _failure(
            f"model output failed schema validation {max_attempts} times; last errors:\n"
            f"{attempts[-1]['error']}",
            attempts,
        )

    try:
        out = apply_guards(resp, text) if guards else without_guards(resp)
        # Re-validate the final object explicitly (guards are code too, code has bugs).
        out = Extraction.model_validate(out.model_dump())
    except ValidationError as e:
        return _failure(f"post-processing produced invalid output: {_format_error(e)}", attempts)
    return Result(extraction=out, raw=resp, ok=True, attempts=attempts)
