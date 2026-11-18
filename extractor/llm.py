"""LLM backends.

* ``OllamaBackend``  - real local model via the Ollama HTTP API (structured output).
* ``RecordingBackend`` - wraps a real backend and saves every answer to a JSON file.
* ``ReplayBackend``  - answers from that file. Lets anyone re-run the *real* pipeline
  and eval on *real* model outputs without installing Ollama. A prompt change
  changes the cache key, so stale recordings are detected instead of silently reused.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import requests

DEFAULT_MODEL = os.environ.get("EXTRACTOR_MODEL", "qwen2.5:3b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "recordings"


class LLMError(RuntimeError):
    """The model could not be called at all (network, missing recording...)."""


class LLMBackend(Protocol):
    model: str

    def complete(self, messages: list[dict], schema: dict) -> str: ...


def cache_key(model: str, messages: list[dict]) -> str:
    payload = json.dumps({"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def recordings_path(model: str) -> Path:
    return RECORDINGS_DIR / (model.replace(":", "_").replace("/", "_") + ".json")


class OllamaBackend:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 600):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def complete(self, messages: list[dict], schema: dict) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,  # Ollama constrains decoding to this JSON schema
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama call failed ({self.host}, model {self.model}): {e}") from e
        return r.json()["message"]["content"]


class RecordingBackend:
    def __init__(self, inner: LLMBackend, path: Path | None = None):
        self.inner, self.model = inner, inner.model
        self.path = path or recordings_path(inner.model)
        self._data = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}

    def complete(self, messages: list[dict], schema: dict) -> str:
        answer = self.inner.complete(messages, schema)
        self._data[cache_key(self.model, messages)] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")
        return answer


class ReplayBackend:
    def __init__(self, model: str = DEFAULT_MODEL, path: Path | None = None):
        self.model = model
        self.path = path or recordings_path(model)
        if not self.path.exists():
            raise LLMError(f"No recordings at {self.path}. Run with --backend ollama --record first.")
        self._data = json.loads(self.path.read_text("utf-8"))

    def complete(self, messages: list[dict], schema: dict) -> str:
        key = cache_key(self.model, messages)
        if key not in self._data:
            raise LLMError(
                f"No recorded answer for this prompt (key {key}). The prompt or document "
                "changed since recording; re-record with --backend ollama --record."
            )
        return self._data[key]


def make_backend(name: str, model: str = DEFAULT_MODEL, record: bool = False) -> LLMBackend:
    if name == "ollama":
        backend: LLMBackend = OllamaBackend(model)
        return RecordingBackend(backend) if record else backend
    if name == "replay":
        return ReplayBackend(model)
    raise ValueError(f"unknown backend {name!r} (use 'ollama' or 'replay')")
