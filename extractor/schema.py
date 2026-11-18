"""Pydantic schemas.

Two layers on purpose:

* ``LLMResponse`` - what we ask the model to return. Shape and types are strict
  (enums, ISO dates, numbers), but ``supplier_bin`` is a free string because the
  model copies it from (possibly OCR-damaged) text; we repair/verify it ourselves.
  If the model's JSON does not pass this schema, the pipeline retries.

* ``Extraction`` - the final output of the service, exactly the schema from the
  assignment. Built only by our own post-processing code, never directly by the LLM.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocType = Literal["invoice", "commercial_offer", "act", "unknown"]
Currency = Literal["KZT", "USD", "EUR"]


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    qty: Optional[float] = Field(default=None, ge=0)
    price: Optional[float] = Field(default=None, ge=0)


class LLMResponse(BaseModel):
    """Contract for the raw model answer. ``extra='forbid'`` rejects invented keys."""

    model_config = ConfigDict(extra="forbid")

    doc_type: DocType
    supplier_name: Optional[str] = None
    supplier_bin: Optional[str] = None
    doc_number: Optional[str] = None
    doc_date: Optional[date] = None
    currency: Optional[Currency] = None
    total_amount: Optional[float] = Field(default=None, ge=0)
    vat_amount: Optional[float] = Field(default=None, ge=0)
    line_items: list[LineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("supplier_name", "supplier_bin", "doc_number", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        # Models sometimes return "" or "null" instead of null.
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a", "-"}:
            return None
        return v

    @field_validator("doc_date", mode="before")
    @classmethod
    def _strict_iso_date(cls, v):
        # Accept only YYYY-MM-DD strings; reject "12.11.2026", timestamps, etc.
        if v is None or isinstance(v, date):
            return v
        if not isinstance(v, str) or len(v) != 10 or v[4] != "-" or v[7] != "-":
            raise ValueError("doc_date must be a YYYY-MM-DD string or null")
        return v


class Extraction(BaseModel):
    """Final output schema (as specified in the assignment)."""

    model_config = ConfigDict(extra="forbid")

    doc_type: DocType = "unknown"
    supplier_name: Optional[str] = None
    supplier_bin: Optional[str] = Field(default=None, pattern=r"^\d{12}$")
    doc_number: Optional[str] = None
    doc_date: Optional[date] = None
    currency: Optional[Currency] = None
    total_amount: Optional[float] = Field(default=None, ge=0)
    vat_amount: Optional[float] = Field(default=None, ge=0)
    line_items: list[LineItem] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("field_confidence")
    @classmethod
    def _confidence_range(cls, v: dict[str, float]):
        for key, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"confidence for {key} out of [0,1]: {score}")
        return v

    def to_json_dict(self) -> dict:
        return self.model_dump(mode="json")


# Fields of the output that are scalar "facts" taken from the document.
SCALAR_FIELDS = [
    "doc_type",
    "supplier_name",
    "supplier_bin",
    "doc_number",
    "doc_date",
    "currency",
    "total_amount",
    "vat_amount",
]
