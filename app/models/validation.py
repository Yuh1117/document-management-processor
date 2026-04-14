from __future__ import annotations

from pydantic import BaseModel


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    message: str | None = None


class ValidationReport(BaseModel):
    doc_id: int
    file_type: str
    checks: list[ValidationCheck]
    overall_passed: bool
