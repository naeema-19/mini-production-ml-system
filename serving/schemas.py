"""Pydantic v2 request/response contracts for the fraud scoring API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.config import MAX_BATCH_SIZE, MAX_MISSING_FEATURES, PCA_COLS

# V1..V28 are nullable floats (missing -> imputed with train median, warning returned).
# NaN / +-Inf are rejected by allow_inf_nan=False.
_pca_fields = {c: (Optional[float], Field(default=None, description=f"PCA component {c}")) for c in PCA_COLS}

_TransactionBase = create_model(
    "_TransactionBase",
    __config__=ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True),
    transaction_id=(str, Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-:.]+$")),
    Time=(float, Field(..., ge=0, description="Seconds elapsed since start of capture window")),
    Amount=(float, Field(..., ge=0, le=1_000_000, description="Transaction amount (EUR)")),
    **_pca_fields,
)


class Transaction(_TransactionBase):
    @model_validator(mode="after")
    def _limit_missing(self):
        n_missing = sum(getattr(self, c) is None for c in PCA_COLS)
        if n_missing > MAX_MISSING_FEATURES:
            raise ValueError(
                f"{n_missing} of {len(PCA_COLS)} PCA features are null; at most {MAX_MISSING_FEATURES} allowed")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    threshold: float
    scorer: Literal["model", "fallback_rule"]
    model_version: str
    warnings: list[str] = []


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n_records: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    model_loaded: bool
    model_version: Optional[str]
    fallback_available: bool
    fast_path: bool = False
    uptime_s: float


class ErrorResponse(BaseModel):
    error: str
    detail: object = None
    request_id: Optional[str] = None
