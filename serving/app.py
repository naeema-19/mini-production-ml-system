"""FastAPI online scoring service.

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8000
Endpoints:
  GET  /health          liveness + model status (ok | degraded | unavailable)
  GET  /metadata        model version, threshold, features
  GET  /metrics         in-process counters and rolling latency percentiles
  POST /predict         score one transaction
  POST /predict/batch   score up to 1000 transactions
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.model_service import ModelService
from serving.schemas import (BatchRequest, BatchResponse, HealthResponse,
                             Prediction, Transaction)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
PRED_LOG_PATH = os.getenv("PRED_LOG_PATH")  # optional JSONL prediction log for drift monitoring

service = ModelService()
latencies_ms: deque[float] = deque(maxlen=10_000)
status_counts: dict[int, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()   # model loading is separated from request handling
    yield


app = FastAPI(title="Real-time Card Fraud Scoring API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error rid=%s", rid)
        response = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    dt = (time.perf_counter() - t0) * 1000
    if request.url.path.startswith("/predict"):
        latencies_ms.append(dt)
    status_counts[response.status_code] = status_counts.get(response.status_code, 0) + 1
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = f"{dt:.3f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid_payload", "detail": errors,
                                                  "request_id": getattr(request.state, "request_id", None)})


def _log_predictions(txs, preds):
    if not PRED_LOG_PATH:
        return
    with open(PRED_LOG_PATH, "a") as f:
        for t, p in zip(txs, preds):
            f.write(json.dumps({"ts": time.time(), **t, "p": p["fraud_probability"], "scorer": p["scorer"]}) + "\n")


def _score(txs: list[Transaction]):
    records = [t.model_dump() for t in txs]
    preds = service.score(records)
    _log_predictions(records, preds)
    return preds


@app.get("/health", response_model=HealthResponse)
def health():
    h = service.health()
    return JSONResponse(status_code=200 if h["status"] != "unavailable" else 503, content=h)


@app.get("/metadata")
def metadata():
    m = service.meta
    return {k: m.get(k) for k in ("model_version", "trained_at_utc", "decision_threshold", "threshold_policy",
                                  "raw_input_columns", "test_metrics", "fallback_rule")}


@app.get("/metrics")
def metrics():
    lat = np.array(latencies_ms) if latencies_ms else np.array([np.nan])
    return {**service.stats, "status_codes": status_counts,
            "latency_ms": {"p50": float(np.nanpercentile(lat, 50)), "p95": float(np.nanpercentile(lat, 95)),
                           "p99": float(np.nanpercentile(lat, 99)), "n": len(latencies_ms)}}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    if not service.meta:
        return JSONResponse(status_code=503, content={"error": "service_not_ready"})
    return _score([tx])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not service.meta:
        return JSONResponse(status_code=503, content={"error": "service_not_ready"})
    t0 = time.perf_counter()
    preds = _score(req.transactions)
    return {"predictions": preds, "n_records": len(preds), "latency_ms": (time.perf_counter() - t0) * 1000}
