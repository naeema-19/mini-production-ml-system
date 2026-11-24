"""Unit / integration tests for schemas, artifact integrity, API contract and fallback.
Run:  pytest -q
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from serving.schemas import BatchRequest, Transaction
from src.config import METADATA_PATH, MODEL_PATH, PCA_COLS, RAW_INPUT_COLS
from src.features import add_time_features, build_preprocessor


def sample(**kw):
    d = {"transaction_id": "tx_test", "Time": 1000.0, "Amount": 42.5, **{c: 0.1 for c in PCA_COLS}}
    d.update(kw)
    return d


@pytest.fixture(scope="module")
def client():
    from serving.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- schemas
def test_schema_accepts_valid():
    assert Transaction(**sample()).Amount == 42.5


@pytest.mark.parametrize("bad", [{"Amount": -1}, {"Time": -5}, {"Amount": float("inf")},
                                 {"transaction_id": ""}, {"unknown": 1}, {"V1": float("nan")}])
def test_schema_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        Transaction(**sample(**bad))


def test_schema_missing_limit():
    ok = sample(**{c: None for c in PCA_COLS[:10]})
    Transaction(**ok)
    with pytest.raises(ValidationError):
        Transaction(**sample(**{c: None for c in PCA_COLS[:11]}))


def test_batch_limits():
    with pytest.raises(ValidationError):
        BatchRequest(transactions=[])
    with pytest.raises(ValidationError):
        BatchRequest(transactions=[sample()] * 1001)


# ---------------------------------------------------------------- features / leakage
def test_time_feature_is_periodic():
    a = add_time_features(pd.DataFrame([sample(Time=3600.0)])[RAW_INPUT_COLS])
    b = add_time_features(pd.DataFrame([sample(Time=3600.0 + 86400)])[RAW_INPUT_COLS])
    assert np.allclose(a[["hour_sin", "hour_cos"]], b[["hour_sin", "hour_cos"]])
    assert "Time" not in a.columns


def test_preprocessor_statistics_come_from_fit_data_only():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame(rng.normal(size=(500, 30)), columns=RAW_INPUT_COLS).abs()
    te = tr * 100  # very different distribution
    p = build_preprocessor().fit(tr)
    med_before = p.named_steps["columns"].named_transformers_["pca"].named_steps["impute"].statistics_.copy()
    p.transform(te)
    med_after = p.named_steps["columns"].named_transformers_["pca"].named_steps["impute"].statistics_
    assert np.array_equal(med_before, med_after)


# ---------------------------------------------------------------- artifact integrity
def test_artifact_loads_and_scores_deterministically():
    meta = json.loads(METADATA_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([sample()])[meta["raw_input_columns"]]
    p1, p2 = model.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1]
    assert 0 <= p1[0] <= 1 and p1[0] == p2[0]
    assert 0 < meta["decision_threshold"] < 1


def test_known_fraud_scores_higher_than_known_legit():
    test = pd.read_parquet("data/splits/test.parquet")
    model = joblib.load(MODEL_PATH)
    s = model.predict_proba(test[RAW_INPUT_COLS])[:, 1]
    assert s[test.Class.values == 1].mean() > 20 * s[test.Class.values == 0].mean()


# ---------------------------------------------------------------- API
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_contract(client):
    r = client.post("/predict", json=sample())
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "tx_test" and body["scorer"] == "model"
    assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")
    assert "x-request-id" in r.headers


def test_predict_missing_values_imputed_with_warning(client):
    r = client.post("/predict", json=sample(V1=None, V2=None))
    assert r.status_code == 200
    assert any(w.startswith("imputed_missing") for w in r.json()["warnings"])


def test_predict_outlier_flagged(client):
    r = client.post("/predict", json=sample(V14=-1e6))
    assert r.status_code == 200
    assert "out_of_training_range:V14" in r.json()["warnings"]


@pytest.mark.parametrize("payload", [{}, sample(Amount=-3), sample(Amount="abc"), sample(extra=1)])
def test_predict_rejects_malformed(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and r.json()["error"] == "invalid_payload"


def test_batch_preserves_ids_and_order(client):
    txs = [sample(transaction_id=f"id{i}", Amount=float(i)) for i in range(25)]
    r = client.post("/predict/batch", json={"transactions": txs})
    assert r.status_code == 200
    assert [p["transaction_id"] for p in r.json()["predictions"]] == [f"id{i}" for i in range(25)]


def test_fallback_when_model_fails(client):
    from serving.app import service
    service.force_failure = True
    try:
        r = client.post("/predict", json=sample(V14=-15.0))
        assert r.status_code == 200
        body = r.json()
        assert body["scorer"] == "fallback_rule" and body["is_fraud"] is True
        assert body["decision"] == "REVIEW"  # degraded mode never auto-blocks
        assert client.get("/health").json()["status"] == "degraded"
    finally:
        service.force_failure = False


def test_fast_path_parity_on_real_test_data():
    from serving.model_service import ModelService
    svc = ModelService(); svc.load()
    assert svc.compiled is not None
    test = pd.read_parquet("data/splits/test.parquet")
    X = test[RAW_INPUT_COLS]
    assert np.allclose(svc.compiled.predict_proba(X.to_numpy()), svc.model.predict_proba(X)[:, 1], atol=1e-10)


def test_tiered_decisions_on_known_fraud(client):
    test = pd.read_parquet("data/splits/test.parquet")
    fr = test[test.Class == 1]
    txs = [{"transaction_id": r.transaction_id, **{c: float(getattr(r, c)) for c in RAW_INPUT_COLS}}
           for r in fr.itertuples()]
    preds = client.post("/predict/batch", json={"transactions": txs}).json()["predictions"]
    decisions = {p["decision"] for p in preds}
    assert "BLOCK" in decisions and "REVIEW" in decisions
