"""Model loading and scoring, decoupled from the web framework.

Artifacts are loaded once (at startup) from artifacts/. Scoring is vectorised and
wrapped in a fallback: if the model raises (corrupt artifact, OOM, library bug)
or is not loaded, the pure-arithmetic V14 rule stored in model_metadata.json is
used, and responses are tagged scorer="fallback_rule" so downstream consumers
and dashboards can see the degradation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from serving.fast_path import CompiledScorer
from src.baselines import rule_scores
from src.config import METADATA_PATH, MODEL_PATH, PCA_COLS

log = logging.getLogger("model_service")


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path, self.metadata_path = Path(model_path), Path(metadata_path)
        self.model = None
        self.compiled: CompiledScorer | None = None
        self.use_fast_path = os.getenv("USE_FAST_PATH", "1") == "1"
        self.meta: dict = {}
        self.started = time.time()
        self.force_failure = os.getenv("SIMULATE_MODEL_FAILURE", "0") == "1"
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "records": 0, "fallback_records": 0, "model_errors": 0}

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        self.meta = json.loads(self.metadata_path.read_text())
        try:
            self.model = joblib.load(self.model_path)
            self.model.named_steps["model"].set_params(n_jobs=1)  # no thread spin-up per request
            self._warmup()
            if self.use_fast_path:
                self._build_fast_path()
            log.info("Loaded model %s", self.meta["model_version"])
        except Exception:  # keep serving with the fallback rule
            log.exception("Model load failed - serving in DEGRADED mode (fallback rule)")
            self.model = None

    def _build_fast_path(self):
        """Compile the fitted preprocessing into numpy and verify parity with the
        reference sklearn pipeline on a synthetic batch incl. NaNs and outliers."""
        cols = self.meta["raw_input_columns"]
        rng = np.random.default_rng(0)
        med = np.array([self.meta["train_medians"][c] for c in cols])
        X = med + rng.normal(scale=3.0, size=(512, len(cols)))
        X[:, cols.index("Time")] = rng.uniform(0, 2e5, 512)
        X[:, cols.index("Amount")] = rng.exponential(100, 512)
        X[rng.random(X.shape) < 0.05] = np.nan
        X[:, cols.index("Time")] = np.nan_to_num(X[:, cols.index("Time")], nan=10.0)
        X[0, 5], X[1, 14] = 1e6, -1e6
        try:
            comp = CompiledScorer(self.model, cols)
            ref = self.model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]
            if np.allclose(comp.predict_proba(X), ref, rtol=1e-7, atol=1e-10):
                self.compiled = comp
                log.info("Fast path enabled (parity verified on %d rows)", len(X))
            else:
                log.warning("Fast path parity FAILED - using reference sklearn pipeline")
        except Exception:
            log.exception("Fast path build failed - using reference sklearn pipeline")

    def _warmup(self):
        row = {c: self.meta["train_medians"][c] for c in self.meta["raw_input_columns"]}
        self.model.predict_proba(pd.DataFrame([row]))

    @property
    def threshold(self) -> float:
        return float(self.meta["decision_threshold"])

    @property
    def version(self) -> str:
        return self.meta.get("model_version", "unknown")

    @property
    def fallback_rule(self) -> dict:
        return self.meta["fallback_rule"]

    def health(self) -> dict:
        status = "ok" if self.model is not None and not self.force_failure else (
            "degraded" if self.meta.get("fallback_rule") else "unavailable")
        return {"status": status, "model_loaded": self.model is not None,
                "model_version": self.version if self.meta else None,
                "fallback_available": bool(self.meta.get("fallback_rule")),
                "fast_path": self.compiled is not None,
                "uptime_s": round(time.time() - self.started, 1)}

    # ------------------------------------------------------------------ scoring
    def _warnings(self, df: pd.DataFrame) -> list[list[str]]:
        out = [[] for _ in range(len(df))]
        miss = df[PCA_COLS].isna().to_numpy()
        for i in np.where(miss.any(axis=1))[0]:
            cols = [PCA_COLS[j] for j in np.where(miss[i])[0]]
            out[i].append(f"imputed_missing:{','.join(cols)}")
        prof = self.meta.get("input_bounds")
        if prof:
            for c, (lo, hi) in prof.items():
                v = df[c].to_numpy(dtype=float)
                for i in np.where((v < lo) | (v > hi))[0]:
                    out[i].append(f"out_of_training_range:{c}")
        return out

    def score(self, records: list[dict]) -> list[dict]:
        cols = self.meta["raw_input_columns"]
        df = pd.DataFrame.from_records(records, columns=["transaction_id"] + cols)
        X = df[cols].astype(float)
        scorer = "model"
        try:
            if self.model is None or self.force_failure:
                raise RuntimeError("model unavailable")
            if self.compiled is not None:
                probs = self.compiled.predict_proba(X.to_numpy())
            else:
                probs = self.model.predict_proba(X)[:, 1]
            if not np.all(np.isfinite(probs)):
                raise FloatingPointError("non-finite model output")
            thr = self.threshold
            block = float(self.meta.get("block_threshold", 1.01))
        except Exception as e:
            if self.model is not None and not self.force_failure:
                log.exception("Model scoring failed; falling back to rule")
                with self._lock:
                    self.stats["model_errors"] += 1
            rule = self.fallback_rule
            probs = rule_scores(X[rule["feature"]].to_numpy(), rule)
            thr = (rule["p_if_fired"] + rule["p_if_not_fired"]) / 2
            block = 1.01  # degraded mode never auto-declines: everything flagged goes to human review
            scorer = "fallback_rule"
        warns = self._warnings(X)
        with self._lock:
            self.stats["requests"] += 1
            self.stats["records"] += len(df)
            if scorer != "model":
                self.stats["fallback_records"] += len(df)
        return [
            {"transaction_id": tid, "fraud_probability": float(p), "is_fraud": bool(p >= thr),
             "decision": "BLOCK" if p >= block else ("REVIEW" if p >= thr else "APPROVE"), "threshold": float(thr),
             "scorer": scorer, "model_version": self.version if scorer == "model" else f"rule:{self.fallback_rule['feature']}",
             "warnings": w}
            for tid, p, w in zip(df["transaction_id"], probs, warns)
        ]
