"""Serving-optimised ("compiled") inference path.

Profiling showed the sklearn ColumnTransformer costs ~6.6 ms per single-row call
(pandas/validation overhead) versus ~0.2 ms for the LightGBM booster itself. This
module extracts the *fitted* parameters of the trained pipeline (imputer medians,
clip bounds, RobustScaler centre/scale) into numpy vectors and applies the same
maths directly. No statistic is re-estimated, so there is no new leakage surface;
parity with the sklearn pipeline is asserted at load time and the service falls
back to the reference pipeline if parity ever fails.
"""
from __future__ import annotations

import numpy as np

from src.config import AMOUNT_COL, PCA_COLS, TIME_COL
from src.features import SECONDS_PER_DAY


class CompiledScorer:
    def __init__(self, pipeline, raw_cols: list[str]):
        ct = pipeline.named_steps["preprocess"].named_steps["columns"]
        pca = ct.named_transformers_["pca"].named_steps
        amt = ct.named_transformers_["amount"].named_steps
        hour = ct.named_transformers_["hour"]
        self.i_time = raw_cols.index(TIME_COL)
        self.i_pca = np.array([raw_cols.index(c) for c in PCA_COLS])
        self.i_amt = raw_cols.index(AMOUNT_COL)
        self.pca_med = pca["impute"].statistics_
        self.pca_lo, self.pca_hi = pca["clip"].lower_, pca["clip"].upper_
        self.pca_c, self.pca_s = pca["scale"].center_, pca["scale"].scale_
        self.amt_med = amt["impute"].statistics_[0]
        self.amt_c, self.amt_s = amt["scale"].center_[0], amt["scale"].scale_[0]
        self.hour_med = hour.statistics_
        self.booster = pipeline.named_steps["model"].booster_
        self.n_iter = pipeline.named_steps["model"].best_iteration_ or None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pca = X[:, self.i_pca]
        pca = np.where(np.isnan(pca), self.pca_med, pca)
        pca = (np.clip(pca, self.pca_lo, self.pca_hi) - self.pca_c) / self.pca_s
        amt = X[:, self.i_amt]
        amt = np.where(np.isnan(amt), self.amt_med, amt)
        amt = (np.log1p(np.clip(amt, 0, None)) - self.amt_c) / self.amt_s
        h = (X[:, self.i_time] % SECONDS_PER_DAY) / 3600.0
        hs, hc = np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)
        hs = np.where(np.isnan(hs), self.hour_med[0], hs)
        hc = np.where(np.isnan(hc), self.hour_med[1], hc)
        return np.column_stack([pca, amt, hs, hc])

    def predict_proba(self, X: np.ndarray, num_threads: int | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # single request: 1 thread (avoids OpenMP spin-up); large batches: all cores
        nt = num_threads if num_threads is not None else (0 if len(X) >= 1000 else 1)
        return self.booster.predict(self.transform(X), num_iteration=self.n_iter, num_threads=nt)
