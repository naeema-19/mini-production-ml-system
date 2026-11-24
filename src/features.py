"""Feature engineering + preprocessing pipeline.

Everything stateful (imputer medians, scaler centres/scales, clipping bounds) lives
inside a scikit-learn transformer that is fitted on the TRAIN partition only and
then serialised together with the model, so training and serving share exactly the
same code path (no training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.config import AMOUNT_COL, PCA_COLS, TIME_COL

SECONDS_PER_DAY = 86_400.0


def add_time_features(X: pd.DataFrame) -> pd.DataFrame:
    """Replace raw `Time` (seconds since the first transaction of the capture
    window - meaningless in production and a look-ahead hazard) with a cyclic
    hour-of-day encoding, which generalises to any future day."""
    X = X.copy()
    t = pd.to_numeric(X[TIME_COL], errors="coerce")
    hour = (t % SECONDS_PER_DAY) / 3600.0
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return X.drop(columns=[TIME_COL])


def log_amount(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [q_low, q_high] learnt on training data. Protects the
    model from extreme / adversarial numeric outliers at inference time."""

    def __init__(self, q_low: float = 0.0005, q_high: float = 0.9995):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.q_low, axis=0)
        self.upper_ = np.nanquantile(X, self.q_high, axis=0)
        self.n_samples_fit_ = X.shape[0]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor() -> Pipeline:
    pca_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper()),
        ("scale", RobustScaler()),
    ])
    amount_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_amount, feature_names_out="one-to-one")),
        ("scale", RobustScaler()),
    ])
    hour_branch = SimpleImputer(strategy="median")

    ct = ColumnTransformer(
        [
            ("pca", pca_branch, PCA_COLS),
            ("amount", amount_branch, [AMOUNT_COL]),
            ("hour", hour_branch, ["hour_sin", "hour_cos"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("time_features", FunctionTransformer(add_time_features)),
        ("columns", ct),
    ])


MODEL_FEATURES = PCA_COLS + [AMOUNT_COL, "hour_sin", "hour_cos"]
