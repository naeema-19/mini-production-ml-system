"""Interpretable baselines. The single-feature rule doubles as the serving fallback
(it is pure arithmetic, so it keeps working if the model artifact fails)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def fit_rule(train: pd.DataFrame, feature: str = "V14", target: str = "Class") -> dict:
    """Learn `flag if <feature> < t` on TRAIN, choosing t to maximise F1.
    The rule outputs a calibrated pseudo-probability: the train precision when it
    fires, and the train fraud rate among non-fired rows otherwise."""
    x, y = train[feature].to_numpy(), train[target].to_numpy()
    candidates = np.quantile(x, np.linspace(0.0005, 0.02, 300))
    f1s = [f1_score(y, (x < t).astype(int), zero_division=0) for t in candidates]
    t = float(candidates[int(np.argmax(f1s))])
    fired = x < t
    return {
        "feature": feature,
        "threshold": t,
        "p_if_fired": float(y[fired].mean()),
        "p_if_not_fired": float(y[~fired].mean()),
        "train_f1": float(max(f1s)),
    }


def rule_scores(values, rule: dict) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    fired = np.nan_to_num(v, nan=np.inf) < rule["threshold"]  # missing -> not fired
    return np.where(fired, rule["p_if_fired"], rule["p_if_not_fired"])
