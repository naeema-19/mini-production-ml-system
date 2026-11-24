"""Data / prediction drift utilities: Population Stability Index and KS test.

* build_reference_profile(): run at training time; stores decile bin edges and
  expected proportions per feature (from TRAIN) and for the model score (from VAL,
  i.e. out-of-sample scores) in artifacts/reference_profile.json.
* drift_report(): compares a window of production records against that profile.

Usage (CLI):  python -m monitoring.drift --input data/splits/test.parquet
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN, PSI_ALERT = 0.10, 0.25
KS_PVALUE_ALERT = 0.01
MISSING_RATE_ALERT = 0.05
EPS = 1e-6
# cyclic time encodings are *expected* to differ for any window shorter than 24 h
SEASONAL_FEATURES = {"hour_sin", "hour_cos"}


def _edges(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = x[~np.isnan(x)]
    counts, _ = np.histogram(x, bins=edges)
    return counts / max(counts.sum(), 1)


def psi(expected_props, actual_props) -> float:
    e = np.clip(np.asarray(expected_props), EPS, None)
    a = np.clip(np.asarray(actual_props), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_profile(train_X: pd.DataFrame, features: list[str], val_scores=None,
                            sample_size: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prof = {"features": {}, "score": None}
    for f in features:
        x = train_X[f].to_numpy(dtype=float)
        e = _edges(x)
        prof["features"][f] = {
            "edges": [float(v) for v in e[1:-1]],
            "props": [float(v) for v in _proportions(x, e)],
            "sample": [float(v) for v in rng.choice(x, min(sample_size, len(x)), replace=False)],
            "mean": float(np.nanmean(x)), "std": float(np.nanstd(x)),
        }
    if val_scores is not None:
        s = np.asarray(val_scores, dtype=float)
        e = _edges(s, 20)
        prof["score"] = {"edges": [float(v) for v in e[1:-1]],
                         "props": [float(v) for v in _proportions(s, e)],
                         "sample": [float(v) for v in rng.choice(s, min(sample_size, len(s)), replace=False)],
                         "alert_rate": None}
    return prof


def _status(p):
    return "ALERT" if p >= PSI_ALERT else ("WARN" if p >= PSI_WARN else "OK")


def drift_report(profile: dict, window: pd.DataFrame, scores=None, critical_features=None) -> dict:
    """critical_features: the model's top features by importance. Drift there (or in
    the score distribution) triggers retraining; drift in low-importance features
    only opens an investigation ticket, because the PCA components of this dataset
    show genuine day-to-day movement that does not hurt the model."""
    out = {"n_records": int(len(window)), "features": {}, "score": None}
    for f, ref in profile["features"].items():
        if f not in window:
            continue
        x = window[f].to_numpy(dtype=float)
        edges = np.array([-np.inf] + ref["edges"] + [np.inf])
        p = psi(ref["props"], _proportions(x, edges))
        ks = ks_2samp(np.asarray(ref["sample"]), x[~np.isnan(x)])
        out["features"][f] = {"psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                              "ks_pvalue": float(ks.pvalue), "psi_status": _status(p),
                              "missing_rate": float(np.isnan(x).mean())}
    if scores is not None and profile.get("score"):
        ref = profile["score"]
        s = np.asarray(scores, dtype=float)
        edges = np.array([-np.inf] + ref["edges"] + [np.inf])
        p = psi(ref["props"], _proportions(s, edges))
        out["score"] = {"psi": round(p, 4), "psi_status": _status(p)}
    monitored = {k: v for k, v in out["features"].items() if k not in SEASONAL_FEATURES}
    psis = [v["psi"] for v in monitored.values()]
    n_alert = sum(v["psi_status"] == "ALERT" for v in monitored.values())
    dq = [k for k, v in monitored.items() if v["missing_rate"] > MISSING_RATE_ALERT]
    out["summary"] = {
        "max_feature_psi": max(psis) if psis else None,
        "max_psi_feature": max(monitored, key=lambda k: monitored[k]["psi"]) if monitored else None,
        "n_features_warn": sum(v["psi_status"] == "WARN" for v in monitored.values()),
        "n_features_alert": n_alert,
        "data_quality_alerts": dq,
        "score_psi_status": out["score"]["psi_status"] if out["score"] else None,
        "critical_features_alert": [k for k in (critical_features or [])
                                    if monitored.get(k, {}).get("psi_status") == "ALERT"],
        "retrain_recommended": bool(
            any(monitored.get(k, {}).get("psi_status") == "ALERT" for k in (critical_features or []))
            or (out["score"] and out["score"]["psi_status"] == "ALERT")),
        "investigate": bool(n_alert > 0),
        "page_oncall": bool(dq) or bool(out["score"] and out["score"]["psi_status"] == "ALERT"),
    }
    return out


def main():
    from src.config import METADATA_PATH, MODEL_PATH, REFERENCE_PROFILE_PATH
    import joblib
    from src.features import add_time_features

    ap = argparse.ArgumentParser(description="Compute PSI/KS drift of a data window vs the training reference")
    ap.add_argument("--input", required=True, help="CSV or parquet with raw transaction columns")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    profile = json.loads(REFERENCE_PROFILE_PATH.read_text())
    meta = json.loads(METADATA_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(df[meta["raw_input_columns"]])[:, 1]
    rep = drift_report(profile, add_time_features(df), scores, meta.get("critical_features"))
    txt = json.dumps(rep, indent=2)
    if args.output:
        open(args.output, "w").write(txt)
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
