"""Drift monitoring demonstration.

Scenarios compared against the training reference profile:
  A. validation window   (natural, next ~5 h)
  B. test window         (natural, last ~6 h)
  C. synthetic covariate shift on test: Amount x3 and V14 shifted by -1.5 sd
     (e.g. a new merchant category / attack campaign)
  D. synthetic upstream pipeline bug: 30 % of V4 set to missing, V12 sign-flipped
Also: hourly PSI on the score distribution and label-delayed rolling PR-AUC
(concept-drift proxy).

Usage:  python -m monitoring.simulate_drift  -> reports/drift_results.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from monitoring.drift import build_reference_profile, drift_report
from src.features import MODEL_FEATURES
from src.config import (METADATA_PATH, MODEL_PATH, RAW_INPUT_COLS,
                        REFERENCE_PROFILE_PATH, REPORTS_DIR, SPLIT_DIR, TARGET)
from src.features import add_time_features


def main():
    profile = json.loads(REFERENCE_PROFILE_PATH.read_text())
    meta = json.loads(METADATA_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    rng = np.random.default_rng(0)
    crit = meta["critical_features"]
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    train_hour = ((train["Time"] % 86400) // 3600).astype(int)

    def hour_matched_profile(window: pd.DataFrame) -> dict:
        """Seasonality-aware reference: TRAIN rows from the same hours-of-day as the window."""
        hrs = set(((window["Time"] % 86400) // 3600).astype(int))
        ref = train[train_hour.isin(hrs)]
        p = build_reference_profile(add_time_features(ref[RAW_INPUT_COLS]), MODEL_FEATURES,
                                    val_scores=model.predict_proba(ref[RAW_INPUT_COLS])[:, 1])
        return p

    shifted = test.copy()
    shifted["Amount"] *= 3.0
    shifted["V14"] -= 1.5 * profile["features"]["V14"]["std"]
    broken = test.copy()
    broken.loc[rng.random(len(broken)) < 0.3, "V4"] = np.nan
    broken["V12"] = -broken["V12"]

    out = {}
    for name, d in [("A_validation_natural", val), ("B_test_natural", test),
                    ("C_test_covariate_shift", shifted), ("D_test_pipeline_bug", broken)]:
        s = model.predict_proba(d[RAW_INPUT_COLS])[:, 1]
        rep_global = drift_report(profile, add_time_features(d[RAW_INPUT_COLS]), s, crit)
        rep = drift_report(hour_matched_profile(d), add_time_features(d[RAW_INPUT_COLS]), s, crit)
        top = sorted(((k, v) for k, v in rep["features"].items() if not k.startswith("hour")),
                     key=lambda kv: -kv[1]["psi"])[:5]
        out[name] = {"summary": rep["summary"], "summary_global_reference": rep_global["summary"],
                     "score_psi": rep["score"],
                     "alert_rate_pct": float(100 * (s >= meta["decision_threshold"]).mean()),
                     "top_features": {k: v for k, v in top},
                     "all_feature_psi": {k: v["psi"] for k, v in rep["features"].items()}}

    # hourly windows over val+test (the "live" period)
    live = pd.concat([val, test]).sort_values("Time")
    live["hour"] = (live["Time"] // 3600).astype(int)
    s_live = model.predict_proba(live[RAW_INPUT_COLS])[:, 1]
    live["score"] = s_live
    hourly = []
    for h, g in live.groupby("hour"):
        rep = drift_report(hour_matched_profile(g), add_time_features(g[RAW_INPUT_COLS]), g["score"].to_numpy(), crit)
        hourly.append({"hour": int(h), "n": int(len(g)), "frauds": int(g[TARGET].sum()),
                       "score_psi": rep["score"]["psi"], "max_feature_psi": rep["summary"]["max_feature_psi"],
                       "amount_psi": rep["features"]["Amount"]["psi"], "v14_psi": rep["features"]["V14"]["psi"],
                       "retrain": rep["summary"]["retrain_recommended"],
                       "alert_rate_pct": float(100 * (g["score"] >= meta["decision_threshold"]).mean())})
    out["hourly_live"] = hourly

    # label-delayed rolling PR-AUC (3-hour windows) - concept drift proxy
    roll = []
    for start in range(int(live.hour.min()), int(live.hour.max()) - 1, 1):
        g = live[(live.hour >= start) & (live.hour < start + 3)]
        if g[TARGET].sum() >= 5:
            roll.append({"window_start_h": start, "frauds": int(g[TARGET].sum()),
                         "pr_auc": float(average_precision_score(g[TARGET], g["score"]))})
    out["rolling_pr_auc_3h"] = roll

    # genuine day-over-day shift at identical hours of day (12:00-18:00)
    allx = pd.concat([train, val, test])
    hod = ((allx["Time"] % 86400) // 3600).astype(int)
    day = (allx["Time"] // 86400).astype(int)
    out["day_over_day_median_12_18h"] = {
        c: {f"day{d + 1}": float(v) for d, v in allx[hod.between(12, 18)].groupby(day[hod.between(12, 18)])[c].median().items()}
        for c in ("V1", "V3", "V14", "Amount")}
    (REPORTS_DIR / "drift_results.json").write_text(json.dumps(out, indent=2))
    for k in ("A_validation_natural", "B_test_natural", "C_test_covariate_shift", "D_test_pipeline_bug"):
        print(k, "GLOBAL:", out[k]["summary_global_reference"]["n_features_alert"], "| MATCHED:", out[k]["summary"], "score", out[k]["score_psi"], "alert%", round(out[k]["alert_rate_pct"], 3),
              list(out[k]["top_features"].keys())[:3])
    print("hourly score psi:", [(h["hour"], h["score_psi"], h["amount_psi"]) for h in hourly])
    print("rolling:", [(r["window_start_h"], r["frauds"], round(r["pr_auc"], 3)) for r in roll])


if __name__ == "__main__":
    main()
