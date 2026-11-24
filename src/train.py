"""Train baselines + LightGBM, tune on VALIDATION, pick a cost-optimal threshold on
VALIDATION, evaluate ONCE on TEST, and export artifacts.

Usage:  python -m src.train
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src import baselines
from src.config import (AMOUNT_COL, ARTIFACT_DIR, BASELINE_LR_PATH, FN_FIXED_COST,
                        FP_COST, METADATA_PATH, MODEL_PATH, RANDOM_STATE,
                        RAW_INPUT_COLS, REFERENCE_PROFILE_PATH, REPORTS_DIR,
                        SPLIT_DIR, TARGET)
from src.evaluate import (best_cost_threshold, best_f1_threshold, bootstrap_ci,
                          full_report)
from src.features import MODEL_FEATURES, add_time_features, build_preprocessor
from monitoring.drift import build_reference_profile

log = logging.getLogger("train")


def load_splits():
    return [pd.read_parquet(SPLIT_DIR / f"{s}.parquet") for s in ("train", "val", "test")]


def fit_lgbm(Xtr, ytr, Xva, yva, params):
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=1500, learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=params["scale_pos_weight"], random_state=RANDOM_STATE,
        n_jobs=2, deterministic=True, force_row_wise=True, verbose=-1)
    clf.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return clf


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    train, val, test = load_splits()
    Xtr_raw, ytr = train[RAW_INPUT_COLS], train[TARGET].to_numpy()
    Xva_raw, yva = val[RAW_INPUT_COLS], val[TARGET].to_numpy()
    Xte_raw, yte = test[RAW_INPUT_COLS], test[TARGET].to_numpy()
    amt_va, amt_te = val[AMOUNT_COL].to_numpy(), test[AMOUNT_COL].to_numpy()

    # ---- Leak-free preprocessing: fit on TRAIN only --------------------------------
    prep = build_preprocessor().fit(Xtr_raw)
    assert prep.named_steps["columns"].named_transformers_["pca"].named_steps["clip"].n_samples_fit_ == len(train)
    Xtr, Xva, Xte = prep.transform(Xtr_raw), prep.transform(Xva_raw), prep.transform(Xte_raw)
    log.info("Preprocessor fitted on %d TRAIN rows only; %d features", len(train), Xtr.shape[1])

    results, val_table = {}, []

    # ---- Baseline 0: prior (dummy) -------------------------------------------------
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    s_va, s_te = dummy.predict_proba(Xva)[:, 1], dummy.predict_proba(Xte)[:, 1]
    results["dummy_prior"] = {"val_pr_auc": float(average_precision_score(yva, s_va)),
                              "test": full_report(yte, s_te, amt_te, threshold=0.5)}

    # ---- Baseline 1: single-feature heuristic rule (also the serving fallback) ----
    rule = baselines.fit_rule(train)
    s_va = baselines.rule_scores(val[rule["feature"]], rule)
    s_te = baselines.rule_scores(test[rule["feature"]], rule)
    rule_thr = (rule["p_if_fired"] + rule["p_if_not_fired"]) / 2
    results["rule_V14"] = {"rule": rule, "val_pr_auc": float(average_precision_score(yva, s_va)),
                           "test": full_report(yte, s_te, amt_te, rule_thr)}
    log.info("Rule baseline: %s", rule)

    # ---- Baseline 2: logistic regression, unweighted vs class-weighted -------------
    lr_models = {}
    for name, cw in [("logreg_unweighted", None), ("logreg_balanced", "balanced")]:
        lr = LogisticRegression(C=0.1, class_weight=cw, max_iter=2000).fit(Xtr, ytr)
        s_va, s_te = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]
        thr, _ = best_cost_threshold(yva, s_va, amt_va)
        results[name] = {"val_pr_auc": float(average_precision_score(yva, s_va)),
                         "test": full_report(yte, s_te, amt_te, thr), "_val_scores": s_va, "_test_scores": s_te}
        lr_models[name] = lr

    # ---- Main model: LightGBM with imbalance handling via loss weighting -----------
    ratio = float((ytr == 0).sum() / (ytr == 1).sum())
    grid = {
        "scale_pos_weight": [1.0, float(round(np.sqrt(ratio), 1)), float(round(ratio / 10, 1)), float(round(ratio, 1))],
        "num_leaves": [15, 31],
        "min_child_samples": [50],
        "learning_rate": [0.05],
    }
    best, best_ap = None, -1
    t0 = time.time()
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        clf = fit_lgbm(Xtr, ytr, Xva, yva, params)
        ap = float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))
        val_table.append({**params, "best_iter": int(clf.best_iteration_), "val_pr_auc": round(ap, 4)})
        log.info("LGBM %s -> val PR-AUC %.4f (iters %d)", params, ap, clf.best_iteration_)
        if ap > best_ap:
            best, best_ap = (params, clf), ap
    log.info("Grid search: %d fits in %.1fs", len(val_table), time.time() - t0)
    best_params, lgbm = best

    s_va, s_te = lgbm.predict_proba(Xva)[:, 1], lgbm.predict_proba(Xte)[:, 1]
    thr_cost, val_cost = best_cost_threshold(yva, s_va, amt_va)
    thr_f1 = best_f1_threshold(yva, s_va)
    results["lightgbm"] = {
        "params": best_params, "best_iteration": int(lgbm.best_iteration_),
        "val_pr_auc": best_ap, "threshold_cost_opt": thr_cost, "threshold_f1_opt": thr_f1,
        "val_at_cost_threshold": full_report(yva, s_va, amt_va, thr_cost),
        "test": full_report(yte, s_te, amt_te, thr_cost),
        "test_at_f1_threshold": full_report(yte, s_te, amt_te, thr_f1),
        "test_at_0.5": full_report(yte, s_te, amt_te, 0.5),
        "test_bootstrap": bootstrap_ci(yte, s_te, amt_te, thr_cost),
    }

    # Imbalance-handling ablation: best config per scale_pos_weight
    abl = (pd.DataFrame(val_table).sort_values("val_pr_auc", ascending=False)
           .groupby("scale_pos_weight").head(1).sort_values("scale_pos_weight"))

    # ---- Export artifacts -------------------------------------------------------------
    pipeline = Pipeline([("preprocess", prep), ("model", lgbm)])
    # integrity check: serialised pipeline must reproduce in-memory scores exactly
    assert np.allclose(pipeline.predict_proba(Xte_raw)[:, 1], s_te)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    joblib.dump(Pipeline([("preprocess", prep), ("model", lr_models["logreg_balanced"])]),
                BASELINE_LR_PATH, compress=3)
    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    profile = build_reference_profile(add_time_features(Xtr_raw), MODEL_FEATURES, val_scores=s_va)
    profile["score"]["alert_rate"] = float((s_va >= thr_cost).mean())
    REFERENCE_PROFILE_PATH.write_text(json.dumps(profile))

    version = "fraud-lgbm-" + datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + model_hash[:8]
    meta = {
        "model_version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_hash, "raw_input_columns": RAW_INPUT_COLS,
        "model_features": MODEL_FEATURES, "decision_threshold": thr_cost,
        "block_threshold": max(thr_f1, thr_cost),  # high-precision tier: auto-decline
        "threshold_policy": f"argmin validation cost; FN = amount + {FN_FIXED_COST} EUR, FP = {FP_COST} EUR",
        "train_fraud_rate": float(ytr.mean()), "lightgbm_params": best_params,
        "best_iteration": int(lgbm.best_iteration_), "fallback_rule": rule,
        "train_medians": {c: float(train[c].median()) for c in RAW_INPUT_COLS},
        # robust range seen in training; values outside are clipped by the pipeline and flagged
        "input_bounds": {c: [float(train[c].quantile(0.0005)), float(train[c].quantile(0.9995))]
                         for c in RAW_INPUT_COLS if c != "Time"},
        "critical_features": [k for k, _ in sorted(
            zip(MODEL_FEATURES, lgbm.booster_.feature_importance("gain")), key=lambda kv: -kv[1])[:5]],
        "test_metrics": {k: results["lightgbm"]["test"][k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")},
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))

    # Scores for figures
    np.savez_compressed(REPORTS_DIR / "scores.npz", y_val=yva, y_test=yte, amt_val=amt_va, amt_test=amt_te,
                        lgbm_val=s_va, lgbm_test=s_te,
                        lr_test=results["logreg_balanced"]["_test_scores"],
                        lr_unw_test=results["logreg_unweighted"]["_test_scores"],
                        rule_test=baselines.rule_scores(test[rule["feature"]], rule))
    for k in ("logreg_unweighted", "logreg_balanced"):
        results[k].pop("_val_scores"); results[k].pop("_test_scores")
    results["grid_search"] = val_table
    results["imbalance_ablation"] = abl.to_dict(orient="records")
    results["feature_importance_gain"] = dict(sorted(
        zip(MODEL_FEATURES, lgbm.booster_.feature_importance("gain").round(1).tolist()),
        key=lambda kv: -kv[1]))
    (REPORTS_DIR / "training_results.json").write_text(json.dumps(results, indent=2))

    log.info("Saved %s (%s)", MODEL_PATH.name, version)
    for k in ("dummy_prior", "rule_V14", "logreg_unweighted", "logreg_balanced", "lightgbm"):
        t = results[k]["test"]
        log.info("%-18s test PR-AUC=%.4f P=%.3f R=%.3f F1=%.3f cost=%.0f", k, t["pr_auc"], t["precision"],
                 t["recall"], t["f1"], t["cost_eur"])


if __name__ == "__main__":
    main()
