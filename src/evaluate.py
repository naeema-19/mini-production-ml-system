"""Metric calculations, business-cost evaluation, bootstrap CIs and plots."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from src.config import FN_FIXED_COST, FP_COST


def business_cost(y, pred, amount) -> float:
    """Total EUR cost: missed fraud loses its amount + admin fee; every false alarm costs FP_COST."""
    y, pred, amount = np.asarray(y), np.asarray(pred), np.asarray(amount, dtype=float)
    fn = (y == 1) & (pred == 0)
    fp = (y == 0) & (pred == 1)
    return float((amount[fn] + FN_FIXED_COST).sum() + FP_COST * fp.sum())


def cost_curve(y, scores, amount, thresholds=None):
    if thresholds is None:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.90, 1.0, 801)))
    costs = np.array([business_cost(y, (scores >= t).astype(int), amount) for t in thresholds])
    return thresholds, costs


def best_cost_threshold(y, scores, amount):
    th, costs = cost_curve(y, scores, amount)
    i = int(np.argmin(costs))
    return float(th[i]), float(costs[i])


def best_f1_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def recall_at_precision(y, scores, min_precision=0.8) -> float:
    p, r, _ = precision_recall_curve(y, scores)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def full_report(y, scores, amount, threshold) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cost = business_cost(y, pred, amount)
    no_model_cost = business_cost(y, np.zeros_like(y), amount)  # approve everything
    fraud_amt = float(np.asarray(amount)[y == 1].sum())
    caught_amt = float(np.asarray(amount)[(y == 1) & (pred == 1)].sum())
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "recall_at_p80": recall_at_precision(y, scores, 0.8),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "accuracy": float((pred == y).mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alerts_per_10k": float(1e4 * pred.mean()),
        "cost_eur": cost,
        "no_model_cost_eur": no_model_cost,
        "savings_vs_no_model_pct": float(100 * (1 - cost / no_model_cost)) if no_model_cost else 0.0,
        "fraud_amount_caught_pct": float(100 * caught_amt / fraud_amt) if fraud_amt else 0.0,
    }


def bootstrap_ci(y, scores, amount, threshold, n_boot=1000, seed=0) -> dict:
    """Stratified bootstrap of the test set: with only ~50 frauds, point estimates
    are noisy, so we report 95 % intervals."""
    rng = np.random.default_rng(seed)
    y, scores, amount = map(np.asarray, (y, scores, amount))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pr, rec, prec, cost = [], [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, ss, aa = y[idx], scores[idx], amount[idx]
        pp = (ss >= threshold).astype(int)
        pr.append(average_precision_score(yy, ss))
        rec.append(recall_score(yy, pp, zero_division=0))
        prec.append(precision_score(yy, pp, zero_division=0))
        cost.append(business_cost(yy, pp, aa))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"pr_auc_95ci": q(pr), "recall_95ci": q(rec), "precision_95ci": q(prec), "cost_95ci": q(cost)}
