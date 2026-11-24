"""Ingestion, validation, leak-free temporal splitting and leakage audit.

Usage:  python -m src.data_pipeline
"""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (AMOUNT_COL, DATA_SHA256, DATA_URL, PCA_COLS, RAW_CSV,
                        RAW_INPUT_COLS, REPORTS_DIR, SPLIT_DIR, TARGET, TIME_COL,
                        TRAIN_FRAC, VAL_FRAC)

log = logging.getLogger("data_pipeline")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_data(force: bool = False) -> None:
    """Download the dataset if absent and verify its checksum."""
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not force:
        log.info("Dataset present at %s", RAW_CSV)
    else:
        log.info("Downloading dataset from %s", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, RAW_CSV)
    digest = sha256(RAW_CSV)
    if digest != DATA_SHA256:
        raise RuntimeError(f"Checksum mismatch for {RAW_CSV}: {digest}")
    log.info("Checksum OK (%s...)", digest[:12])


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    expected = RAW_INPUT_COLS + [TARGET]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[expected]
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary 0/1")
    if (df[AMOUNT_COL] < 0).any():
        raise ValueError("Negative amounts present")
    # Stable transaction id = position in the original stream
    df.insert(0, "transaction_id", [f"tx_{i:06d}" for i in range(len(df))])
    return df


def temporal_split(df: pd.DataFrame):
    """Strict time-cutoff split: train = earliest 70 %, val = next 15 %, test = latest 15 %.
    Models are deployed on *future* traffic, so evaluation must be on the future too."""
    df = df.sort_values([TIME_COL, "transaction_id"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    i_train, i_val = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    t_val_cut, t_test_cut = df.loc[i_train, TIME_COL], df.loc[i_val, TIME_COL]
    # cut on timestamps (not row index) so no second is shared by two partitions
    train = df[df[TIME_COL] < t_val_cut]
    val = df[(df[TIME_COL] >= t_val_cut) & (df[TIME_COL] < t_test_cut)]
    test = df[df[TIME_COL] >= t_test_cut]
    return train.copy(), val.copy(), test.copy(), (float(t_val_cut), float(t_test_cut))


def leakage_audit(train, val, test) -> dict:
    feats = PCA_COLS + [AMOUNT_COL]
    audit = {}
    audit["time_ordering_ok"] = bool(train[TIME_COL].max() < val[TIME_COL].min()
                                     and val[TIME_COL].max() < test[TIME_COL].min())
    audit["id_overlap"] = int(len(set(train.transaction_id) & set(val.transaction_id))
                              + len(set(train.transaction_id) & set(test.transaction_id))
                              + len(set(val.transaction_id) & set(test.transaction_id)))

    def keys(d):
        return set(pd.util.hash_pandas_object(d[feats].round(6), index=False))

    k_train = keys(train)
    audit["val_rows_with_identical_train_features"] = int(
        pd.util.hash_pandas_object(val[feats].round(6), index=False).isin(k_train).sum())
    audit["test_rows_with_identical_train_features"] = int(
        pd.util.hash_pandas_object(test[feats].round(6), index=False).isin(k_train).sum())
    audit["train_exact_duplicates_removed"] = int(train.duplicated(subset=feats + [TIME_COL, TARGET]).sum())

    # Single-feature AUC screen on TRAIN: a feature with AUC ~1.0 would indicate
    # a target proxy (leak). Report the max.
    aucs = {}
    for c in feats + [TIME_COL]:
        a = roc_auc_score(train[TARGET], train[c])
        aucs[c] = round(max(a, 1 - a), 4)
    top = sorted(aucs.items(), key=lambda kv: -kv[1])[:5]
    audit["top_single_feature_auc_train"] = top
    audit["suspicious_proxy_features(auc>0.99)"] = [c for c, a in aucs.items() if a > 0.99]
    return audit


def split_summary(parts: dict) -> list[dict]:
    rows = []
    for name, d in parts.items():
        rows.append({
            "split": name, "rows": int(len(d)), "frauds": int(d[TARGET].sum()),
            "fraud_rate_pct": round(100 * d[TARGET].mean(), 4),
            "time_start_h": round(d[TIME_COL].min() / 3600, 2),
            "time_end_h": round(d[TIME_COL].max() / 3600, 2),
        })
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fetch_data()
    df = load_raw()
    log.info("Loaded %d rows, %d frauds (%.3f%%)", len(df), df[TARGET].sum(), 100 * df[TARGET].mean())
    train, val, test, cuts = temporal_split(df)
    audit = leakage_audit(train, val, test)
    # Dedupe only inside TRAIN (duplicates over-weight some points); val/test stay
    # untouched so they reflect real traffic.
    feats = PCA_COLS + [AMOUNT_COL, TIME_COL, TARGET]
    train = train.drop_duplicates(subset=feats)
    assert audit["time_ordering_ok"] and audit["id_overlap"] == 0

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
    # An "unseen" sample file for the batch CLI demo (labels stripped)
    test.drop(columns=[TARGET]).head(5000).to_csv(SPLIT_DIR / "unseen_sample.csv", index=False)

    summary = {
        "n_rows_raw": int(len(df)), "n_frauds_raw": int(df[TARGET].sum()),
        "duplicate_rows_raw": int(df.drop(columns="transaction_id").duplicated().sum()),
        "time_cutoffs_seconds": cuts,
        "splits": split_summary({"train": train, "val": val, "test": test}),
        "leakage_audit": audit,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Split summary: %s", json.dumps(summary["splits"], indent=1))
    log.info("Leakage audit: %s", json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
