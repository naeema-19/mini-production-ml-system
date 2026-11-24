"""Batch scoring CLI (near-real-time / offline path).

Usage:
  python -m serving.predict_batch --input data/splits/unseen_sample.csv \
         --output outputs/scored.csv [--chunk-size 50000]

Reads CSV/parquet of unseen transactions, validates the schema, scores in
vectorised chunks with the same artifact the API uses, and writes
transaction_id, fraud_probability, is_fraud, scorer.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from serving.model_service import ModelService

log = logging.getLogger("predict_batch")


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def validate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if "transaction_id" not in df.columns:
        log.warning("No transaction_id column - generating row-based ids")
        df = df.assign(transaction_id=[f"row_{i}" for i in range(len(df))])
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(X[["Time", "Amount"]]).all(axis=1) | (X["Amount"] < 0)
    if bad.any():
        log.warning("%d rows with invalid Time/Amount will be rejected", int(bad.sum()))
    X = X.replace([np.inf, -np.inf], np.nan)
    return df[["transaction_id"]].join(X), bad


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    args = ap.parse_args(argv)

    svc = ModelService()
    svc.load()
    cols = svc.meta["raw_input_columns"]
    t0 = time.perf_counter()
    try:
        df, bad = validate(read_any(args.input), cols)
    except Exception as e:
        log.error("Input validation failed: %s", e)
        return 2
    log.info("Read %d records from %s", len(df), args.input)

    good = df[~bad]
    outs = []
    for start in range(0, len(good), args.chunk_size):
        chunk = good.iloc[start:start + args.chunk_size]
        recs = chunk.to_dict(orient="records")
        res = pd.DataFrame(svc.score(recs))[["transaction_id", "fraud_probability", "is_fraud", "scorer"]]
        outs.append(res)
        log.info("Scored chunk %d-%d", start, start + len(chunk))
    out = pd.concat(outs) if outs else pd.DataFrame(columns=["transaction_id", "fraud_probability", "is_fraud", "scorer"])
    if bad.any():
        rej = pd.DataFrame({"transaction_id": df.loc[bad, "transaction_id"], "fraud_probability": np.nan,
                            "is_fraud": pd.NA, "scorer": "rejected_invalid_input"})
        out = pd.concat([out, rej])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    dt = time.perf_counter() - t0
    log.info("Wrote %d rows to %s | flagged=%d | rejected=%d | %.2fs (%.0f records/s)", len(out), args.output,
             int((out["is_fraud"] == True).sum()), int(bad.sum()), dt, len(df) / dt)  # noqa: E712
    return 0


if __name__ == "__main__":
    sys.exit(main())
