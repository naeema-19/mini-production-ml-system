"""Empirical latency / throughput profiling.

Starts the API in a subprocess (uvicorn, 1 worker), then measures
  1. in-process model-only latency (no HTTP)              - n=1000
  2. sequential /predict latency over HTTP keep-alive      - n=2000
  3. concurrent /predict load at concurrency 1/8/32        - n=2000 each
  4. /predict/batch with 100 records per request           - n=200
  5. fallback (degraded) mode /predict latency             - n=1000
  6. offline batch throughput (records/s) for chunk sizes

Usage:  python -m serving.benchmark [--port 8011] [--n 2000]
Writes reports/benchmark_results.json and reports/latencies_raw.npz
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time

import httpx
import numpy as np
import pandas as pd

from serving.model_service import ModelService
from src.config import RAW_INPUT_COLS, REPORTS_DIR, SPLIT_DIR


def pct(a) -> dict:
    a = np.asarray(a)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def payloads(n: int) -> list[dict]:
    df = pd.read_parquet(SPLIT_DIR / "test.parquet")
    df = df.sample(n=n, replace=len(df) < n, random_state=1)
    return [{"transaction_id": r["transaction_id"], **{c: float(r[c]) for c in RAW_INPUT_COLS}}
            for r in df.to_dict(orient="records")]


def start_server(port: int, env_extra=None):
    env = {**os.environ, "LOG_LEVEL": "WARNING", **(env_extra or {})}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(port),
                          "--workers", "1", "--log-level", "warning", "--no-access-log"], env=env)
    for _ in range(200):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code in (200, 503):
                return p
        except Exception:
            time.sleep(0.1)
    p.kill()
    raise RuntimeError("server did not start")


def seq_bench(url, data, path="/predict", warmup=50):
    lat, errors = [], 0
    with httpx.Client(base_url=url, timeout=5) as c:
        for d in data[:warmup]:
            c.post(path, json=d)
        for d in data:
            t0 = time.perf_counter()
            r = c.post(path, json=d)
            lat.append((time.perf_counter() - t0) * 1000)
            errors += r.status_code != 200
    return lat, errors


async def _conc(url, data, concurrency):
    sem = asyncio.Semaphore(concurrency)
    lat, errors = [], 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=10, limits=limits) as c:
        async def one(d):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await c.post("/predict", json=d)
                lat.append((time.perf_counter() - t0) * 1000)
                errors += r.status_code != 200
        await asyncio.gather(*[one(d) for d in data[:50]])  # warm-up
        lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*[one(d) for d in data])
        wall = time.perf_counter() - t0
    return lat, errors, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    data = payloads(args.n)
    res = {"env": {"python": platform.python_version(), "cpu_count": os.cpu_count(),
                   "machine": platform.machine(), "server": "uvicorn, 1 worker"}}
    raw = {}

    # 1. in-process: reference sklearn pipeline vs compiled fast path (ablation)
    ref = ModelService(); ref.use_fast_path = False; ref.load()
    lat = []
    for d in data[:1000]:
        t0 = time.perf_counter(); ref.score([d]); lat.append((time.perf_counter() - t0) * 1000)
    res["in_process_single_ms_reference_pipeline"] = pct(lat); raw["in_process_ref"] = lat
    svc = ModelService(); svc.load()
    lat = []
    for d in data[:1000]:
        t0 = time.perf_counter(); svc.score([d]); lat.append((time.perf_counter() - t0) * 1000)
    res["in_process_single_ms"] = pct(lat); raw["in_process"] = lat

    proc = start_server(args.port)
    try:
        lat, err = seq_bench(url, data)
        res["http_sequential_ms"] = {**pct(lat), "errors": err, "rps": 1000 / np.mean(lat)}
        raw["http_sequential"] = lat
        res["http_concurrent"] = {}
        for cc in (1, 8, 32):
            lat, err, wall = asyncio.run(_conc(url, data, cc))
            res["http_concurrent"][str(cc)] = {**pct(lat), "errors": err, "throughput_rps": len(data) / wall}
            raw[f"conc_{cc}"] = lat
        batch_payloads = [{"transactions": data[i:i + 100]} for i in range(0, 100 * 200, 100) if i + 100 <= len(data)]
        batch_payloads = (batch_payloads * 10)[:200]
        lat, err = seq_bench(url, batch_payloads, "/predict/batch", warmup=5)
        res["http_batch100_ms"] = {**pct(lat), "errors": err, "records_per_s": 100 * 1000 / np.mean(lat)}
        res["server_side_metrics"] = httpx.get(f"{url}/metrics").json()
    finally:
        proc.terminate(); proc.wait()

    proc = start_server(args.port, {"SIMULATE_MODEL_FAILURE": "1"})
    try:
        lat, err = seq_bench(url, data[:1000])
        sample = httpx.post(f"{url}/predict", json=data[0]).json()
        res["http_fallback_mode_ms"] = {**pct(lat), "errors": err, "health": httpx.get(f"{url}/health").json(),
                                        "sample_scorer": sample["scorer"]}
        raw["fallback"] = lat
    finally:
        proc.terminate(); proc.wait()

    # 6. offline batch throughput
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    X = test[RAW_INPUT_COLS]
    res["batch_throughput"] = {}
    for size in (1_000, 10_000, len(X)):
        chunk = X.iloc[:size]
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); svc.compiled.predict_proba(chunk.to_numpy()); ts.append(time.perf_counter() - t0)
        res["batch_throughput"][str(size)] = {"seconds": float(np.median(ts)), "records_per_s": float(size / np.median(ts))}
    recs = test[["transaction_id"] + RAW_INPUT_COLS].to_dict(orient="records")
    t0 = time.perf_counter(); svc.score(recs); dt = time.perf_counter() - t0
    res["batch_throughput"]["service_score_full_test"] = {"records": len(recs), "seconds": dt, "records_per_s": len(recs) / dt}

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "benchmark_results.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(REPORTS_DIR / "latencies_raw.npz", **{k: np.array(v) for k, v in raw.items()})
    print(json.dumps({k: v for k, v in res.items() if k != "server_side_metrics"}, indent=2))


if __name__ == "__main__":
    main()
