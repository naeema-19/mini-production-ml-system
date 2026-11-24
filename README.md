# Mini Production ML System: Real-Time Card Fraud Detection

An end-to-end fraud scoring system for the ULB/Worldline credit-card dataset (284,807 transactions, 0.173% fraud). It covers a leak-free temporal split, baselines, a cost-sensitive LightGBM model, a FastAPI service with Pydantic validation and a fallback scorer, a batch CLI, latency/throughput benchmarks and PSI/KS drift monitoring.

| | Held-out test set (last 6 h of the stream, 52 frauds) |
|---|---|
| **PR-AUC** | **0.738** (95% bootstrap CI 0.62-0.85) vs 0.692 logistic regression, 0.370 rule baseline |
| **Business cost** | **EUR 2,148**, 69% lower than approving everything (EUR 6,949) |
| **Decision tiers** | BLOCK 40 tx (37 fraud) · REVIEW 46 tx (3 fraud) · APPROVE the rest |
| **API latency** | p50 5.3 ms · p95 6.7 ms · p99 7.6 ms (HTTP, single worker, 2 vCPU) |
| **Batch throughput** | ~15,000 records/s |

## Repository layout

```
mini-production-ml-system/
├── README.md, requirements.txt (pinned), Makefile, Dockerfile
├── data/                  # creditcard.csv is fetched by the pipeline (checksum-verified, >100 MB so not committed)
├── artifacts/             # fraud_pipeline.joblib, model_metadata.json, reference_profile.json, baseline_logreg.joblib
├── src/
│   ├── config.py          # paths, feature lists, split fractions, business cost model
│   ├── data_pipeline.py   # ingestion, validation, temporal split, leakage audit
│   ├── features.py        # preprocessing pipeline (fit on TRAIN only)
│   ├── baselines.py       # interpretable V14 rule (also the serving fallback)
│   ├── train.py           # baselines + LightGBM tuning, threshold selection, export
│   └── evaluate.py        # PR-AUC, cost, recall@precision, bootstrap CIs
├── serving/
│   ├── app.py             # FastAPI: /health /metadata /metrics /predict /predict/batch
│   ├── schemas.py         # Pydantic request/response contracts
│   ├── model_service.py   # artifact loading, scoring, fallback, tiered decisions
│   ├── fast_path.py       # compiled numpy preprocessing (parity-checked at load)
│   ├── predict_batch.py   # batch scoring CLI
│   ├── benchmark.py       # p50/p95/p99 latency + throughput profiling
│   └── robustness_check.py# edge-case payload catalogue
├── monitoring/
│   ├── drift.py           # PSI / KS drift report + CLI
│   └── simulate_drift.py  # natural + synthetic drift scenarios
├── tests/test_service.py  # 25 tests: schemas, leakage, artifact integrity, API, fallback
├── report/                # figure + PDF report generators
├── pytest.ini
└── reports/               # generated JSON results + figures
```

## Quickstart

Requires Python 3.11+ (tested on 3.13).

```bash
git clone <YOUR_REPO_URL> mini-production-ml-system && cd mini-production-ml-system
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_pipeline      # downloads data (SHA-256 checked), temporal split, leakage audit
python -m src.train              # ~3 min on 2 CPUs; writes artifacts/ and reports/training_results.json
python -m pytest -q              # 25 tests
```

If the download is blocked, get `creditcard.csv` from Kaggle (`mlg-ulb/creditcardfraud`) and place it in `data/`. The checksum is still verified.

Or run everything with `make all`.

## Online serving (REST)

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "transaction_id": "tx_demo_1", "Time": 152000, "Amount": 239.93,
  "V1": -2.3, "V2": 1.9, "V3": -1.6, "V4": 3.9, "V5": -0.5, "V6": -1.4, "V7": -2.5,
  "V8": 1.4, "V9": -2.8, "V10": -2.8, "V11": 3.2, "V12": -2.9, "V13": -0.6,
  "V14": -4.3, "V15": 0.4, "V16": -1.1, "V17": -2.8, "V18": -0.02, "V19": 0.4,
  "V20": 0.1, "V21": 0.5, "V22": -0.04, "V23": -0.5, "V24": 0.3, "V25": 0.04,
  "V26": 0.2, "V27": 0.26, "V28": 0.14 }'
```

Example response:

```json
{"transaction_id":"tx_demo_1","fraud_probability":0.99999,"is_fraud":true,"decision":"BLOCK",
 "threshold":0.0036,"scorer":"model","model_version":"fraud-lgbm-...","warnings":[]}
```

**Input contract.** `transaction_id`, `Time` (≥0) and `Amount` (0 to 1e6) are required. `V1`–`V28` are nullable: missing values are imputed with training medians and reported in `warnings`, up to 10 per request. Unknown fields, NaN/Inf, wrong types and oversized batches are rejected with **422** `{"error":"invalid_payload", ...}`.

**Decisions.** `APPROVE`. `REVIEW` when p ≥ the cost-optimal threshold. `BLOCK` when p ≥ the high-precision threshold.

**Degraded mode.** If the model fails, the service falls back to the V14 rule: `scorer:"fallback_rule"`, `/health` reports `degraded`, and nothing is auto-blocked. To simulate it, start the service with `SIMULATE_MODEL_FAILURE=1`.

**Optional settings.** `PRED_LOG_PATH=preds.jsonl` writes a prediction log for drift jobs. `USE_FAST_PATH=0` forces the reference sklearn pipeline.

Docker (optional): `docker build -t fraud-api . && docker run -p 8000:8000 fraud-api`

## Batch scoring

```bash
python -m serving.predict_batch --input data/splits/unseen_sample.csv --output outputs/scored.csv
```

Output columns: `transaction_id, fraud_probability, is_fraud, scorer`. Invalid rows are kept and tagged `rejected_invalid_input`.

## Profiling, robustness, drift

```bash
python -m serving.benchmark          # reports/benchmark_results.json (p50/p95/p99, rps, records/s)
python -m serving.robustness_check   # reports/robustness_results.json
python -m monitoring.simulate_drift  # reports/drift_results.json
python -m monitoring.drift --input data/splits/test.parquet   # drift check for any new window
python -m report.make_figures        # reports/figures/*.png
```

## Reproducing the PDF report

```bash
python -m report.build_report --name "Your Name" --student-id "2025XX00000" --repo-url "https://github.com/<you>/mini-production-ml-system"
```

## Data & references

Dataset: A. Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015. Made available by ULB Machine Learning Group / Worldline (Kaggle: `mlg-ulb/creditcardfraud`, DbCL licence).
Libraries: scikit-learn, LightGBM, FastAPI, Pydantic, SciPy.
