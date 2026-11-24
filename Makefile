PY ?= python

.PHONY: all data train test robustness benchmark drift batch figures report serve clean

all: data train test robustness batch benchmark drift figures

data:        ## download (checksum-verified) + temporal split + leakage audit
	$(PY) -m src.data_pipeline
train:       ## baselines, LightGBM tuning, threshold, artifacts
	$(PY) -m src.train
test:        ## unit + integration tests
	$(PY) -m pytest -q
robustness:  ## edge-case payload catalogue
	$(PY) -m serving.robustness_check
batch:       ## batch CLI on unseen sample
	$(PY) -m serving.predict_batch --input data/splits/unseen_sample.csv --output outputs/scored_unseen_sample.csv
benchmark:   ## p50/p95/p99 latency + throughput
	$(PY) -m serving.benchmark
drift:       ## PSI/KS drift scenarios
	$(PY) -m monitoring.simulate_drift
figures:
	$(PY) -m report.make_figures
report: figures
	$(PY) -m report.build_report --name "$(NAME)" --student-id "$(ID)" --repo-url "$(REPO)"
serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000
clean:
	rm -rf data/splits outputs reports/*.json reports/*.npz reports/figures/*.png
