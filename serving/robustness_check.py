"""Edge-case / robustness evaluation of the inference service.

Fires a catalogue of adversarial and malformed payloads at the API (in-process
TestClient) and records status code, scorer and warnings for each.
Usage:  python -m serving.robustness_check   -> reports/robustness_results.json
"""
from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from src.config import PCA_COLS, RAW_INPUT_COLS, REPORTS_DIR, SPLIT_DIR


def base_payload() -> dict:
    r = pd.read_parquet(SPLIT_DIR / "test.parquet").iloc[0]
    return {"transaction_id": "tx_robust", **{c: float(r[c]) for c in RAW_INPUT_COLS}}


def cases(b: dict):
    def mod(**kw):
        d = dict(b); d.update(kw); return d

    def drop(*keys):
        return {k: v for k, v in b.items() if k not in keys}

    big = dict(b); big.update({c: None for c in PCA_COLS[:12]})
    return [
        ("valid transaction", "/predict", b),
        ("3 PCA features null (imputed)", "/predict", mod(V1=None, V5=None, V20=None)),
        ("12 PCA features null (> limit)", "/predict", big),
        ("PCA feature key absent (treated as null)", "/predict", drop("V3")),
        ("Amount missing", "/predict", drop("Amount")),
        ("Negative Amount", "/predict", mod(Amount=-10.0)),
        ("Extreme Amount 900,000", "/predict", mod(Amount=900_000.0)),
        ("Extreme V14 = -1e6 (clipped)", "/predict", mod(V14=-1e6)),
        ("Amount as string '12.5' (coerced)", "/predict", mod(Amount="12.5")),
        ("Amount as non-numeric string", "/predict", mod(Amount="abc")),
        ("Unknown extra field", "/predict", mod(merchant="XYZ")),
        ("Empty transaction_id", "/predict", mod(transaction_id="")),
        ("Injection-like transaction_id", "/predict", mod(transaction_id="1;DROP TABLE")),
        ("Empty JSON object", "/predict", {}),
        ("Batch of 0", "/predict/batch", {"transactions": []}),
        ("Batch of 1001 (> limit)", "/predict/batch", {"transactions": [b] * 1001}),
        ("Batch of 500 valid", "/predict/batch", {"transactions": [mod(transaction_id=f"t{i}") for i in range(500)]}),
    ]


def main():
    from serving.app import app
    b = base_payload()
    out = []
    with TestClient(app) as c:
        for name, path, payload in cases(b):
            r = c.post(path, json=payload)
            body = r.json()
            row = {"case": name, "status": r.status_code}
            if r.status_code == 200:
                p = body if path == "/predict" else body["predictions"][0]
                row.update(scorer=p["scorer"], prob=round(p["fraud_probability"], 5), warnings=p["warnings"])
            else:
                row["error"] = body.get("error")
                row["msg"] = (body.get("detail") or [{}])[0].get("msg") if isinstance(body.get("detail"), list) else None
            out.append(row)
        # raw malformed JSON
        r = c.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
        out.append({"case": "Malformed JSON body", "status": r.status_code, "error": r.json().get("error")})
        r = c.post("/predict", content=json.dumps(b).replace(json.dumps(b["V1"]), "NaN", 1),
                   headers={"content-type": "application/json"})
        out.append({"case": "NaN literal in V1", "status": r.status_code, "error": r.json().get("error")})
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "robustness_results.json").write_text(json.dumps(out, indent=2))
    for row in out:
        print(row)


if __name__ == "__main__":
    main()
