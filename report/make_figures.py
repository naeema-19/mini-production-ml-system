"""Generate all report figures from the JSON/NPZ outputs in reports/.
Usage:  python -m report.make_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve

from src.config import FIG_DIR, REPORTS_DIR, SPLIT_DIR
from src.evaluate import cost_curve

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e6e5e0", "surface": "#fcfcfb"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlecolor": C["ink"], "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.facecolor": "white"})


def load(name):
    return json.loads((REPORTS_DIR / name).read_text())


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(-4, 48); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, sub="", color=C["blue"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold", color=C["ink"])
        if sub:
            ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="top", fontsize=6.8, color=C["ink2"], linespacing=1.3)

    def arrow(x1, y1, x2, y2, label="", dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                     color=C["ink2"], linestyle="--" if dashed else "-"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, label, ha="center", fontsize=6.5, color=C["ink2"])

    ax.text(1, 46.5, "OFFLINE  (training pipeline)", fontsize=8, weight="bold", color=C["orange"])
    ax.text(1, 21.5, "ONLINE / BATCH  (serving)", fontsize=8, weight="bold", color=C["blue"])
    ax.plot([0, 100], [23.5, 23.5], color=C["grid"], lw=1)
    o = C["orange"]
    box(1, 29, 15, 15, "Ingest", "fetch + SHA-256\nschema checks\ncreditcard.csv", o)
    box(20, 29, 17, 15, "Temporal split", "train 70% | val 15%\n| test 15% by Time\nleakage audit", o)
    box(41, 29, 17, 15, "Preprocess (fit\non TRAIN only)", "\nimpute, clip, log,\nRobustScaler, hour", o)
    box(62, 29, 17, 15, "Train + tune", "baselines, LightGBM\nloss weighting; tune\n& threshold on VAL", o)
    box(83, 29, 16, 15, "Artifacts", "pipeline.joblib\nmetadata.json\nreference_profile", o)
    for x1, x2 in [(16.8, 19.2), (37.8, 40.2), (58.8, 61.2), (79.8, 82.2)]:
        arrow(x1, 36.5, x2, 36.5)

    b = C["blue"]
    box(1, 4, 14, 14, "Clients", "card switch /\nauth service\n(JSON POST)", b)
    box(19, 4, 18, 14, "FastAPI", "Pydantic schema\n422 on bad payload\nrequest-id, timing", b)
    box(41, 4, 19, 14, "ModelService", "compiled fast path\n(parity-checked)\nfallback: V14 rule", b)
    box(64, 4, 16, 14, "Batch CLI", "predict_batch.py\nCSV/parquet in\nscored CSV out", b)
    box(84, 4, 15, 14, "Monitoring", "/metrics, pred log\nPSI / KS drift\nretrain triggers", C["aqua"])
    arrow(15.8, 11, 18.2, 11, "")
    arrow(37.8, 11, 40.2, 11)
    arrow(91, 28.4, 91, 18.8, "")
    arrow(88, 28.4, 52, 18.8, "load once", dashed=True)
    arrow(86, 28.4, 74, 18.8, "", dashed=True)
    ax.plot([50.5, 50.5, 91.5], [3.4, -0.5, -0.5], ls="--", lw=1.2, color=C["ink2"])
    ax.add_patch(FancyArrowPatch((91.5, -0.5), (91.5, 3.4), arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                 color=C["ink2"], linestyle="--"))
    ax.text(71, -2.8, "predictions + latency logs", ha="center", fontsize=6.5, color=C["ink2"])
    fig.savefig(FIG_DIR / "architecture.png"); plt.close(fig)


def fig_split_timeline():
    df = pd.concat([pd.read_parquet(SPLIT_DIR / f"{s}.parquet").assign(split=s) for s in ("train", "val", "test")])
    df["hour"] = (df.Time // 3600).astype(int)
    g = df.groupby("hour").agg(n=("Class", "size"), fr=("Class", "sum"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    axes[0].bar(g.index, g.n / 1000, color=C["blue"], width=0.85)
    axes[0].set_ylabel("tx / hour (k)"); axes[0].set_ylim(0, 12)
    axes[0].set_title("Transaction stream (48 h) and temporal partitions")
    axes[1].bar(g.index, g.fr, color=C["orange"], width=0.85)
    axes[1].set_ylabel("frauds / hour"); axes[1].set_xlabel("hours since start of capture")
    s = load("data_summary.json")["time_cutoffs_seconds"]
    for ax in axes:
        for c in s:
            ax.axvline(c / 3600 - 0.5, color=C["ink"], lw=1, ls="--")
    for x, lab in [(18, "TRAIN 70%"), (39.5, "VAL"), (45, "TEST")]:
        axes[0].text(x, 10.6, lab, ha="center", fontsize=8, weight="bold", color=C["ink"])
    fig.savefig(FIG_DIR / "split_timeline.png"); plt.close(fig)


def fig_pr_curves():
    z = np.load(REPORTS_DIR / "scores.npz")
    tr = load("training_results.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    series = [("lgbm_test", "LightGBM (weighted)", C["blue"], tr["lightgbm"]["test"]["pr_auc"]),
              ("lr_test", "LogReg balanced", C["orange"], tr["logreg_balanced"]["test"]["pr_auc"]),
              ("lr_unw_test", "LogReg unweighted", C["aqua"], tr["logreg_unweighted"]["test"]["pr_auc"]),
              ("rule_test", "Rule V14 < t", C["violet"], tr["rule_V14"]["test"]["pr_auc"])]
    for k, name, col, ap in series:
        p, r, _ = precision_recall_curve(z["y_test"], z[k])
        ax.step(r, p, where="post", color=col, label=f"{name}  (AP {ap:.3f})", lw=1.8)
    base = z["y_test"].mean()
    ax.axhline(base, color=C["muted"], ls=":", lw=1); ax.text(0.02, base + 0.02, f"prior {base:.4f}", fontsize=7, color=C["ink2"])
    t = tr["lightgbm"]["test"]
    ax.plot(t["recall"], t["precision"], "o", ms=8, color=C["blue"], mec="white", mew=2)
    ax.annotate("cost-optimal\noperating point", (t["recall"], t["precision"]), (0.47, 0.30), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.8), color=C["ink2"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall on held-out TEST (last 6 h)")
    ax.legend(loc="lower left", fontsize=7, bbox_to_anchor=(0, 0.02))
    fig.savefig(FIG_DIR / "pr_curves.png"); plt.close(fig)


def fig_cost_threshold():
    z = np.load(REPORTS_DIR / "scores.npz")
    tr = load("training_results.json")
    th = np.logspace(-4, -0.01, 300)
    _, cv = cost_curve(z["y_val"], z["lgbm_val"], z["amt_val"], th)
    _, ct = cost_curve(z["y_test"], z["lgbm_test"], z["amt_test"], th)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(th, cv, color=C["blue"], label="validation (used to choose t)")
    ax.plot(th, ct, color=C["orange"], label="test (reported once)")
    t = tr["lightgbm"]["threshold_cost_opt"]
    ax.axvline(t, color=C["ink"], ls="--", lw=1)
    ax.text(t * 1.15, ax.get_ylim()[1] * 0.9, f"t* = {t:.3f}", fontsize=8, color=C["ink"])
    ax.axvline(0.5, color=C["muted"], ls=":", lw=1); ax.text(0.52, ax.get_ylim()[1] * 0.75, "0.5", fontsize=7, color=C["ink2"])
    ax.set_xscale("log"); ax.set_xlabel("decision threshold on fraud probability (log)")
    ax.set_ylabel("business cost (EUR)"); ax.set_title("Expected cost vs threshold (LightGBM)")
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(FIG_DIR / "cost_threshold.png"); plt.close(fig)


def fig_ablation():
    tr = load("training_results.json")
    g = pd.DataFrame(tr["grid_search"])
    best = g.groupby("scale_pos_weight").val_pr_auc.max()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = [f"{v:g}" for v in best.index]
    bars = ax.bar(labels, best.values, color=C["blue"], width=0.55)
    for b_, v in zip(bars, best.values):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=C["ink"])
    ax.set_ylim(0, 1); ax.set_ylabel("validation PR-AUC")
    ax.set_xlabel("scale_pos_weight  (1 = none, 23 = sqrt ratio, 54 = ratio/10, 542 = full ratio)")
    ax.set_title("Imbalance handling ablation (best config per weight)")
    fig.savefig(FIG_DIR / "imbalance_ablation.png"); plt.close(fig)


def fig_latency():
    z = np.load(REPORTS_DIR / "latencies_raw.npz")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(0, 20, 81)
    ax.hist(z["in_process_ref"], bins=bins, color=C["orange"], alpha=0.85, label="reference sklearn pipeline")
    ax.hist(z["http_sequential"], bins=bins, color=C["blue"], alpha=0.85, label="HTTP end-to-end, fast path")
    ax.hist(z["in_process"], bins=bins, color=C["aqua"], alpha=0.85, label="in-process, fast path")
    ax.set_xlabel("latency per request (ms)"); ax.set_ylabel("requests")
    ax.set_title("Single-transaction latency distribution")
    ax.legend(fontsize=7)
    fig.savefig(FIG_DIR / "latency_hist.png"); plt.close(fig)


def fig_drift():
    d = load("drift_results.json")
    h = pd.DataFrame(d["hourly_live"]); r = pd.DataFrame(d["rolling_pr_auc_3h"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    ax.plot(h.hour, h.score_psi, "-o", ms=4, color=C["blue"], label="score PSI")
    ax.plot(h.hour, h.v14_psi, "-o", ms=4, color=C["orange"], label="V14 PSI (top feature)")
    ax.plot(h.hour, h.amount_psi, "-o", ms=4, color=C["aqua"], label="Amount PSI")
    ax.axhline(0.10, color=C["muted"], ls=":", lw=1); ax.text(h.hour.min(), 0.105, "warn 0.10", fontsize=7, color=C["ink2"])
    ax.axhline(0.25, color=C["ink2"], ls="--", lw=1); ax.text(h.hour.min(), 0.255, "alert 0.25", fontsize=7, color=C["ink2"])
    ax.set_ylim(0, 0.3); ax.set_xlabel("hour of live period"); ax.set_ylabel("PSI vs hour-matched train ref")
    ax.set_title("Hourly drift (val + test windows)"); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.79))
    ax = axes[1]
    ax.plot(r.window_start_h, r.pr_auc, "-o", ms=4, color=C["blue"])
    for x, y, n in zip(r.window_start_h, r.pr_auc, r.frauds):
        ax.text(x, y + 0.03, f"n={n}", ha="center", fontsize=6.5, color=C["ink2"])
    ax.set_ylim(0, 1.05); ax.set_xlabel("3-h window start (hour)"); ax.set_ylabel("PR-AUC")
    ax.set_title("Label-delayed rolling PR-AUC (concept drift)")
    fig.savefig(FIG_DIR / "drift.png"); plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (fig_architecture, fig_split_timeline, fig_pr_curves, fig_cost_threshold, fig_ablation, fig_latency, fig_drift):
        f(); print("ok", f.__name__)


if __name__ == "__main__":
    main()
