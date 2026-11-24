"""Build the consolidated PDF report from the generated results.

Usage:
  python -m report.build_report --name "Full Name" --student-id 2025XX00000 \
      --repo-url https://github.com/<user>/mini-production-ml-system
Output: MLME_Assignment_<Student_ID>_<Full_Name>.pdf in the repo root.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from src.config import FIG_DIR, FN_FIXED_COST, FP_COST, REPORTS_DIR, ROOT

# ------------------------------------------------------------------ fonts & styles
_fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_fd / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_fd / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(_fd / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-M", str(_fd / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

INK, INK2, BLUE, RULE, TINT = (colors.HexColor(h) for h in ("#0b0b0b", "#52514e", "#2a78d6", "#d9d8d3", "#f1f6fd"))
BODY = ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=12.6, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.8, leading=10, textColor=INK2)
CAP = ParagraphStyle("cap", parent=SMALL, fontName="DV-I", spaceBefore=2, spaceAfter=8, alignment=TA_CENTER)
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5,
                    keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="DV-B", fontSize=10.2, leading=13, textColor=BLUE, spaceBefore=7, spaceAfter=3,
                    keepWithNext=1)
CELL = ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=9.4, textColor=INK)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
MONO = ParagraphStyle("mono", fontName="DV-M", fontSize=7.4, leading=9.6, textColor=INK, backColor=TINT,
                      borderPadding=5, spaceBefore=3, spaceAfter=8)
W = A4[0] - 2 * 18 * mm


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items, style=BODY):
    return [Paragraph(f"•&nbsp;&nbsp;{i}", ParagraphStyle("b", parent=style, leftIndent=10, firstLineIndent=-8,
                                                           spaceAfter=2)) for i in items]


def table(rows, widths, header=True, highlight_row=None, align_right_from=1):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE), ("TOPPADDING", (0, 0), (-1, -1), 2.4),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4), ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    for i in range(1, len(rows) - 1):
        st.append(("LINEBELOW", (0, i), (-1, i), 0.25, RULE))
    if highlight_row is not None:
        st.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), TINT))
    t.setStyle(TableStyle(st))
    t.spaceAfter = 7
    return t


def fig(name, width_frac, caption):
    path = FIG_DIR / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = W * width_frac
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, CAP)])


def side_by_side(f1, f2, cap1, cap2):
    from PIL import Image as PILImage
    cells = []
    for f, c in ((f1, cap1), (f2, cap2)):
        w, h = PILImage.open(FIG_DIR / f).size
        width = W * 0.485
        cells.append([Image(str(FIG_DIR / f), width=width, height=width * h / w), P(c, CAP)])
    t = Table([cells], colWidths=[W * 0.5, W * 0.5])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def eur(x):
    return f"€{x:,.0f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ build
def build(name: str, sid: str, repo: str) -> Path:
    from xml.sax.saxutils import escape
    repo_txt = escape(repo)
    J = lambda f: json.loads((REPORTS_DIR / f).read_text())  # noqa: E731
    ds, tr, bm, rb, dr = (J("data_summary.json"), J("training_results.json"), J("benchmark_results.json"),
                          J("robustness_results.json"), J("drift_results.json"))
    meta = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    lg, lt = tr["lightgbm"], tr["lightgbm"]["test"]
    bs = lg["test_bootstrap"]
    splits = {s["split"]: s for s in ds["splits"]}
    audit = ds["leakage_audit"]
    fi = tr["feature_importance_gain"]; fi_tot = sum(fi.values())
    top_fi = list(fi.items())[:5]
    rule = tr["rule_V14"]["rule"]

    # decision tiers on test
    import numpy as np
    z = np.load(REPORTS_DIR / "scores.npz")
    s, y, a = z["lgbm_test"], z["y_test"], z["amt_test"]
    t_rev, t_blk = meta["decision_threshold"], meta["block_threshold"]
    tiers = []
    for lab, lo, hi in [("BLOCK (auto-decline)", t_blk, 9), ("REVIEW (analyst queue)", t_rev, t_blk), ("APPROVE", -1, t_rev)]:
        k = (s >= lo) & (s < hi)
        tiers.append([lab, f"p ≥ {lo:.4f}" if lo > 0 else f"p < {t_rev:.4f}", f"{k.sum():,}", f"{int(y[k].sum())}",
                      pct(y[k].mean()) if k.sum() else "-", eur(a[k & (y == 1)].sum())])

    # genuine day-over-day shift (same hours of day)
    m = {0: dr["day_over_day_median_12_18h"]["V3"]["day1"], 1: dr["day_over_day_median_12_18h"]["V3"]["day2"]}

    out = ROOT / f"MLME_Assignment_{sid}_{name.replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Mini Production ML System - Card Fraud Detection",
                            author=name)
    name, sid = escape(name), escape(sid)
    S = []

    # ======================= TITLE PAGE ==========================================================
    S += [P("BITS Pilani WILP · ML Model Engineering · Graded Assignment 1",
            ParagraphStyle("k", parent=SMALL, textColor=BLUE, fontName="DV-B")), Spacer(1, 4),
          P("Design &amp; Build a Mini Production ML System:<br/>Real-Time Credit-Card Fraud Detection (Option B)",
            ParagraphStyle("t", fontName="DV-B", fontSize=17, leading=21, textColor=INK, spaceAfter=8))]
    S.append(table([["Student", "Student ID", "Submitted"], [name, sid, date.today().strftime("%d %B %Y")]],
                   [0.45, 0.3, 0.25]))
    S.append(Spacer(1, 6))
    S.append(P(f'<b>Code repository (public):</b> <link href="{repo_txt}" color="#2a78d6"><u>{repo_txt}</u></link>',
               ParagraphStyle("repo", parent=BODY, fontSize=10, backColor=TINT, borderPadding=6, spaceBefore=4,
                              spaceAfter=10)))
    S.append(P("Executive summary", H1))
    S.append(P(
        f"We built a complete online fraud-scoring system on the ULB/Worldline card dataset "
        f"({ds['n_rows_raw']:,} transactions, {ds['n_frauds_raw']} frauds, 0.173% positive rate). The data is split "
        f"<b>by time</b> (train on the first 37 h, validate on the next 5 h, test once on the final 6 h) so that every "
        f"reported number comes from <i>future</i> traffic. All preprocessing is fitted on the training window only. "
        f"A LightGBM classifier with class-weighted loss beats a dummy prior, an interpretable V14 rule and two logistic "
        f"regressions on PR-AUC. Its decision threshold is chosen to minimise an explicit euro cost of missed fraud "
        f"and false alarms. The model is served by a FastAPI service with strict Pydantic contracts, a parity-checked "
        f"fast inference path, a rule-based fallback and tiered APPROVE / REVIEW / BLOCK decisions. A batch CLI and a "
        f"PSI/KS drift monitor that accounts for daily seasonality complete the system."))
    S.append(table([
        ["Dimension", "Result (held-out test window, 42,724 tx, 52 frauds)"],
        ["Ranking quality", f"PR-AUC <b>{lt['pr_auc']:.3f}</b> (95% CI {bs['pr_auc_95ci'][0]:.2f}-{bs['pr_auc_95ci'][1]:.2f}); "
                            f"best baseline {tr['logreg_balanced']['test']['pr_auc']:.3f}; prior {tr['dummy_prior']['test']['pr_auc']:.4f}"],
        ["Business impact", f"Cost {eur(lt['cost_eur'])} vs {eur(lt['no_model_cost_eur'])} with no model "
                            f"(<b>-{lt['savings_vs_no_model_pct']:.0f}%</b>); {lt['fraud_amount_caught_pct']:.0f}% of fraud value intercepted"],
        ["Operating point", f"Recall {lt['recall']:.2f}, precision {lt['precision']:.2f}; BLOCK tier: {tiers[0][3]} of {tiers[0][2]} auto-declines are fraud"],
        ["Online latency", f"HTTP p50 <b>{bm['http_sequential_ms']['p50']:.1f} ms</b>, p95 {bm['http_sequential_ms']['p95']:.1f} ms, "
                           f"p99 {bm['http_sequential_ms']['p99']:.1f} ms; {bm['http_sequential_ms']['n']:,} requests, 0 errors"],
        ["Batch throughput", f"~{bm['batch_throughput'][str(splits['test']['rows'])]['records_per_s']:,.0f} records/s (vectorised, 2 vCPU)"],
        ["Reliability", f"{len(rb)}/{len(rb)} malformed and edge-case payloads handled deterministically (200 + warning or structured 422); model-failure fallback verified"],
    ], [0.22, 0.78]))
    S.append(Spacer(1, 6))
    S.append(P("<b>Contents.</b> 1 Problem formulation &amp; data · 2 Model development · 3 Production system design · "
               "4 Evaluation &amp; production considerations · 5 Limitations · 6 Repository &amp; reproduction", SMALL))
    S.append(PageBreak())

    # ======================= 1. PROBLEM ========================================================
    S.append(P("1. Problem Formulation, Ingestion &amp; Leak-Free Splitting", H1))
    S.append(P("1.1 Business objective and serving paradigm", H2))
    S.append(P(
        "The card issuer must decide, <i>during authorisation</i>, whether a transaction should be approved, sent to "
        "an analyst, or declined. A missed fraud costs the issuer the charged-back amount plus handling. A false alarm "
        "costs analyst time and customer friction. The paradigm is therefore <b>online, synchronous scoring</b>: one "
        "transaction per request, inside an authorisation latency budget (we target p99 &lt; 50 ms for the scorer). A "
        "batch path exists for back-scoring and audits."))
    S.append(P(
        "<b>Formal statement.</b> Each transaction gives x ∈ ℝ<super>30</super> (Time, V1-V28, Amount) and label "
        "y ∈ {0,1} (1 = fraud). We learn a score f(x) ∈ [0,1] and a decision δ(x) = 1[f(x) ≥ t]. The threshold is chosen "
        "on validation data as t* = argmin<sub>t</sub> Σ<sub>i</sub> [ y<sub>i</sub>(1-δ<sub>i</sub>)(Amount<sub>i</sub> + "
        f"{FN_FIXED_COST:.0f}) + (1-y<sub>i</sub>)δ<sub>i</sub>·{FP_COST:.0f} ] (EUR). The cost parameters are explicit, "
        "configurable assumptions in <font name='DV-M'>src/config.py</font>."))
    S.append(P("1.2 Data and feature space", H2))
    S.append(table([
        ["Field", "Description", "Treatment"],
        ["Time", "Seconds since first transaction of the 48 h capture window", "Raw value dropped (not meaningful in production, look-ahead hazard). Replaced by cyclic hour-of-day (sin, cos)"],
        ["V1-V28", "PCA components of confidential card/merchant attributes (provider-anonymised)", "Median impute → clip to train [0.05%, 99.95%] → RobustScaler"],
        ["Amount", "Transaction amount (EUR), heavy-tailed (max 25,691)", "Median impute → log1p → RobustScaler; FN cost weight"],
        ["Class", f"Target: {ds['n_frauds_raw']} frauds (0.173%)", "Never used in features; only in loss weighting and evaluation"],
    ], [0.12, 0.38, 0.5]))
    S.append(P("1.3 Partitioning strategy", H2))
    S.append(P(
        "Fraud models are always deployed on <i>future</i> traffic, and fraud patterns change over time. A random or "
        "stratified split would mix transactions from the same hours (and repeated cards) into both train and test, "
        "which inflates scores. We therefore use a <b>strict time cutoff</b>. Rows are ordered by Time, cuts fall at "
        "the 70% and 85% quantiles, and the boundaries are on timestamps so no second is shared between partitions. "
        "Stratification is not needed to keep positives in each split (each window has ≥ 52 frauds). We report "
        "bootstrap confidence intervals because positives are scarce."))
    S.append(table([["Split", "Rows", "Frauds", "Fraud rate", "Window (h)", "Role"]] + [
        [s_["split"].upper(), f"{s_['rows']:,}", s_["frauds"], f"{s_['fraud_rate_pct']:.3f}%",
         f"{s_['time_start_h']:.1f} - {s_['time_end_h']:.1f}", r] for s_, r in zip(
            ds["splits"], ["fit preprocessing + model", "early stopping, tuning, threshold", "reported once"])],
        [0.1, 0.13, 0.1, 0.13, 0.16, 0.38]))
    S.append(Spacer(1, 4))
    S.append(fig("split_timeline.png", 0.92, "Figure 1. Hourly volume and fraud counts with temporal cut points. The "
                 "test window is the last six hours of day 2, which the model never saw in training."))
    S.append(P("1.4 Leakage audit", H2))
    S.append(table([
        ["Check", "Result", "Interpretation"],
        ["Temporal ordering", "max(t_train) < min(t_val) < min(t_test): <b>pass</b>", "No look-ahead"],
        ["Transaction-ID overlap", f"{audit['id_overlap']}", "Partitions disjoint"],
        ["Preprocessing fit scope", "Asserted: imputer/clipper/scaler fitted on 198,648 TRAIN rows", "Val/test/inference only transform; unit-tested"],
        ["Exact duplicates in TRAIN", f"{audit['train_exact_duplicates_removed']} removed (train only)", "Val/test left as real traffic"],
        ["Test rows identical to a train feature vector", f"{audit['test_rows_with_identical_train_features']} (0 frauds, median €2)",
         "Recurring small legitimate payments. Cannot inflate fraud metrics"],
        ["Target-proxy screen (single-feature AUC)", f"max {audit['top_single_feature_auc_train'][0][0]} = {audit['top_single_feature_auc_train'][0][1]:.3f}; none > 0.99",
         "No leaked label proxy"],
        ["Threshold / hyper-parameters", "Selected on VAL only", "TEST evaluated exactly once"],
    ], [0.27, 0.38, 0.35]))
    S.append(P(
        "<i>Residual caveat:</i> the provider computed the PCA projection on the full dataset before release. This "
        "upstream transformation cannot be undone. It is unsupervised (no label information), so its effect on our "
        "estimates is small, but in a real deployment the PCA would be refitted on training data.", SMALL))

    # ======================= 2. MODEL ========================================================
    S.append(P("2. Preprocessing, Baselines &amp; Model Engineering", H1))
    S.append(P("2.1 Metric choice", H2))
    S.append(P(
        f"With 0.12% fraud in the test window, a model that approves everything reaches "
        f"<b>{pct(tr['dummy_prior']['test']['accuracy'], 2)} accuracy</b> while catching nothing, so accuracy is "
        "excluded. Our <b>primary metric is PR-AUC</b> (average precision), which is threshold-free and focuses on the "
        "positive class. ROC-AUC is reported but is optimistic under heavy skew. At the operating point we report "
        "precision, recall, F1/MCC, recall at 80% precision, and the <b>business cost in EUR</b>, which ties the "
        "model to money."))
    S.append(P("2.2 Baselines and model comparison", H2))
    names = [("dummy_prior", "Dummy (class prior)"), ("rule_V14", f"Rule: V14 < {rule['threshold']:.2f} (fit on train)"),
             ("logreg_unweighted", "Logistic regression, unweighted"), ("logreg_balanced", "Logistic regression, class-weighted"),
             ("lightgbm", "<b>LightGBM, weighted loss (final)</b>")]
    rows = [["Model", "Val PR-AUC", "Test PR-AUC", "ROC AUC", "Prec.", "Recall", "F1", "R@ P80", "Cost (EUR)"]]
    for k, lab in names:
        t = tr[k]["test"]
        rows.append([lab, f"{tr[k]['val_pr_auc']:.3f}", f"{t['pr_auc']:.3f}", f"{t['roc_auc']:.3f}", f"{t['precision']:.2f}",
                     f"{t['recall']:.2f}", f"{t['f1']:.2f}", f"{t['recall_at_p80']:.2f}", f"{t['cost_eur']:,.0f}"])
    S.append(table(rows, [0.30, 0.085, 0.085, 0.08, 0.075, 0.075, 0.06, 0.07, 0.10], highlight_row=5))
    S.append(P("Thresholds for every model except the dummy and the rule are cost-optimal on validation. R@P80 = best "
               "recall with precision ≥ 0.80.", SMALL))
    S.append(side_by_side("pr_curves.png", "imbalance_ablation.png",
                          "Figure 2. Test precision-recall curves.",
                          "Figure 3. Validation PR-AUC by scale_pos_weight."))
    S.append(P(
        f"<b>Reading the results.</b> The single-feature rule already gives a non-trivial baseline (PR-AUC "
        f"{tr['rule_V14']['test']['pr_auc']:.2f}), and V14 carries {100 * top_fi[0][1] / fi_tot:.0f}% of the LightGBM gain. "
        f"Class weighting improves logistic regression from {tr['logreg_unweighted']['test']['pr_auc']:.3f} to "
        f"{tr['logreg_balanced']['test']['pr_auc']:.3f} PR-AUC. LightGBM ranks best (PR-AUC {lt['pr_auc']:.3f}) and, more "
        f"importantly, has the lowest cost: {eur(lt['cost_eur'])} vs {eur(tr['logreg_balanced']['test']['cost_eur'])} "
        f"for the best baseline. The bootstrap CI for test PR-AUC ({bs['pr_auc_95ci'][0]:.2f}-{bs['pr_auc_95ci'][1]:.2f}) "
        f"is wide because the test window holds only 52 frauds. The ranking against the rule and dummy is robust; the "
        f"margin over logistic regression is suggestive but not conclusive."))
    S.append(P("2.3 Class-imbalance handling", H2))
    ab = {a_["scale_pos_weight"]: a_ for a_ in tr["imbalance_ablation"]}
    ks = sorted(ab)
    S.append(P(
        "Following the brief, we use <b>loss re-weighting and threshold tuning instead of oversampling</b>. SMOTE-style "
        "resampling creates synthetic points between frauds that may belong to different modi operandi, and changes "
        "the score distribution that monitoring relies on. We grid-searched LightGBM's positive-class weight together "
        "with tree size, using early stopping on validation average precision. A <i>moderate</i> weight (√ratio ≈ "
        f"{ks[1]:g}) was best (val PR-AUC {ab[ks[1]]['val_pr_auc']:.3f} vs {ab[ks[0]]['val_pr_auc']:.3f} unweighted). The "
        f"full inverse-frequency weight ({ks[-1]:g}) collapsed to {ab[ks[-1]]['val_pr_auc']:.3f}. It over-weights ~360 "
        f"positives, overfits them within {ab[ks[-1]]['best_iter']} trees and destroys precision. Final model: "
        f"{lg['params']['num_leaves']} leaves, learning rate {lg['params']['learning_rate']}, {lg['best_iteration']} trees."))
    S.append(P("2.4 Threshold selection tied to business cost", H2))
    S.append(side_by_side("cost_threshold.png", "latency_hist.png",
                          "Figure 4. Cost vs threshold. t* is chosen on validation.",
                          "Figure 5. Single-request latency (see §4.1)."))
    f1t, t05 = lg["test_at_f1_threshold"], lg["test_at_0.5"]
    S.append(P(
        f"Because weighting shifts probabilities upward for frauds and the cost of a miss is roughly 20-40 times the "
        f"cost of a review, the cost-optimal threshold is low (t* = {t_rev:.4f}). The usual defaults are worse in euros. "
        f"At t = 0.5 the test cost is {eur(t05['cost_eur'])}; at the F1-optimal t = {t_blk:.2f} it is "
        f"{eur(f1t['cost_eur'])} (F1 {f1t['f1']:.2f}, precision {f1t['precision']:.2f}); at t* it is "
        f"<b>{eur(lt['cost_eur'])}</b> (F1 {lt['f1']:.2f}). The F1-maximising model would look better on a leaderboard "
        f"and lose more money. Neither threshold is discarded. Together they define the <b>three-tier policy</b> served "
        f"in production:"))
    S.append(table([["Tier", "Rule", "Test tx", "Frauds", "Precision", "Fraud value"]] + tiers,
                   [0.26, 0.17, 0.12, 0.1, 0.12, 0.13]))
    S.append(P(
        "The validation cost curve shows a cliff just above t*, so the threshold estimate is fragile. In production it "
        "would be re-estimated on a larger rolling window of labelled data, and the REVIEW-queue size (≈ "
        f"{lt['alerts_per_10k']:.0f} alerts per 10k transactions) checked against analyst capacity.", SMALL))

    # ======================= 3. SYSTEM ========================================================
    S.append(CondPageBreak(120 * mm))
    S.append(P("3. Production System Architecture &amp; Inference", H1))
    S.append(fig("architecture.png", 1.0, "Figure 6. System architecture. The offline pipeline produces versioned "
                 "artifacts. The online API and the batch CLI share one ModelService, which loads them once at startup."))
    S.append(P("3.1 Decoupled artifacts", H2))
    S.append(P(
        "Training exports <font name='DV-M'>fraud_pipeline.joblib</font>, a single sklearn Pipeline (fitted "
        "preprocessing + LightGBM), so training and serving cannot diverge. "
        "<font name='DV-M'>model_metadata.json</font> holds the version (date + SHA-256 prefix), both thresholds and "
        "the threshold policy, the input schema, training medians, input bounds, the fallback rule, the critical "
        "features and library versions. <font name='DV-M'>reference_profile.json</font> holds the drift baselines. "
        "The service never imports training code; it only reads these files."))
    S.append(P("3.2 Online API contract (FastAPI + Pydantic v2)", H2))
    S.append(table([
        ["Endpoint", "Purpose", "Responses"],
        ["GET /health", "Liveness + model status", "200 ok | 200 degraded (fallback active) | 503 unavailable"],
        ["GET /metadata", "Model version, thresholds, schema, test metrics", "200"],
        ["GET /metrics", "Request/record counters, fallback count, rolling p50/p95/p99, status-code histogram", "200"],
        ["POST /predict", "Score one transaction → probability, is_fraud, decision, scorer, model_version, warnings", "200 | 422 invalid_payload | 503 not ready"],
        ["POST /predict/batch", "Score 1-1,000 transactions, order and IDs preserved", "200 | 422"],
    ], [0.17, 0.53, 0.30]))
    S.append(P(
        "The schema forbids unknown fields and rejects NaN/±Inf, negative or absurd amounts, malformed IDs (regex "
        "allow-list) and oversized batches. V1-V28 are nullable: up to 10 missing values are imputed and listed in "
        "<font name='DV-M'>warnings</font>; more than that is rejected, because the score would be mostly "
        "imputation. Values outside the training range are clipped by the pipeline and flagged. A middleware adds "
        "an <font name='DV-M'>x-request-id</font> and server-side timing to every response, and a custom handler "
        "returns structured 422 errors."))
    S.append(P("3.3 Serving optimisation: compiled fast path", H2))
    S.append(P(
        f"Profiling showed that the sklearn ColumnTransformer took ~6.6 ms of a ~7.6 ms single-row prediction; the "
        f"booster itself takes ~0.2 ms. <font name='DV-M'>serving/fast_path.py</font> reads the <i>already-fitted</i> "
        f"statistics (medians, clip bounds, scaler centre/scale) into numpy vectors and applies the same arithmetic. "
        f"At startup the service checks parity against the reference pipeline on 512 synthetic rows that include NaNs "
        f"and ±1e6 outliers (tolerance 1e-10). A unit test repeats the check on all 42,724 test rows. If parity fails, "
        f"the service silently uses the reference pipeline. In-process p50 fell from "
        f"{bm['in_process_single_ms_reference_pipeline']['p50']:.1f} ms to {bm['in_process_single_ms']['p50']:.1f} ms. "
        f"LightGBM uses 1 thread for single requests (avoiding OpenMP start-up) and all cores for batches of 1,000 or more."))
    S.append(P("3.4 Batch path", H2))
    S.append(P(
        "<font name='DV-M'>python -m serving.predict_batch --input new.csv --output scored.csv</font> validates columns, "
        "coerces types, tags invalid rows as <font name='DV-M'>rejected_invalid_input</font> instead of failing the "
        "run, scores in vectorised chunks, writes record IDs with probabilities, and logs counts, flags and records/s. "
        "Exit code 2 signals a schema failure to the scheduler."))

    # ======================= 4. EVALUATION ========================================================
    S.append(P("4. Operational Analysis, Profiling &amp; Monitoring", H1))
    S.append(P("4.1 Empirical latency and throughput", H2))
    S.append(P(
        f"<font name='DV-M'>serving/benchmark.py</font> starts uvicorn (1 worker) in a subprocess and sends real test "
        f"transactions over HTTP keep-alive after a warm-up. Client and server share the same 2-vCPU machine "
        f"({bm['env']['machine']}, Python {bm['env']['python']}), so these figures are conservative."))
    hc = bm["http_concurrent"]
    lat_rows = [["Scenario", "Requests", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Throughput", "Errors"],
                ["In-process, reference sklearn pipeline", "1,000", *(f"{bm['in_process_single_ms_reference_pipeline'][k]:.2f}" for k in ("p50", "p95", "p99")), "-", "0"],
                ["In-process, compiled fast path", "1,000", *(f"{bm['in_process_single_ms'][k]:.2f}" for k in ("p50", "p95", "p99")), "-", "0"],
                ["<b>HTTP /predict, sequential</b>", f"{bm['http_sequential_ms']['n']:,}", *(f"<b>{bm['http_sequential_ms'][k]:.2f}</b>" for k in ("p50", "p95", "p99")), f"{bm['http_sequential_ms']['rps']:.0f} req/s", bm['http_sequential_ms']['errors']]]
    for c_ in ("1", "8", "32"):
        lat_rows.append([f"HTTP /predict, concurrency {c_}", f"{hc[c_]['n']:,}", *(f"{hc[c_][k]:.1f}" for k in ("p50", "p95", "p99")), f"{hc[c_]['throughput_rps']:.0f} req/s", hc[c_]["errors"]])
    hb, hf = bm["http_batch100_ms"], bm["http_fallback_mode_ms"]
    lat_rows.append(["HTTP /predict/batch (100 tx)", f"{hb['n']}", *(f"{hb[k]:.1f}" for k in ("p50", "p95", "p99")), f"{hb['records_per_s']:,.0f} rec/s", hb["errors"]])
    lat_rows.append(["HTTP /predict, fallback mode", f"{hf['n']:,}", *(f"{hf[k]:.2f}" for k in ("p50", "p95", "p99")), "-", hf["errors"]])
    bt = bm["batch_throughput"]
    lat_rows.append(["Offline batch, 42,724 rows", "3 runs", "-", "-", "-", f"{bt[str(splits['test']['rows'])]['records_per_s']:,.0f} rec/s", "0"])
    S.append(table(lat_rows, [0.33, 0.1, 0.09, 0.09, 0.09, 0.17, 0.08], highlight_row=3))
    S.append(P(
        f"Single-request latency is well within the 50 ms budget (p99 {bm['http_sequential_ms']['p99']:.1f} ms). Of the "
        f"~{bm['http_sequential_ms']['p50']:.0f} ms over HTTP, about half is FastAPI/Pydantic/JSON overhead. Under "
        f"concurrency the single worker saturates near {hc['8']['throughput_rps']:.0f}-{bm['http_sequential_ms']['rps']:.0f} "
        f"req/s, so latency grows by queueing (Little's law: 32 in flight / ~137 req/s ≈ 230 ms, matching the measured "
        f"p50). Scoring is CPU-bound and stateless, so capacity scales almost linearly with uvicorn workers and replicas. "
        f"Capacity planning should use ≈ {bm['http_sequential_ms']['rps'] * 0.6:.0f} req/s per vCPU at 60% utilisation "
        f"to keep p99 low. Micro-batching callers through /predict/batch raises throughput to "
        f"{hb['records_per_s']:,.0f} records/s."))
    S.append(P("4.2 Robustness and edge cases", H2))
    rrows = [["Input case", "Status", "Behaviour"]]
    pick = ["valid transaction", "3 PCA features null (imputed)", "12 PCA features null (> limit)",
            "PCA feature key absent (treated as null)", "Amount missing", "Negative Amount", "Extreme Amount 900,000",
            "Extreme V14 = -1e6 (clipped)", "Amount as string '12.5' (coerced)", "Amount as non-numeric string",
            "Unknown extra field", "Injection-like transaction_id", "NaN literal in V1", "Malformed JSON body",
            "Batch of 1001 (> limit)"]
    for r_ in rb:
        if r_["case"] in pick:
            beh = ", ".join(r_.get("warnings") or []) or (r_.get("msg") or r_.get("error") or "scored normally")
            rrows.append([r_["case"], r_["status"], beh[:95]])
    S.append(table(rrows, [0.36, 0.08, 0.56]))
    S.append(P("4.3 Failure modes and graceful degradation", H2))
    S.append(table([
        ["Failure", "Detection", "System response"],
        ["Model artifact corrupt / load error", "Exception at startup; /health = degraded", "Serve V14 rule (stored in metadata, pure arithmetic); scorer=fallback_rule; nothing auto-blocked; page on-call"],
        ["Runtime scoring exception or non-finite output", "try/except in ModelService; model_errors counter", "Same request re-scored by the fallback (p50 4.8 ms); error logged with request-id"],
        ["Fast-path parity mismatch", "Parity check at load", "Use reference sklearn pipeline (+~5 ms), warn"],
        ["Malformed / hostile payload", "Pydantic schema", "422 with field-level reasons; never reaches the model"],
        ["Upstream feature outage (many nulls)", "warnings + missing-rate drift check", "≤10 nulls imputed and flagged; >10 rejected; DQ alert if missing rate > 5%"],
        ["Overload / latency SLO breach", "/metrics p99, 5xx rate", "Scale replicas. Callers time out at 50 ms and apply a static policy (approve below an amount limit, review above)"],
        ["Silent performance decay", "Score PSI + delayed-label PR-AUC", "Retrain trigger (§4.4); shadow-deploy the new model before promotion"],
    ], [0.24, 0.27, 0.49]))
    S.append(P(
        f"The fallback rule is weaker (test PR-AUC {tr['rule_V14']['test']['pr_auc']:.2f}, recall "
        f"{tr['rule_V14']['test']['recall']:.2f}) but still cuts cost by {tr['rule_V14']['test']['savings_vs_no_model_pct']:.0f}% "
        f"against approving everything, which makes it far better than failing open. It was verified end-to-end with "
        f"<font name='DV-M'>SIMULATE_MODEL_FAILURE=1</font>: {hf['n']:,} requests, 0 errors, /health = degraded."))

    S.append(P("4.4 Monitoring blueprint: system, data and concept drift", H2))
    S.append(table([
        ["Layer", "Signal (window)", "Threshold / action"],
        ["System", "p50/p95/p99 latency, 4xx/5xx rate, fallback rate, CPU, queue depth (1 min)", "p99 > 50 ms for 5 min or 5xx > 0.5% → page; any fallback traffic → page"],
        ["Data quality", "Missing rate per feature, out-of-range rate, schema-reject rate (15 min)", "Missing > 5% on any feature or 422 rate > 1% → page (likely upstream bug)"],
        ["Feature drift", "PSI (deciles of train) + KS on 28 PCA features and Amount vs an <b>hour-of-day-matched</b> train reference (1 h / 1 day)", "PSI ≥ 0.10 warn, ≥ 0.25 alert. Alert on a top-5 importance feature → retrain candidate; on others → investigate"],
        ["Prediction drift", "PSI of score distribution vs validation; REVIEW/BLOCK rates (1 h)", "Score PSI ≥ 0.25 or alert rate ×2 → retrain candidate + review queue capacity"],
        ["Concept drift", "PR-AUC, recall at t*, cost per 10k tx on chargeback-labelled data (labels arrive with delay, often weeks)", "Rolling PR-AUC drop > 15% relative for 2 windows → retrain on most recent data, shadow test, promote"],
    ], [0.14, 0.46, 0.40]))
    S.append(fig("drift.png", 1.0, "Figure 7. Left: hourly PSI against an hour-matched training reference. Right: "
                 "label-delayed rolling PR-AUC (3 h windows; n = number of frauds)."))
    sc = {k: dr[k] for k in ("A_validation_natural", "B_test_natural", "C_test_covariate_shift", "D_test_pipeline_bug")}
    drows = [["Scenario", "Score PSI", "Alert rate", "Feature alerts (critical)", "Outcome"]]
    lab = {"A_validation_natural": "Validation window (natural)", "B_test_natural": "Test window (natural)",
           "C_test_covariate_shift": "Synthetic attack: Amount ×3, V14 -1.5σ", "D_test_pipeline_bug": "Synthetic bug: 30% V4 null, V12 sign flip"}
    for k, v in sc.items():
        su = v["summary"]
        outc = "page on-call (DQ)" if su["page_oncall"] else ("retrain candidate" if su["retrain_recommended"] else "investigate only")
        drows.append([lab[k], f"{v['score_psi']['psi']:.3f}", f"{v['alert_rate_pct']:.2f}%",
                      f"{su['n_features_alert']} ({', '.join(su['critical_features_alert']) or 'none'})", outc])
    S.append(table(drows, [0.33, 0.1, 0.1, 0.24, 0.23]))
    S.append(P(
        f"<b>Findings.</b> (i) A naive PSI against the global training distribution raised alerts on every window, "
        f"because a 5-6 h window covers only part of the daily cycle. Matching the reference to the same hours of day "
        f"and excluding the cyclic features removes this false alarm. (ii) Even then, several PCA features drift "
        f"<i>genuinely</i> from day to day: for example, the median of V3 between 12:00 and 18:00 is {m.get(0, float('nan')):.2f} "
        f"on day 1 and {m.get(1, float('nan')):.2f} on day 2. Yet the score PSI stays below 0.03 and PR-AUC holds near "
        f"0.8, so feature drift alone is not a reason to retrain. We therefore weight alerts by feature importance. "
        f"(iii) In the natural test window, V4 (a top-5 feature) crosses the alert line, and delayed labels confirm a "
        f"decline (rolling PR-AUC {dr['rolling_pr_auc_3h'][0]['pr_auc']:.2f} → {dr['rolling_pr_auc_3h'][-1]['pr_auc']:.2f}, "
        f"on few frauds). The policy recommends retraining here, which is the right call. (iv) The synthetic attack "
        f"moves the score PSI into the warning band and triples the alert rate. The pipeline bug is caught by the "
        f"missing-rate check and by PSI on V12, even though its score PSI looks harmless. This is why data-quality "
        f"checks have to run alongside score monitoring."))

    # ======================= 5. LIMITATIONS ========================================================
    S.append(P("5. Limitations and Next Steps", H1))
    S += bullets([
        "Two days of data and only 52 test frauds: the confidence intervals are wide. A production evaluation needs "
        "months of data and backtesting over several time cutoffs.",
        "Anonymised PCA features prevent domain features (card velocity, merchant risk, geography) that usually "
        "dominate real fraud systems, and the PCA was fitted upstream on all data.",
        "The cost model is an assumption (FN = amount + €15, FP = €5). It should be calibrated with the "
        "fraud-operations team, with analyst capacity as a constraint on the REVIEW tier.",
        "Scores are not calibrated probabilities (weighted loss). Add isotonic calibration on a rolling window if "
        "consumers need true probabilities.",
        "Next steps: champion/challenger shadow deployment, Prometheus exporter for /metrics, scheduled drift job on "
        "the JSONL prediction log, ONNX export for a dependency-light runtime.",
    ])

    # ======================= 6. REPO ========================================================
    S.append(P("6. Code Repository &amp; Reproduction", H1))
    S.append(P(f'<b>Public repository:</b> <link href="{repo_txt}" color="#2a78d6"><u>{repo_txt}</u></link>',
               ParagraphStyle("repo2", parent=BODY, backColor=TINT, borderPadding=5, spaceAfter=8)))
    S.append(P(
        "All dependencies are pinned in <font name='DV-M'>requirements.txt</font> (Python 3.11+, tested 3.13). The "
        "dataset is downloaded and checked by SHA-256. Runs are deterministic: re-training reproduces identical "
        "metrics. The test suite has 25 tests covering schemas, leakage-safe preprocessing, artifact integrity, "
        "fast-path parity, the API contract, tiered decisions and the fallback."))
    S.append(P(
        "pip install -r requirements.txt<br/>"
        "python -m src.data_pipeline&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# fetch + verify + temporal split + leakage audit<br/>"
        "python -m src.train&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# baselines, tuning, threshold, artifacts<br/>"
        "python -m pytest -q&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 25 tests<br/>"
        "uvicorn serving.app:app --port 8000&nbsp;&nbsp;# online API<br/>"
        "python -m serving.predict_batch --input data/splits/unseen_sample.csv --output outputs/scored.csv<br/>"
        "python -m serving.benchmark&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# latency percentiles + throughput<br/>"
        "python -m monitoring.simulate_drift&nbsp;# PSI/KS scenarios", MONO))
    S.append(P(
        "<b>References.</b> Dal Pozzolo, Caelen, Johnson &amp; Bontempi, <i>Calibrating Probability with Undersampling "
        "for Unbalanced Classification</i>, IEEE SSCI 2015 (dataset, ULB MLG / Worldline). Ke et al., <i>LightGBM</i>, "
        "NeurIPS 2017. Saito &amp; Rehmsmeier, <i>The Precision-Recall Plot Is More Informative than the ROC Plot on "
        "Imbalanced Datasets</i>, PLoS ONE 2015. Siddiqi, <i>Credit Risk Scorecards</i> (PSI), Wiley 2006. Libraries: "
        "scikit-learn, LightGBM, FastAPI, Pydantic, SciPy, Matplotlib, ReportLab.", SMALL))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DV", 7); canvas.setFillColor(INK2)
        canvas.drawString(18 * mm, 9 * mm, f"MLME Assignment 1 · {name} ({sid}) · Card fraud detection")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Your Full Name")
    ap.add_argument("--student-id", default="STUDENTID")
    ap.add_argument("--repo-url", default="https://github.com/<your-username>/mini-production-ml-system")
    a = ap.parse_args()
    print(build(a.name or "Your Full Name", a.student_id or "STUDENTID", a.repo_url))


if __name__ == "__main__":
    main()
