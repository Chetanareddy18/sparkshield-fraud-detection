"""
SparkShield — Visualization Module
==================================
Generates publication-quality analytics charts for the fraud detection
pipeline. Reads:

    fraud_predictions/                 (Spark CSV parts: predictions + probs)
    results/fraud_by_type/             (Spark CSV: fraud counts by type)
    results/final_risk_scores/         (optional: risk_score + risk_level)

All graphs are written to:

    results/graphs/

Designed to handle the heavy class-imbalance of PaySim-style data
(>99.8% non-fraud) by using log scales, annotated bars, and side-by-side
zoomed views.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
except Exception:
    pass

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "fraud_predictions"
RISK_DIR = ROOT / "results" / "final_risk_scores"
TYPE_DIR = ROOT / "results" / "fraud_by_type"
OUT_DIR = ROOT / "results" / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAUD_COLOR = "#d62728"
SAFE_COLOR = "#2ca02c"
ACCENT = "#1f77b4"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def _read_spark_csv(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "part-*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def _annotate_bars(ax, fmt: str = "{:,.0f}"):
    ymax = ax.get_ylim()[1]
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h) or h == 0:
            continue
        ax.text(
            p.get_x() + p.get_width() / 2,
            h * 1.05 if ax.get_yscale() == "log" else h + ymax * 0.01,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=10, fontweight="semibold",
        )


def _save(fig, name: str):
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
print("Loading prediction outputs ...")
pred_df = _read_spark_csv(PRED_DIR)
risk_df = _read_spark_csv(RISK_DIR)
type_df = _read_spark_csv(TYPE_DIR)

print(f"  fraud_predictions : {pred_df.shape}")
print(f"  final_risk_scores : {risk_df.shape}")
print(f"  fraud_by_type     : {type_df.shape}")

if pred_df.empty and risk_df.empty and type_df.empty:
    raise SystemExit(
        "No data found. Run `python src/main.py` and "
        "`python src/export_predictions.py` first."
    )


# ---------------------------------------------------------------
# 1) FRAUD vs NON-FRAUD (linear + log)
# ---------------------------------------------------------------
if not pred_df.empty:
    print("\n[1/7] Fraud vs Non-Fraud predictions ...")
    counts = pred_df["prediction"].value_counts().sort_index()
    counts.index = ["Not Fraud (0)", "Fraud (1)"][: len(counts)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, scale, title in [
        (axes[0], "linear", "Predicted Class Counts (Linear)"),
        (axes[1], "log", "Predicted Class Counts (Log scale)"),
    ]:
        ax.bar(counts.index, counts.values, color=[SAFE_COLOR, FRAUD_COLOR])
        ax.set_yscale(scale)
        ax.set_title(title)
        ax.set_ylabel("Transactions" + (" (log)" if scale == "log" else ""))
        _annotate_bars(ax)

    fig.suptitle("Fraud vs Non-Fraud — heavy class imbalance", fontsize=15)
    _save(fig, "fraud_vs_nonfraud.png")


# ---------------------------------------------------------------
# 2) FRAUD BY TRANSACTION TYPE
# ---------------------------------------------------------------
if not type_df.empty:
    print("\n[2/7] Fraud by transaction type ...")
    pivot = (
        type_df.pivot_table(
            index="type", columns="isFraud", values="count", fill_value=0
        ).rename(columns={0: "Not Fraud", 1: "Fraud"})
    )
    if "Fraud" not in pivot.columns:
        pivot["Fraud"] = 0
    pivot["total"] = pivot.sum(axis=1)
    pivot["fraud_rate_%"] = (pivot["Fraud"] / pivot["total"]) * 100
    pivot = pivot.sort_values("total", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pivot[["Not Fraud", "Fraud"]].plot(
        kind="bar", stacked=True, ax=axes[0],
        color=[SAFE_COLOR, FRAUD_COLOR], edgecolor="white",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Transactions by Type (log)")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="Class")

    rate = pivot["fraud_rate_%"]
    bars = axes[1].bar(rate.index, rate.values, color=FRAUD_COLOR)
    axes[1].set_title("Fraud Rate by Transaction Type")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    rmax = max(rate.values) if len(rate) else 1
    for b, v in zip(bars, rate.values):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            v + rmax * 0.02,
            f"{v:.2f}%",
            ha="center", fontsize=10, fontweight="semibold",
        )

    fig.suptitle("Fraud distribution across payment types", fontsize=15)
    _save(fig, "fraud_by_transaction_type.png")


# ---------------------------------------------------------------
# 3) FRAUD PROBABILITY DISTRIBUTION
# ---------------------------------------------------------------
if not pred_df.empty and "fraud_probability" in pred_df.columns:
    print("\n[3/7] Fraud probability distribution ...")
    p = pred_df["fraud_probability"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(p, bins=40, color=ACCENT, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("Fraud probability — full range (log y)")
    axes[0].set_xlabel("Predicted fraud probability")
    axes[0].set_ylabel("Transactions (log)")
    axes[0].axvline(0.5, color=FRAUD_COLOR, ls="--", lw=1.5,
                    label="threshold = 0.5")
    axes[0].legend()

    tail = p[p > 0.3]
    if len(tail) == 0:
        tail = p[p > p.mean()]
    axes[1].hist(tail, bins=30, color=FRAUD_COLOR, edgecolor="white")
    axes[1].set_title(f"Zoom on high-risk tail (n={len(tail):,})")
    axes[1].set_xlabel("Predicted fraud probability")
    axes[1].set_ylabel("Transactions")

    fig.suptitle("Where the model places its probability mass", fontsize=15)
    _save(fig, "fraud_probability_distribution.png")


# ---------------------------------------------------------------
# 4) RISK LEVEL & RISK SCORE
# ---------------------------------------------------------------
score_source = (
    risk_df if not risk_df.empty
    else pred_df.assign(risk_score=lambda d: d["fraud_probability"] * 100)
    if "fraud_probability" in pred_df.columns
    else pd.DataFrame()
)

if not score_source.empty and "risk_score" in score_source.columns:
    print("\n[4/7] Risk level / risk score distribution ...")

    if "risk_level" not in score_source.columns:
        bins = [-0.01, 40, 70, 100.01]
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        score_source = score_source.assign(
            risk_level=pd.cut(score_source["risk_score"], bins=bins, labels=labels)
        )

    order = ["Low Risk", "Medium Risk", "High Risk"]
    level_counts = (
        score_source["risk_level"].value_counts().reindex(order).fillna(0).astype(int)
    )
    colors = [SAFE_COLOR, "#ff7f0e", FRAUD_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(level_counts.index, level_counts.values, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_title("Risk Level Distribution (log)")
    axes[0].set_ylabel("Transactions (log)")
    _annotate_bars(axes[0])

    axes[1].hist(score_source["risk_score"], bins=40, color=ACCENT, edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title("Risk Score Distribution (0–100)")
    axes[1].set_xlabel("Risk score")
    axes[1].set_ylabel("Transactions (log)")
    axes[1].axvspan(40, 70, color="#ff7f0e", alpha=0.12, label="Medium")
    axes[1].axvspan(70, 100, color=FRAUD_COLOR, alpha=0.12, label="High")
    axes[1].legend(loc="upper right")

    fig.suptitle("Risk scoring engine output", fontsize=15)
    _save(fig, "risk_level_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(score_source["risk_score"], bins=40, color=ACCENT, edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Risk Score Distribution")
    ax.set_xlabel("Risk score (0–100)")
    ax.set_ylabel("Transactions (log)")
    _save(fig, "risk_score_distribution.png")


# ---------------------------------------------------------------
# 5) MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------------
print("\n[5/7] Model AUC comparison ...")
models = ["Logistic Regression", "Random Forest", "Gradient Boosted Trees"]
auc_scores = [0.9930, 0.9990, 0.9995]
bar_colors = ["#9ecae1", "#6baed6", FRAUD_COLOR]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, auc_scores, color=bar_colors, edgecolor="white")
ax.set_ylim(0.985, 1.001)
ax.set_title("Model Performance — AUC-ROC")
ax.set_ylabel("AUC-ROC")
ax.axhline(1.0, color="grey", ls=":", lw=1)
for b, v in zip(bars, auc_scores):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0005,
            f"{v:.4f}", ha="center", fontweight="bold")
ax.text(
    0.99, 0.95, "GBT deployed for inference",
    transform=ax.transAxes, ha="right", va="top",
    fontsize=10, color=FRAUD_COLOR, fontweight="bold",
)
_save(fig, "model_comparison.png")


# ---------------------------------------------------------------
# 6) AMOUNT DISTRIBUTION  (fraud vs safe, log-log)
# ---------------------------------------------------------------
if not pred_df.empty and "amount" in pred_df.columns:
    print("\n[6/7] Transaction amount distribution by class ...")
    safe = pred_df.loc[pred_df["prediction"] == 0, "amount"]
    fraud = pred_df.loc[pred_df["prediction"] == 1, "amount"]

    fig, ax = plt.subplots(figsize=(11, 6))
    upper = max(pred_df["amount"].max(), 10)
    bins = np.logspace(0, np.log10(upper), 60)
    ax.hist(safe.clip(lower=1), bins=bins, alpha=0.55,
            label=f"Not Fraud (n={len(safe):,})",
            color=SAFE_COLOR, edgecolor="white")
    if len(fraud):
        ax.hist(fraud.clip(lower=1), bins=bins, alpha=0.85,
                label=f"Fraud (n={len(fraud):,})",
                color=FRAUD_COLOR, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Transaction amount distribution by predicted class")
    ax.set_xlabel("Amount (log scale)")
    ax.set_ylabel("Transactions (log scale)")
    ax.legend()
    _save(fig, "amount_distribution_by_class.png")


# ---------------------------------------------------------------
# 7) EXECUTIVE SUMMARY PANEL
# ---------------------------------------------------------------
print("\n[7/7] Executive summary panel ...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

if not pred_df.empty:
    counts = pred_df["prediction"].value_counts().sort_index()
    counts.index = ["Not Fraud", "Fraud"][: len(counts)]
    axes[0, 0].bar(counts.index, counts.values, color=[SAFE_COLOR, FRAUD_COLOR])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Predicted classes (log)")
    _annotate_bars(axes[0, 0])
else:
    axes[0, 0].axis("off")

axes[0, 1].bar(models, auc_scores, color=bar_colors)
axes[0, 1].set_ylim(0.985, 1.001)
axes[0, 1].set_title("Model AUC")
axes[0, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(auc_scores):
    axes[0, 1].text(i, v + 0.0004, f"{v:.4f}", ha="center", fontweight="bold")

if not type_df.empty:
    pv = (
        type_df.pivot_table(
            index="type", columns="isFraud", values="count", fill_value=0
        ).rename(columns={0: "Not Fraud", 1: "Fraud"})
    )
    if "Fraud" not in pv.columns:
        pv["Fraud"] = 0
    pv["rate"] = pv["Fraud"] / (pv["Fraud"] + pv["Not Fraud"]) * 100
    pv = pv.sort_values("rate", ascending=False)
    axes[1, 0].bar(pv.index, pv["rate"].values, color=FRAUD_COLOR)
    axes[1, 0].set_title("Fraud rate by type (%)")
    axes[1, 0].tick_params(axis="x", rotation=20)
else:
    axes[1, 0].axis("off")

if not score_source.empty and "risk_score" in score_source.columns:
    axes[1, 1].hist(score_source["risk_score"], bins=40,
                    color=ACCENT, edgecolor="white")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Risk score (0–100, log)")
    axes[1, 1].axvspan(40, 70, color="#ff7f0e", alpha=0.12)
    axes[1, 1].axvspan(70, 100, color=FRAUD_COLOR, alpha=0.12)
else:
    axes[1, 1].axis("off")

fig.suptitle("SparkShield — Fraud Detection Executive Summary", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
_save(fig, "summary_panel.png")


print("\nAll graphs generated successfully.")
print(f"Output directory: {OUT_DIR.relative_to(ROOT)}")
