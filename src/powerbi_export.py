"""
Power BI Export
===============
Aggregates the Spark prediction outputs in `fraud_predictions/` into a
small set of clean, single-file CSVs that Power BI (or Excel / Tableau)
can ingest with zero transformations.

Inputs:
    fraud_predictions/part-*.csv   (from src/export_predictions.py)
    results/fraud_by_type/part-*.csv

Outputs (all in results/powerbi/):
    transactions_scored.csv        flat fact table (downsampled if huge)
    summary_kpis.csv               headline KPIs (single row)
    fraud_by_type.csv              counts + fraud-rate by transaction type
    risk_level_summary.csv         counts + amount totals by risk band
    amount_buckets.csv             fraud rate per amount bucket
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "fraud_predictions"
TYPE_DIR = ROOT / "results" / "fraud_by_type"
OUT_DIR = ROOT / "results" / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS_FACT_TABLE = 500_000  # keep BI imports fast


def _read_spark_csv(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "part-*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def main() -> None:
    print("Loading prediction outputs ...")
    pred = _read_spark_csv(PRED_DIR)
    types = _read_spark_csv(TYPE_DIR)

    if pred.empty:
        raise SystemExit(
            "fraud_predictions/ is empty. Run `python src/export_predictions.py` first."
        )

    pred["risk_score"] = pred["fraud_probability"] * 100
    pred["risk_level"] = pd.cut(
        pred["risk_score"],
        bins=[-0.01, 40, 70, 100.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    # ---------- 1) flat fact table (downsample if needed) -------------
    if len(pred) > MAX_ROWS_FACT_TABLE:
        # keep ALL fraud + sample of non-fraud
        frauds = pred[pred["prediction"] == 1]
        safe = pred[pred["prediction"] == 0].sample(
            n=min(MAX_ROWS_FACT_TABLE - len(frauds), len(pred)), random_state=42
        )
        fact = pd.concat([frauds, safe], ignore_index=True)
        print(f"Downsampled fact table: {len(fact):,} rows "
              f"({len(frauds):,} fraud + {len(safe):,} sampled non-fraud)")
    else:
        fact = pred
    fact.to_csv(OUT_DIR / "transactions_scored.csv", index=False)

    # ---------- 2) headline KPIs --------------------------------------
    total = len(pred)
    fraud_n = int((pred["prediction"] == 1).sum())
    high_n = int((pred["risk_level"] == "High Risk").sum())
    med_n = int((pred["risk_level"] == "Medium Risk").sum())
    kpis = pd.DataFrame([{
        "total_transactions": total,
        "predicted_fraud": fraud_n,
        "fraud_rate_%": round(fraud_n / total * 100, 4),
        "high_risk_transactions": high_n,
        "medium_risk_transactions": med_n,
        "total_amount": float(pred["amount"].sum()),
        "fraud_amount": float(pred.loc[pred["prediction"] == 1, "amount"].sum()),
        "avg_fraud_amount": float(
            pred.loc[pred["prediction"] == 1, "amount"].mean() or 0
        ),
    }])
    kpis.to_csv(OUT_DIR / "summary_kpis.csv", index=False)

    # ---------- 3) fraud-by-type --------------------------------------
    if not types.empty:
        pv = (
            types.pivot_table(
                index="type", columns="isFraud", values="count", fill_value=0
            ).rename(columns={0: "not_fraud", 1: "fraud"}).reset_index()
        )
        if "fraud" not in pv.columns:
            pv["fraud"] = 0
        pv["total"] = pv["not_fraud"] + pv["fraud"]
        pv["fraud_rate_%"] = (pv["fraud"] / pv["total"] * 100).round(4)
        pv.to_csv(OUT_DIR / "fraud_by_type.csv", index=False)

    # ---------- 4) risk-level summary ---------------------------------
    risk_grp = (
        pred.groupby("risk_level", observed=True)
            .agg(transactions=("amount", "size"),
                 total_amount=("amount", "sum"),
                 avg_amount=("amount", "mean"))
            .reset_index()
    )
    risk_grp.to_csv(OUT_DIR / "risk_level_summary.csv", index=False)

    # ---------- 5) amount-bucket fraud rate ---------------------------
    bins = [0, 1_000, 10_000, 100_000, 200_000, 500_000, np.inf]
    labels = ["<1K", "1K–10K", "10K–100K", "100K–200K", "200K–500K", "500K+"]
    pred["amount_bucket"] = pd.cut(pred["amount"], bins=bins, labels=labels, right=False)
    bucket = (
        pred.groupby("amount_bucket", observed=True)
            .agg(transactions=("amount", "size"),
                 fraud=("prediction", "sum"))
            .reset_index()
    )
    bucket["fraud_rate_%"] = (bucket["fraud"] / bucket["transactions"] * 100).round(4)
    bucket.to_csv(OUT_DIR / "amount_buckets.csv", index=False)

    print("\nPower BI ready files written to:", OUT_DIR.relative_to(ROOT))
    for f in sorted(OUT_DIR.glob("*.csv")):
        print(f"  - {f.name}  ({f.stat().st_size/1024:,.1f} KB)")


if __name__ == "__main__":
    main()
