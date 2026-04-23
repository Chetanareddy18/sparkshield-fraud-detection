"""
SparkShield Streamlit Dashboard
================================
Two-page UI:
  1. Live transaction screening — scores a single transaction with the
     trained Spark GBT model when available, otherwise falls back to a
     deterministic rules-based scorer so the demo still works without
     PySpark installed.
  2. Analytics — visual KPIs computed from the batch prediction outputs.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "fraud_predictions"
GRAPHS_DIR = ROOT / "results" / "graphs"
MODEL_DIR = ROOT / "models" / "gbt_model"

st.set_page_config(page_title="SparkShield — Fraud Detection",
                   page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_spark_model():
    """Load the trained PySpark GBT model. Returns None if unavailable."""
    try:
        from pyspark.sql import SparkSession  # noqa: F401
        from pyspark.ml.classification import GBTClassificationModel
    except Exception:
        return None
    if not MODEL_DIR.exists():
        return None
    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .appName("SparkShield-Dashboard")
            .master("local[*]")
            .getOrCreate()
        )
        model = GBTClassificationModel.load(str(MODEL_DIR))
        return {"spark": spark, "model": model}
    except Exception as e:
        st.warning(f"Could not load Spark model: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_predictions(sample: int = 200_000) -> pd.DataFrame:
    files = sorted(glob.glob(str(PRED_DIR / "part-*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    if len(df) > sample:
        frauds = df[df["prediction"] == 1]
        safe = df[df["prediction"] == 0].sample(
            n=min(sample - len(frauds), len(df)), random_state=42
        )
        df = pd.concat([frauds, safe], ignore_index=True)
    df["risk_score"] = df["fraud_probability"] * 100
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 40, 70, 100.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return df


def rules_score(amount, old_o, new_o, old_d, new_d, hi_val, drained,
                bal_err, dest_err) -> tuple[int, float, str]:
    """Lightweight fallback scorer used when PySpark is unavailable."""
    score = 0.0
    score += 0.35 if hi_val else 0.0
    score += 0.30 if drained else 0.0
    score += 0.15 if amount > 200_000 else 0.0
    score += 0.10 if abs(bal_err) > 1_000 else 0.0
    score += 0.10 if abs(dest_err) > 1_000 else 0.0
    p = min(score, 0.99)
    pred = int(p >= 0.5)
    level = "High Risk" if p > 0.7 else "Medium Risk" if p > 0.4 else "Low Risk"
    return pred, p, level


def spark_score(ctx, payload: dict) -> tuple[int, float, str]:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import Row

    spark = ctx["spark"]
    model = ctx["model"]
    df = spark.createDataFrame([Row(**payload)])
    cols = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "balanceError", "destBalanceError",
        "highValueTransaction", "accountDrained",
    ]
    df = VectorAssembler(inputCols=cols, outputCol="features").transform(df)
    out = model.transform(df).select("prediction", "probability").collect()[0]
    p = float(out["probability"][1])
    pred = int(out["prediction"])
    level = "High Risk" if p > 0.7 else "Medium Risk" if p > 0.4 else "Low Risk"
    return pred, p, level


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🛡️ SparkShield")
page = st.sidebar.radio("Navigate", ["Live Screening", "Analytics", "About"])
ctx = get_spark_model()
if ctx is not None:
    st.sidebar.success("PySpark GBT model loaded")
else:
    st.sidebar.info("Spark model not available — using rules fallback")


# ------------------------------------------------------------------
# Page 1: Live Screening
# ------------------------------------------------------------------
if page == "Live Screening":
    st.title("💳 Live Transaction Screening")
    st.caption("Scores a single transaction with the trained Spark GBT model "
               "(or rules fallback).")

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Transaction Amount", value=1000.0, step=100.0)
        old_o = st.number_input("Sender Old Balance", value=5000.0, step=100.0)
        new_o = st.number_input("Sender New Balance", value=4000.0, step=100.0)
        old_d = st.number_input("Receiver Old Balance", value=2000.0, step=100.0)
    with c2:
        new_d = st.number_input("Receiver New Balance", value=3000.0, step=100.0)
        bal_err = (old_o - new_o) - amount
        dest_err = (new_d - old_d) - amount
        st.metric("Balance Error (auto)", f"{bal_err:,.2f}")
        st.metric("Destination Balance Error (auto)", f"{dest_err:,.2f}")
        hi_val = st.selectbox("High Value Transaction (>200k)",
                              [int(amount > 200_000)], index=0)
        drained = st.selectbox("Account Drained (newOrig == 0)",
                               [int(new_o == 0 and old_o > 0)], index=0)

    if st.button("🔍 Predict Fraud", type="primary"):
        payload = dict(
            amount=float(amount),
            oldbalanceOrg=float(old_o),
            newbalanceOrig=float(new_o),
            oldbalanceDest=float(old_d),
            newbalanceDest=float(new_d),
            balanceError=float(bal_err),
            destBalanceError=float(dest_err),
            highValueTransaction=int(hi_val),
            accountDrained=int(drained),
        )
        if ctx is not None:
            pred, p, level = spark_score(ctx, payload)
            engine = "Spark GBT"
        else:
            pred, p, level = rules_score(
                amount, old_o, new_o, old_d, new_d, hi_val, drained,
                bal_err, dest_err,
            )
            engine = "Rules fallback"

        st.subheader(f"Result  ·  engine: {engine}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", "🚨 Fraud" if pred == 1 else "✅ Not Fraud")
        m2.metric("Fraud Probability", f"{p*100:.2f}%")
        m3.metric("Risk Level", level)
        st.progress(min(max(p, 0.0), 1.0))


# ------------------------------------------------------------------
# Page 2: Analytics
# ------------------------------------------------------------------
elif page == "Analytics":
    st.title("📊 Batch Analytics")
    df = load_predictions()
    if df.empty:
        st.warning("No predictions found. Run "
                   "`python src/export_predictions.py` first.")
    else:
        total = len(df)
        fraud_n = int((df["prediction"] == 1).sum())
        high_n = int((df["risk_level"] == "High Risk").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions (sample)", f"{total:,}")
        m2.metric("Predicted Fraud", f"{fraud_n:,}")
        m3.metric("Fraud Rate", f"{fraud_n/total*100:.3f}%")
        m4.metric("High-Risk", f"{high_n:,}")

        st.divider()
        cols = st.columns(2)
        for ax, name in zip(cols, ["summary_panel.png",
                                   "fraud_by_transaction_type.png"]):
            p = GRAPHS_DIR / name
            if p.exists():
                ax.image(str(p), use_container_width=True)

        cols = st.columns(2)
        for ax, name in zip(cols, ["fraud_vs_nonfraud.png",
                                   "amount_distribution_by_class.png"]):
            p = GRAPHS_DIR / name
            if p.exists():
                ax.image(str(p), use_container_width=True)

        cols = st.columns(2)
        for ax, name in zip(cols, ["risk_level_distribution.png",
                                   "fraud_probability_distribution.png"]):
            p = GRAPHS_DIR / name
            if p.exists():
                ax.image(str(p), use_container_width=True)

        st.subheader("High-risk transactions (top 50 by score)")
        st.dataframe(
            df.sort_values("risk_score", ascending=False).head(50),
            use_container_width=True,
        )


# ------------------------------------------------------------------
# Page 3: About
# ------------------------------------------------------------------
else:
    st.title("About SparkShield")
    st.markdown(
        """
        **SparkShield** is a distributed fraud-detection pipeline built on
        Apache Spark MLlib (Logistic Regression, Random Forest, Gradient
        Boosted Trees), with a rule-based feature engineering layer and a
        risk-scoring engine.

        ### Stack
        - **PySpark / MLlib** — distributed training + inference
        - **Streamlit** — this UI
        - **Kafka + Spark Structured Streaming** — live scoring
          (`src/kafka_producer.py`, `src/kafka_consumer.py`)
        - **Power BI / Excel** — consume `results/powerbi/*.csv`

        ### Pipelines
        ```
        digital_payments.csv
          → src/main.py            (train + evaluate + score)
          → src/export_predictions.py (batch score)
          → src/visualization.py   (charts → results/graphs/)
          → src/powerbi_export.py  (BI-ready CSVs)
          → kafka_producer + kafka_consumer (live scoring)
        ```
        """
    )
