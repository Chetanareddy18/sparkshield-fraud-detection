"""
Power BI-style interactive HTML dashboard
=========================================
Builds a single self-contained HTML file using Plotly that mimics the
look-and-feel of a Power BI report. Reads the small CSVs produced by
`src/powerbi_export.py` plus the scored predictions, and writes:

    results/dashboard.html

Open it directly in any browser — no Power BI Desktop required.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "fraud_predictions"
PBI_DIR = ROOT / "results" / "powerbi"
OUT = ROOT / "results" / "dashboard.html"

PRIMARY = "#118DFF"     # Power BI blue
DANGER = "#E81123"      # high risk red
WARN = "#FFB900"        # medium risk yellow
SAFE = "#00B294"        # low risk teal
BG = "#F3F2F1"
CARD = "#FFFFFF"
TEXT = "#252423"


def _read_spark_csv(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "part-*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


# ---------------------------------------------------------------- load
print("Loading data ...")
pred = _read_spark_csv(PRED_DIR)
if pred.empty:
    raise SystemExit("No fraud_predictions/. Run src/export_predictions.py first.")

pred["risk_score"] = pred["fraud_probability"] * 100
pred["risk_level"] = pd.cut(
    pred["risk_score"],
    bins=[-0.01, 40, 70, 100.01],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

kpis = pd.read_csv(PBI_DIR / "summary_kpis.csv").iloc[0]
fbt = pd.read_csv(PBI_DIR / "fraud_by_type.csv")
risk_summary = pd.read_csv(PBI_DIR / "risk_level_summary.csv")
buckets = pd.read_csv(PBI_DIR / "amount_buckets.csv")


# ---------------------------------------------------------------- charts
def _layout(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.02, font=dict(size=14, color=TEXT)),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        margin=dict(l=40, r=20, t=45, b=40),
        font=dict(family="Segoe UI, Arial", color=TEXT, size=12),
        showlegend=False,
    )


# 1) class counts donut
counts = pred["prediction"].value_counts().sort_index()
fig_donut = go.Figure(
    go.Pie(
        labels=["Not Fraud", "Fraud"][: len(counts)],
        values=counts.values,
        hole=0.62,
        marker=dict(colors=[SAFE, DANGER]),
        textinfo="label+percent",
        textfont_size=12,
    )
)
fig_donut.update_layout(**_layout("Fraud vs Non-Fraud (predicted)"))


# 2) fraud by type stacked bar
fbt_sorted = fbt.sort_values("total", ascending=False)
fig_type = go.Figure()
fig_type.add_bar(
    name="Not Fraud", x=fbt_sorted["type"], y=fbt_sorted["not_fraud"],
    marker_color=SAFE,
)
fig_type.add_bar(
    name="Fraud", x=fbt_sorted["type"], y=fbt_sorted["fraud"],
    marker_color=DANGER,
)
fig_type.update_layout(
    barmode="stack", showlegend=True, yaxis_type="log",
    legend=dict(orientation="h", y=1.12, x=0),
    **{k: v for k, v in _layout("Transactions by Type (log)").items()
       if k != "showlegend"},
)


# 3) fraud rate by type
fig_rate = go.Figure(
    go.Bar(
        x=fbt_sorted["type"], y=fbt_sorted["fraud_rate_%"],
        marker_color=DANGER,
        text=[f"{v:.2f}%" for v in fbt_sorted["fraud_rate_%"]],
        textposition="outside",
    )
)
fig_rate.update_layout(yaxis_title="Fraud rate (%)",
                       **_layout("Fraud Rate by Type"))


# 4) risk level
order = ["Low Risk", "Medium Risk", "High Risk"]
risk_summary = risk_summary.set_index("risk_level").reindex(order).reset_index()
fig_risk = go.Figure(
    go.Bar(
        x=risk_summary["risk_level"], y=risk_summary["transactions"],
        marker_color=[SAFE, WARN, DANGER],
        text=[f"{int(v):,}" for v in risk_summary["transactions"]],
        textposition="outside",
    )
)
fig_risk.update_layout(yaxis_type="log",
                       **_layout("Risk Level Distribution (log)"))


# 5) probability histogram (pre-binned to keep HTML small)
hist_counts, hist_edges = np.histogram(pred["fraud_probability"], bins=40)
hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
fig_prob = go.Figure(
    go.Bar(
        x=hist_centers, y=hist_counts,
        marker_color=PRIMARY,
        width=(hist_edges[1] - hist_edges[0]) * 0.95,
    )
)
fig_prob.add_vline(x=0.5, line=dict(color=DANGER, dash="dash"))
fig_prob.update_layout(yaxis_type="log", xaxis_title="Fraud probability",
                       **_layout("Fraud Probability (log y)"))


# 6) amount-bucket fraud rate
fig_bucket = go.Figure(
    go.Bar(
        x=buckets["amount_bucket"], y=buckets["fraud_rate_%"],
        marker_color=DANGER,
        text=[f"{v:.2f}%" for v in buckets["fraud_rate_%"]],
        textposition="outside",
    )
)
fig_bucket.update_layout(yaxis_title="Fraud rate (%)",
                         **_layout("Fraud Rate by Amount Bucket"))


# ---------------------------------------------------------------- HTML
def _kpi_card(title: str, value: str, color: str = PRIMARY) -> str:
    return f"""
    <div class="kpi">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
    </div>
    """


def _chart_div(fig, fid: str) -> str:
    return fig.to_html(
        include_plotlyjs=False, full_html=False, div_id=fid,
        config={"displaylogo": False, "responsive": True},
    )


total = int(kpis["total_transactions"])
fraud_n = int(kpis["predicted_fraud"])
fraud_rate = float(kpis["fraud_rate_%"])
high_n = int(kpis["high_risk_transactions"])
fraud_amt = float(kpis["fraud_amount"])
total_amt = float(kpis["total_amount"])

html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SparkShield — Fraud Detection Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:{BG}; font-family:'Segoe UI', Arial, sans-serif; color:{TEXT}; }}
    header {{
      background: linear-gradient(90deg, #003C71 0%, {PRIMARY} 100%);
      color:white; padding:18px 28px; display:flex; align-items:center; justify-content:space-between;
    }}
    header h1 {{ margin:0; font-size:20px; font-weight:600; }}
    header .sub {{ font-size:12px; opacity:0.85; }}
    .container {{ padding: 18px 24px; }}
    .kpi-row {{ display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; margin-bottom:16px; }}
    .kpi {{
      background:{CARD}; border-radius:6px; padding:14px 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }}
    .kpi-title {{ font-size:12px; color:#605E5C; text-transform:uppercase; letter-spacing:.5px; }}
    .kpi-value {{ font-size:24px; font-weight:600; margin-top:6px; }}
    .grid {{ display:grid; gap:14px; }}
    .grid.two {{ grid-template-columns: 1fr 1fr; }}
    .grid.three {{ grid-template-columns: 1fr 1fr 1fr; }}
    .card {{
      background:{CARD}; border-radius:6px; padding:6px 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,.08); min-height:340px;
    }}
    footer {{
      padding:12px 24px; font-size:11px; color:#605E5C; text-align:center;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>🛡️ SparkShield — Fraud Detection</h1>
      <div class="sub">Spark MLlib · Gradient Boosted Trees · Risk Scoring · Kafka stream-ready</div>
    </div>
    <div class="sub">Power BI–style interactive report</div>
  </header>

  <div class="container">
    <div class="kpi-row">
      {_kpi_card("Total Transactions", f"{total:,}")}
      {_kpi_card("Predicted Fraud", f"{fraud_n:,}", DANGER)}
      {_kpi_card("Fraud Rate", f"{fraud_rate:.3f}%", DANGER)}
      {_kpi_card("High-Risk Cases", f"{high_n:,}", WARN)}
      {_kpi_card("Fraud Amount", f"${fraud_amt/1e6:,.1f}M", DANGER)}
    </div>

    <div class="grid two">
      <div class="card">{_chart_div(fig_donut, "g1")}</div>
      <div class="card">{_chart_div(fig_type, "g2")}</div>
    </div>
    <div style="height:14px"></div>
    <div class="grid three">
      <div class="card">{_chart_div(fig_rate, "g3")}</div>
      <div class="card">{_chart_div(fig_risk, "g4")}</div>
      <div class="card">{_chart_div(fig_bucket, "g5")}</div>
    </div>
    <div style="height:14px"></div>
    <div class="card">{_chart_div(fig_prob, "g6")}</div>
  </div>

  <footer>
    Generated by <code>src/build_dashboard.py</code> · Total amount processed
    ${total_amt/1e9:,.2f}B · Best model AUC = 0.9995 (GBT)
  </footer>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:,.1f} KB)")
print("Open it in any browser to view the interactive dashboard.")
