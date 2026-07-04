# SparkShield – Real Power BI Dashboard (5 minute build)

## 1. Open the project in Power BI Desktop

Double-click **`SparkShield.pbids`** in this folder.
Power BI Desktop will open with a Folder connector pre-pointed at
`Fraud_Detection\results\powerbi`. Click **Combine & Load**.

You will get one table per CSV plus an auto-combined table:

| Table | Rows | Use it for |
|-------|------|------------|
| `summary_kpis` | 1 | KPI cards (top of page) |
| `fraud_by_type` | 5 | Fraud-by-type bars |
| `risk_level_summary` | 3 | Risk donut |
| `amount_buckets` | 6 | Amount-bucket bars |
| `transactions_scored` | 500 K | Detail table + slicers |

## 2. Page 1 – Executive Overview

Drop these visuals from the **Visualizations** pane:

| Visual | Field wells |
|--------|-------------|
| **Card** | `summary_kpis[total_transactions]` |
| **Card** | `summary_kpis[predicted_fraud]` |
| **Card** | `summary_kpis[fraud_rate_pct]` (format %, 2 dp) |
| **Card** | `summary_kpis[high_risk_count]` |
| **Card** | `summary_kpis[fraud_amount_usd_m]` (format $, M) |
| **Donut chart** | Legend = `risk_level_summary[risk_level]`, Values = `risk_level_summary[count]` |
| **Stacked bar chart** | Y = `fraud_by_type[type]`, X = `fraud_by_type[fraud_count]` |
| **Clustered column** | X = `amount_buckets[bucket]`, Y = `amount_buckets[fraud_rate_pct]` |

## 3. Page 2 – Transaction Drill-down

| Visual | Field wells |
|--------|-------------|
| **Table** | All cols of `transactions_scored` |
| **Slicer** | `transactions_scored[type]` |
| **Slicer** | `transactions_scored[risk_level]` |
| **Histogram (built-in column)** | `transactions_scored[fraud_probability]` |

## 4. Theme (optional)

View → Themes → Browse for themes → use the built-in **Executive** theme,
or set the page background to `#F2F2F2` and accent colour `#118DFF` for
the SparkShield blue.

## 5. Publish

File → Publish → My workspace.
That uploads a real `.pbix` to the Power BI Service so you can share a
public web link and embed it on LinkedIn.
