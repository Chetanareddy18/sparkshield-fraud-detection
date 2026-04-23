# SparkShield — Digital Payment Fraud Detection

An end-to-end **Big Data fraud detection pipeline** built with **Apache Spark MLlib**, delivering scalable feature engineering, multi-model training, fraud risk scoring, and an interactive **Streamlit dashboard** for real-time transaction screening.

> Detects fraudulent digital payment transactions on millions of records using distributed Spark ML, classifies risk levels (Low / Medium / High), and ships predictions to a UI and Power BI ready outputs.

---

## Highlights

- **Distributed ML on Apache Spark** — handles the full PaySim-style digital payments dataset (~6M transactions) using Spark DataFrames + MLlib.
- **Three trained classifiers** — Logistic Regression, Random Forest, and Gradient Boosted Trees (best model used for inference).
- **Behavioral feature engineering** — `balanceError`, `destBalanceError`, `highValueTransaction`, `accountDrained`.
- **Risk Scoring Engine** — converts model probabilities to a 0–100 risk score and Low/Medium/High risk tiers.
- **Streamlit dashboard** for single-transaction fraud screening.
- **Batch export pipeline** writing predictions to CSV for BI tools (Power BI ready).
- **Reproducible artifacts** — saved Spark ML models in `models/` and analytics graphs in `results/graphs/`.

---

## Model Performance (AUC-ROC)

| Model | AUC |
|---|---|
| Logistic Regression | 0.993 |
| Random Forest | 0.999 |
| **Gradient Boosted Trees (deployed)** | **0.9995** |

GBT is used as the production model for inference and risk scoring.

---

## Architecture

```
        ┌─────────────────────┐
        │ digital_payments.csv│
        └──────────┬──────────┘
                   │ (Spark CSV reader)
                   ▼
        ┌─────────────────────┐
        │ Feature Engineering │  balanceError, destBalanceError,
        │  (Spark DataFrame)  │  highValueTransaction, accountDrained
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │   Spark MLlib       │  LR  |  RF  |  GBT
        │  Train / Evaluate   │  (BinaryClassificationEvaluator)
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │   Risk Scoring      │  fraud_probability → 0–100 score
        │   + Risk Levels     │  Low / Medium / High
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────────────┐
        │ Outputs                     │
        │  • results/final_risk_scores│
        │  • fraud_predictions/       │
        │  • results/graphs/*.png     │
        │  • Streamlit dashboard      │
        └─────────────────────────────┘
```

---

## Project Structure

```
Fraud_Detection/
├── config/
│   └── config.py                 # Dataset path config
├── data/
│   └── digital_payments.csv      # (gitignored — large)
├── models/                       # Saved Spark ML models
│   ├── logistic_model/
│   ├── random_forest_model/
│   └── gbt_model/
├── results/
│   ├── graphs/                   # Generated PNG analytics
│   └── ...                       # Spark CSV/parquet outputs (gitignored)
├── src/
│   ├── data_loader.py            # Spark session + CSV loader
│   ├── feature_engineering.py    # Domain features
│   ├── fraud_analysis.py         # Aggregations / fraud summaries
│   ├── ml_pipeline.py            # Train LR / RF / GBT + evaluation
│   ├── risk_scoring.py           # Probability → risk score / level
│   ├── visualization.py          # Pandas/Matplotlib graphs
│   ├── export_predictions.py     # Batch predictions → CSV
│   ├── predict.py                # Single-transaction inference
│   └── main.py                   # End-to-end pipeline entry point
├── dashboard.py                  # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Dataset

PaySim-style **digital payments dataset** with the following key columns:

`step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`

The CSV (~470 MB) is **not committed**. Place it locally at:

```
data/digital_payments.csv
```

You can use the public PaySim dataset from Kaggle (Synthetic Financial Datasets For Fraud Detection).

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Chetanareddy18/sparkshield-fraud-detection.git
cd sparkshield-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Java 8+** installed and `JAVA_HOME` configured (PySpark requirement).

### 4. Add the dataset

Drop `digital_payments.csv` into the `data/` folder.

---

## Usage

### Run the full pipeline (train + evaluate + score + export)

```bash
python src/main.py
```

This will:
1. Start a Spark session
2. Load and feature-engineer the data
3. Run fraud analytics (saved to `results/`)
4. Train Logistic Regression, Random Forest, and GBT
5. Print AUC for each model
6. Save trained models to `models/`
7. Generate fraud risk scores and write to `results/final_risk_scores/`

### Predict a single transaction

```bash
python src/predict.py
```

### Export batch predictions for BI

```bash
python src/export_predictions.py
```

Outputs partitioned CSVs to `fraud_predictions/`.

### Generate analytics graphs

```bash
python src/visualization.py
```

PNG charts are written to `results/graphs/`.

### Launch the Streamlit dashboard

```bash
streamlit run dashboard.py
```

Enter transaction details (amount, balances, flags) and get an instant fraud prediction with probability and risk level.

---

## Engineered Features

| Feature | Definition | Why it matters |
|---|---|---|
| `balanceError` | `oldbalanceOrg − newbalanceOrig − amount` | Detects ledger inconsistencies on the sender side |
| `destBalanceError` | `newbalanceDest − oldbalanceDest − amount` | Detects ledger inconsistencies on the receiver side |
| `highValueTransaction` | `1 if amount > 200,000 else 0` | Captures large transfer risk |
| `accountDrained` | `1 if oldbalanceOrg > 0 and newbalanceOrig == 0` | Strong fraud signal — full account sweep |

---

## Risk Scoring

```
risk_score    = fraud_probability * 100
risk_level    = Low Risk      (score ≤ 40)
              | Medium Risk   (40 < score ≤ 70)
              | High Risk     (score > 70)
```

---

## Tech Stack

- **Apache Spark / PySpark** — distributed processing & MLlib
- **Spark MLlib** — Logistic Regression, Random Forest, Gradient Boosted Trees
- **Pandas / NumPy / Matplotlib** — analytics & visualization
- **Streamlit** — real-time prediction UI
- **Power BI** — downstream dashboarding (CSV export)

---

## Results Snapshot

Generated charts in `results/graphs/`:

- `fraud_vs_nonfraud.png`
- `risk_level_distribution.png`
- `fraud_probability_distribution.png`
- `risk_score_distribution.png`
- `model_comparison.png`

---

## Roadmap

- [ ] SHAP-based explainability for flagged transactions
- [ ] Streaming inference with Spark Structured Streaming + Kafka
- [ ] REST API (FastAPI) wrapping the GBT model
- [ ] Containerization (Docker) and CI/CD
- [ ] Auto-retraining workflow with MLflow tracking

---

## Author

**Chetana Reddy** — [GitHub @Chetanareddy18](https://github.com/Chetanareddy18)

---

## License

Released under the MIT License.
