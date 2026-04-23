import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==============================
# LOAD FINAL RISK DATA
# ==============================

files = glob.glob("results/final_risk_scores/*.csv")

df_list = [pd.read_csv(file) for file in files]

df = pd.concat(df_list, ignore_index=True)

print("Dataset Loaded:", df.shape)

# ==============================
# CREATE GRAPH DIRECTORY
# ==============================

os.makedirs("results/graphs", exist_ok=True)

# ==============================
# GRAPH 1 - Fraud vs Non Fraud
# ==============================

fraud_counts = df["prediction"].value_counts()

plt.figure()
fraud_counts.plot(kind="bar")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Prediction")
plt.ylabel("Count")

plt.savefig("results/graphs/fraud_vs_nonfraud.png")
plt.close()

# ==============================
# GRAPH 2 - Risk Level Distribution
# ==============================

plt.figure()
df["risk_level"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Risk Level Distribution")
plt.ylabel("")

plt.savefig("results/graphs/risk_level_distribution.png")
plt.close()

# ==============================
# GRAPH 3 - Fraud Probability Distribution
# ==============================

plt.figure()
plt.hist(df["fraud_probability"], bins=30)

plt.title("Fraud Probability Distribution")
plt.xlabel("Fraud Probability")
plt.ylabel("Transactions")

plt.savefig("results/graphs/fraud_probability_distribution.png")
plt.close()

# ==============================
# GRAPH 4 - Risk Score Distribution
# ==============================

plt.figure()
plt.hist(df["risk_score"], bins=30)

plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Transactions")

plt.savefig("results/graphs/risk_score_distribution.png")
plt.close()

# ==============================
# GRAPH 5 - Model Performance Comparison
# ==============================

models = ["Logistic Regression", "Random Forest", "Gradient Boosted Trees"]
auc_scores = [0.993, 0.9990, 0.99954]

plt.figure()
plt.bar(models, auc_scores)

plt.title("Model Performance Comparison (AUC)")
plt.ylabel("AUC Score")

plt.savefig("results/graphs/model_comparison.png")
plt.close()

print("\nAll graphs generated successfully!")
print("Graphs saved in: results/graphs")