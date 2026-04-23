import os
import sys

# =============================
# FIX PYSPARK PYTHON PATH
# =============================
os.environ["PYSPARK_PYTHON"] = r"C:\Users\Chetana\OneDrive\Desktop\DAV_LAB\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\Chetana\OneDrive\Desktop\DAV_LAB\venv\Scripts\python.exe"

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import create_spark_session, load_dataset
from feature_engineering import create_features
from fraud_analysis import fraud_summary
from config.config import DATA_PATH

# ML pipeline
from ml_pipeline import prepare_ml_data, split_data, train_models, evaluate, save_models

# Risk scoring
from risk_scoring import add_risk_score, add_risk_level


def main():

    # =============================
    # START SPARK SESSION
    # =============================
    spark = create_spark_session()

    # Send risk_scoring file to Spark workers
    spark.sparkContext.addPyFile("src/risk_scoring.py")

    print("\nStarting Fraud Detection Pipeline...\n")

    # =============================
    # LOAD DATASET
    # =============================
    df = load_dataset(spark, DATA_PATH)

    df = df.repartition(8)

    print("Total Transactions:", df.count())

    print("\nRaw Dataset Preview:")
    df.show(5)

    # =============================
    # FEATURE ENGINEERING
    # =============================
    print("\nCreating Features...")

    df = create_features(df)

    df.cache()

    print("\nEngineered Features Preview:")

    df.select(
        "amount",
        "balanceError",
        "destBalanceError",
        "highValueTransaction",
        "accountDrained"
    ).show(5)

    # =============================
    # SAVE PROCESSED DATA
    # =============================
    print("\nSaving processed dataset...")

    df.write.mode("overwrite").parquet("results/processed_transactions")

    # =============================
    # FRAUD ANALYSIS
    # =============================
    print("\nRunning Fraud Analysis...")

    fraud_summary(df)

    # =============================
    # MACHINE LEARNING PIPELINE
    # =============================
    print("\nPreparing ML Features...")

    ml_df = prepare_ml_data(df)

    train_df, test_df = split_data(ml_df)

    print("\nTraining ML Models...")

    lr_model, rf_model, gbt_model = train_models(train_df)

    # =============================
    # MODEL EVALUATION
    # =============================
    print("\nModel Evaluation Results\n")

    lr_auc = evaluate(lr_model, test_df)
    rf_auc = evaluate(rf_model, test_df)
    gbt_auc = evaluate(gbt_model, test_df)

    print("Logistic Regression AUC:", lr_auc)
    print("Random Forest AUC:", rf_auc)
    print("Gradient Boosted Trees AUC:", gbt_auc)

    # =============================
    # SAVE MODELS
    # =============================
    print("\nSaving trained models...")

    save_models(lr_model, rf_model, gbt_model)

    # =============================
    # RISK SCORING
    # =============================
    print("\nGenerating Fraud Risk Scores...")

    # Use best model (GBT)
    predictions = gbt_model.transform(test_df)

    risk_df = add_risk_score(predictions)
    risk_df = add_risk_level(risk_df)

    print("\nRisk Scoring Output:")

    risk_df.select(
        "prediction",
        "fraud_probability",
        "risk_score",
        "risk_level"
    ).show(10)

    # =============================
    # CLEAN DATA BEFORE SAVING
    # =============================
    output_df = risk_df.select(
        "prediction",
        "fraud_probability",
        "risk_score",
        "risk_level"
    )

    # =============================
    # SAVE FINAL RISK SCORES
    # =============================
    print("\nSaving final risk scoring results...")

    output_df.write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("results/final_risk_scores")

    print("Final risk scores saved to results/final_risk_scores")

    print("\nPipeline Completed Successfully!")

    spark.stop()


if __name__ == "__main__":
    main()