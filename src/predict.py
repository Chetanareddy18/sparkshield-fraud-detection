import os
import sys

# -----------------------------
# FIX PYTHON PATH
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.append(PROJECT_ROOT)

# -----------------------------
# IMPORTS
# -----------------------------
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

from src.risk_scoring import add_risk_score, add_risk_level


# -----------------------------
# CREATE SPARK SESSION
# -----------------------------
def create_spark():

    spark = (
        SparkSession.builder
        .appName("FraudPrediction")
        .getOrCreate()
    )

    # send risk_scoring.py to workers
    spark.sparkContext.addPyFile(os.path.join(CURRENT_DIR, "risk_scoring.py"))

    return spark


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
def load_model():

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "gbt_model"
    )

    model = GBTClassificationModel.load(model_path)

    return model


# -----------------------------
# PROBABILITY EXTRACTOR
# -----------------------------
def get_probability(v):
    return float(v[1])


prob_udf = udf(get_probability, DoubleType())


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_transaction():

    spark = create_spark()

    model = load_model()

    # --------------------------------
    # SAMPLE TRANSACTION
    # --------------------------------
    data = [Row(

        amount=150000.0,
        oldbalanceOrg=200000.0,
        newbalanceOrig=50000.0,

        oldbalanceDest=0.0,
        newbalanceDest=150000.0,

        balanceError=50000.0,
        destBalanceError=10000.0,

        highValueTransaction=1,
        accountDrained=1

    )]

    df = spark.createDataFrame(data)

    # --------------------------------
    # VECTOR ASSEMBLER
    # --------------------------------
    assembler = VectorAssembler(

        inputCols=[
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "balanceError",
            "destBalanceError",
            "highValueTransaction",
            "accountDrained"
        ],

        outputCol="features"

    )

    df = assembler.transform(df)

    # --------------------------------
    # MODEL PREDICTION
    # --------------------------------
    predictions = model.transform(df)

    predictions = predictions.withColumn(
        "fraud_probability",
        prob_udf("probability")
    )

    # --------------------------------
    # RISK SCORING
    # --------------------------------
    risk_df = add_risk_score(predictions)
    risk_df = add_risk_level(risk_df)

    result = risk_df.select(
        "prediction",
        "fraud_probability",
        "risk_score",
        "risk_level"
    ).collect()[0]

    spark.stop()

    return result


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":

    print("\nRunning Fraud Prediction Test...\n")

    result = predict_transaction()

    print("\nPrediction Result:\n")
    print(result)