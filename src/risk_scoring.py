from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType


# extract fraud probability from probability vector
def extract_prob(v):
    return float(v[1])


prob_udf = udf(extract_prob, DoubleType())


def add_risk_score(predictions):

    df = predictions.withColumn(
        "fraud_probability",
        prob_udf(col("probability"))
    )

    # risk score 0–100
    df = df.withColumn(
        "risk_score",
        col("fraud_probability") * 100
    )

    return df


def classify_risk(score):

    if score <= 40:
        return "Low Risk"
    elif score <= 70:
        return "Medium Risk"
    else:
        return "High Risk"


risk_udf = udf(classify_risk, StringType())


def add_risk_level(df):

    df = df.withColumn(
        "risk_level",
        risk_udf(col("risk_score"))
    )

    return df