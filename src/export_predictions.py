import os

# -------------------------------------------------
# FORCE SPARK TO USE YOUR VENV PYTHON
# -------------------------------------------------

os.environ["PYSPARK_PYTHON"] = r"C:\Users\Chetana\OneDrive\Desktop\DAV_LAB\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\Chetana\OneDrive\Desktop\DAV_LAB\venv\Scripts\python.exe"

# -------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.types import DoubleType

# -------------------------------------------------
# START SPARK
# -------------------------------------------------

spark = SparkSession.builder \
    .appName("FraudPredictionExport") \
    .master("local[*]") \
    .getOrCreate()

print("✅ Spark Started")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

data = spark.read.csv(
    "data/digital_payments.csv",
    header=True,
    inferSchema=True
)

print("✅ Dataset Loaded:", data.count())

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------

data = data.withColumn(
    "balanceError",
    col("oldbalanceOrg") - col("newbalanceOrig") - col("amount")
)

data = data.withColumn(
    "destBalanceError",
    col("newbalanceDest") - col("oldbalanceDest") - col("amount")
)

data = data.withColumn(
    "highValueTransaction",
    when(col("amount") > 200000, 1).otherwise(0)
)

data = data.withColumn(
    "accountDrained",
    when(col("newbalanceOrig") == 0, 1).otherwise(0)
)

print("✅ Feature Engineering Completed")

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

print("Loading GBT Model...")

model = GBTClassificationModel.load("models/gbt_model")

print("✅ Model Loaded")

# -------------------------------------------------
# CREATE FEATURE VECTOR
# -------------------------------------------------

features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceError",
    "destBalanceError",
    "highValueTransaction",
    "accountDrained"
]

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

data = assembler.transform(data)

print("✅ Feature Vector Created")

# -------------------------------------------------
# MAKE PREDICTIONS
# -------------------------------------------------

print("Generating predictions...")

predictions = model.transform(data)

print("✅ Predictions Generated")

# -------------------------------------------------
# EXTRACT FRAUD PROBABILITY
# -------------------------------------------------

def extract_prob(v):
    return float(v[1])

prob_udf = udf(extract_prob, DoubleType())

predictions = predictions.withColumn(
    "fraud_probability",
    prob_udf(col("probability"))
)

# -------------------------------------------------
# SELECT FINAL COLUMNS
# -------------------------------------------------

final_df = predictions.select(
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "prediction",
    "fraud_probability"
)

print("✅ Final Columns Selected")

# -------------------------------------------------
# EXPORT RESULTS (SAFE FOR LARGE DATA)
# -------------------------------------------------

print("Exporting predictions...")

final_df.write.mode("overwrite").option("header", True).csv(
    "fraud_predictions"
)

print("✅ Predictions exported successfully!")

spark.stop()

print("✅ Spark Stopped")