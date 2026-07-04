"""
Kafka -> Spark Structured Streaming Consumer
============================================
Subscribes to a Kafka topic of incoming transactions, runs them through
the same feature engineering used in training, scores them with the
saved GBT model, and writes scored events to disk + console.

Usage:
    # 1. start the producer:  python src/kafka_producer.py
    # 2. python src/kafka_consumer.py

Notes:
    - Requires the Spark Kafka package at runtime, e.g.:
        spark-submit --packages \\
          org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \\
          src/kafka_consumer.py
      or set SPARK_KAFKA_PACKAGE before launching.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Force workers to use the SAME interpreter as the driver to avoid
# Spark's PYTHON_VERSION_MISMATCH error.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Convenience: auto-add the Kafka package if not already configured.
PKG = os.environ.get(
    "SPARK_KAFKA_PACKAGE",
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
)
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f"--packages {PKG} pyspark-shell"
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, from_json, when, udf)
from pyspark.sql.types import (DoubleType, IntegerType, StringType, StructField,
                               StructType)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassificationModel


# Schema for the JSON messages produced by kafka_producer.py
TX_SCHEMA = StructType([
    StructField("step", IntegerType()),
    StructField("type", StringType()),
    StructField("amount", DoubleType()),
    StructField("nameOrig", StringType()),
    StructField("oldbalanceOrg", DoubleType()),
    StructField("newbalanceOrig", DoubleType()),
    StructField("nameDest", StringType()),
    StructField("oldbalanceDest", DoubleType()),
    StructField("newbalanceDest", DoubleType()),
    StructField("isFraud", IntegerType()),
    StructField("isFlaggedFraud", IntegerType()),
])

FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balanceError", "destBalanceError",
    "highValueTransaction", "accountDrained",
]


def _engineer(df):
    df = df.withColumn(
        "balanceError",
        col("oldbalanceOrg") - col("newbalanceOrig") - col("amount"),
    )
    df = df.withColumn(
        "destBalanceError",
        col("newbalanceDest") - col("oldbalanceDest") - col("amount"),
    )
    df = df.withColumn(
        "highValueTransaction", when(col("amount") > 200000, 1).otherwise(0),
    )
    df = df.withColumn(
        "accountDrained", when(col("newbalanceOrig") == 0, 1).otherwise(0),
    )
    return df


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
    topic = os.environ.get("KAFKA_TOPIC", "transactions")
    out_dir = ROOT / "results" / "stream_predictions"
    chk_dir = ROOT / "results" / "stream_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("SparkShield-StreamingConsumer")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading GBT model from {ROOT / 'models' / 'gbt_model'} ...")
    model = GBTClassificationModel.load(str(ROOT / "models" / "gbt_model"))

    print(f"Subscribing to Kafka {bootstrap}/{topic} ...")
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json")
           .select(from_json(col("json"), TX_SCHEMA).alias("d"))
           .select("d.*")
    )

    enriched = _engineer(parsed)

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    vectorised = assembler.transform(enriched)

    scored = model.transform(vectorised)

    extract_prob = udf(lambda v: float(v[1]), DoubleType())
    scored = (
        scored.withColumn("fraud_probability", extract_prob(col("probability")))
              .withColumn("risk_score", col("fraud_probability") * 100)
              .withColumn(
                  "risk_level",
                  when(col("risk_score") > 70, "High Risk")
                  .when(col("risk_score") > 40, "Medium Risk")
                  .otherwise("Low Risk"),
              )
    )

    output = scored.select(
        "step", "type", "amount",
        "prediction", "fraud_probability", "risk_score", "risk_level",
    )

    # Console sink for live debugging
    console_q = (
        output.writeStream.format("console")
              .outputMode("append")
              .option("truncate", False)
              .start()
    )

    # Persistent CSV sink for downstream BI
    file_q = (
        output.writeStream.format("csv")
              .option("path", str(out_dir))
              .option("checkpointLocation", str(chk_dir))
              .option("header", True)
              .outputMode("append")
              .start()
    )

    print("Streaming started. Press Ctrl+C to stop.")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
