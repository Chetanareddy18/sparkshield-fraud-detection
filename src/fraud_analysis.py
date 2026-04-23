from pyspark.sql.functions import col


def fraud_summary(df):

    print("\nFraud vs Normal Transactions")

    fraud_vs_normal = df.groupBy("isFraud").count()
    fraud_vs_normal.show()

    fraud_vs_normal.write.mode("overwrite").csv(
        "results/fraud_vs_normal", header=True
    )


    print("\nFraud by Transaction Type")

    fraud_by_type = df.groupBy("type", "isFraud").count()
    fraud_by_type.show()

    fraud_by_type.write.mode("overwrite").csv(
        "results/fraud_by_type", header=True
    )


    print("\nHigh Value Fraud Transactions")

    high_value = df.filter(col("highValueTransaction") == 1) \
        .groupBy("isFraud") \
        .count()

    high_value.show()

    high_value.write.mode("overwrite").csv(
        "results/high_value_fraud", header=True
    )


    print("\nAccount Drained Cases")

    drained = df.filter(col("accountDrained") == 1) \
        .groupBy("isFraud") \
        .count()

    drained.show()

    drained.write.mode("overwrite").csv(
        "results/account_drained_cases", header=True
    )