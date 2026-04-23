from pyspark.sql.functions import col


def create_features(df):

    # balance error
    df = df.withColumn(
        "balanceError",
        (col("oldbalanceOrg") - col("newbalanceOrig")) - col("amount")
    )

    # destination balance error
    df = df.withColumn(
        "destBalanceError",
        (col("newbalanceDest") - col("oldbalanceDest")) - col("amount")
    )

    # high value transaction flag
    df = df.withColumn(
        "highValueTransaction",
        (col("amount") > 200000).cast("int")
    )

    # account drained flag
    df = df.withColumn(
        "accountDrained",
        ((col("oldbalanceOrg") > 0) & (col("newbalanceOrig") == 0)).cast("int")
    )

    return df