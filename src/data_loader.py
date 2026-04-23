from pyspark.sql import SparkSession


def create_spark_session():
    """
    Create and return a Spark session with optimized settings
    """

    spark = (
        SparkSession.builder
        .appName("DigitalPaymentFraudDetection")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )

    return spark


def load_dataset(spark, path):
    """
    Load the digital payment dataset
    """

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )

    return df