from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def prepare_ml_data(df):

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

    df = assembler.transform(df)

    return df.select("features", "isFraud")


def split_data(df):

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    print("Train size:", train.count())
    print("Test size:", test.count())

    return train, test


def train_models(train):

    lr = LogisticRegression(featuresCol="features", labelCol="isFraud")
    rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", numTrees=50)
    gbt = GBTClassifier(featuresCol="features", labelCol="isFraud", maxIter=20)

    print("Training Logistic Regression...")
    lr_model = lr.fit(train)

    print("Training Random Forest...")
    rf_model = rf.fit(train)

    print("Training Gradient Boosted Trees...")
    gbt_model = gbt.fit(train)

    return lr_model, rf_model, gbt_model


def evaluate(model, test):

    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(
        labelCol="isFraud",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    return auc


def save_models(lr_model, rf_model, gbt_model):

    lr_model.write().overwrite().save("models/logistic_model")
    rf_model.write().overwrite().save("models/random_forest_model")
    gbt_model.write().overwrite().save("models/gbt_model")

    print("Models saved successfully!")