from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_gbt(train_df):

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="isFraud",
        maxIter=20
    )

    model = gbt.fit(train_df)

    return model


def evaluate_gbt(model, test_df):

    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="isFraud",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    print("Gradient Boosted Trees AUC:", auc)

    return predictions