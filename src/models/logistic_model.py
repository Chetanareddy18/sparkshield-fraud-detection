from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_logistic(train_df):

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="isFraud"
    )

    model = lr.fit(train_df)

    return model


def evaluate_logistic(model, test_df):

    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="isFraud",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    print("Logistic Regression AUC:", auc)

    return predictions