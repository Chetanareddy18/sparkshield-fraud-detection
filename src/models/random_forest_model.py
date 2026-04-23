from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_random_forest(train_df):

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="isFraud",
        numTrees=50
    )

    model = rf.fit(train_df)

    return model


def evaluate_random_forest(model, test_df):

    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="isFraud",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    print("Random Forest AUC:", auc)

    return predictions