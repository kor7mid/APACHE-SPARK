from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import pandas as pd
import numpy as np
# Create a spark context
spark = SparkSession.builder.appName("MlTest").getOrCreate()
sc = spark.sparkContext
def pre_processing(data_file, train=True):
    df = pd.read_csv(data_file)
    embarked = {'C': 0, 'Q': 1, 'S': 2, np.nan: 3}
    cabins = {c: i for i, c in enumerate(df['Cabin'].unique())}
    df['Embarked'] = df['Embarked'].apply(lambda x: embarked[x])
    df['Cabin'] = df['Cabin'].apply(lambda x: cabins[x])
    df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 0)
    if not train:
        df = df.assign(Survived=df['Name'].map(lambda x: 0))
    avoid = ["Name", "Ticket"]
    labelCol = "Survived"
    featuresCols = [c for c in df.columns if not c in [labelCol] + avoid]
    features = df[featuresCols].values
    labels = df[labelCol].values
    data = [LabeledPoint(l, fts) for l, fts in zip(labels, features)]
    return sc.parallelize(data)
with open(SparkFiles.get('titanic_train.csv')) as data_file:
    trainingData = pre_processing(data_file)
with open(SparkFiles.get('titanic_test.csv')) as data_file:
    testingData = pre_processing(data_file, train=False)
# Train a DecisionTree model.
model = DecisionTree.trainClassifier(
    trainingData, numClasses=2, categoricalFeaturesInfo={},
    impurity='gini', maxDepth=3, maxBins=16
)
# Evaluate model on test instances
passenger = [int(p) for p in testingData.map(lambda x: x.features[0]).collect()]
predictions = model.predict(testingData.map(lambda x: x.features)).collect()
predictions = [int(p) for p in predictions]
# Save locally for submission
df = pd.DataFrame([{
    "PassengerId": pas,
    "Survived": pred
} for pas, pred in zip(passenger, predictions)])
df.to_csv("submission.csv", index=False)