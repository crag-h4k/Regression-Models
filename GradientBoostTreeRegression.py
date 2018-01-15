from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

pp_df = spark.read.csv("/Users/danemorgan/Documents/DataScience/CCPP/powerplant.csv",header=True,inferSchema=True)

pp_df.take(1)
vectorAssembler=VectorAssembler(inputCols=["AT","V","AP","RH"],outputCol="features")

vpp_df = vectorAssembler.transform(pp_df)
splits = vpp_df.randomSplit([0.7,0.3])

train_df = splits[0]
train_df.count()

test_df = splits[1]
gbt = GBTRegressor(featuresCol="features",labelCol="PE")
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)

gbt_evaluator = RegressionEvaluator(labelCol="PE",predictionCol="prediction",metricName="rmse")
gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)

gbt_rmse
#output will be around 4.04


