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
dt = DecisionTreeRegressor(featuresCol="features",labelCol="PE")
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction",metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
rmse
#should output : 4.42834426084713
