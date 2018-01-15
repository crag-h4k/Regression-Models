from pyspark.ml.regression import LinearRegression
pp_df = spark.read.csv("/Users/danemorgan/Documents/DataScience/CCPP/powerplant.csv",header="True",inferSchema=True)
pp_df
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["AT","V","AP","RH"], outputCol="features")

vpp_df = vectorAssembler.transform(pp_df)

vpp_df.take(1)

LR = LinearRegression(featuresCol="features",labelCol="PE")
lr_model = LR.fit(vpp_df)

lr_model.coefficients
#should output: DenseVector([-1.9775, -0.2339, 0.0621, -0.1581])
lr_model.intercept
#should output: 454.6092744523414
lr_model.summary.rootMeanSquaredError
#should output: 4.557126016749488
lr_model.save("linearRegression1.model")
