from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("Crop Yield Prediction Evaluation") \
    .getOrCreate()

# 2. Read Dataset & Split
data_path = "data/crop_yield.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

_, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 3. Load Saved Model
model_path = "models/crop_yield_model"
model = PipelineModel.load(model_path)

# 4. Generate Predictions
predictions = model.transform(test_data)

# 5. Evaluate Performance
evaluator_rmse = RegressionEvaluator(
    labelCol="Yield_tons_per_hectare", 
    predictionCol="prediction", 
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="Yield_tons_per_hectare", 
    predictionCol="prediction", 
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n--- Model Evaluation Results ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (R2): {r2:.4f}")
print("--------------------------------\n")

spark.stop()