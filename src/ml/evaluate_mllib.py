from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Crop Yield Model Evaluation") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv("data/crop_yield.csv", header=True, inferSchema=True)
df_cleaned = df.dropDuplicates().dropna()
df_cleaned = df_cleaned.withColumn(
    "Fertilizer_Used",
    when(col("Fertilizer_Used") == "TRUE", 1.0).otherwise(0.0)
).withColumn(
    "Irrigation_Used",
    when(col("Irrigation_Used") == "TRUE", 1.0).otherwise(0.0)
)

_, test_df = df_cleaned.randomSplit([0.8, 0.2], seed=42)

model_path = "models/rf_crop_yield_model"
model = PipelineModel.load(model_path)

predictions = model.transform(test_df)

evaluator_rmse = RegressionEvaluator(labelCol="Yield_tons_per_hectare", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="Yield_tons_per_hectare", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="Yield_tons_per_hectare", predictionCol="prediction", metricName="mae")

print("=" * 60)
print("Model Evaluation Results (On Preprocessed Data):")
print("=" * 60)
print(f"Root Mean Squared Error (RMSE): {evaluator_rmse.evaluate(predictions):.4f}")
print(f"R-squared (R2 Score):            {evaluator_r2.evaluate(predictions):.4f}")
print(f"Mean Absolute Error (MAE):       {evaluator_mae.evaluate(predictions):.4f}")
print("=" * 60)


rmse_value = evaluator_rmse.evaluate(predictions)
r2_value = evaluator_r2.evaluate(predictions)
mae_value = evaluator_mae.evaluate(predictions)

print("=" * 40)
print(f"Model RMSE: {rmse_value}")
print(f"Model R2 Score: {r2_value}")
print(f"Model MAE: {mae_value}")
print("=" * 40)
spark.stop()
