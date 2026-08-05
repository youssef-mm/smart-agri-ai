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

rmse_val = evaluator_rmse.evaluate(predictions)
r2_val = evaluator_r2.evaluate(predictions)
mae_val = evaluator_mae.evaluate(predictions)

print("=" * 60)
print("Model Evaluation Results (On Preprocessed Data):")
print("=" * 60)
print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
print(f"R-squared (R2 Score):           {r2_val:.4f}")
print(f"Mean Absolute Error (MAE):       {mae_val:.4f}")
print("=" * 60)

spark.stop()
