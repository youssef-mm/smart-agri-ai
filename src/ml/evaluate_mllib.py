from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import RegressionEvaluator
import joblib
import pandas as pd

# 1. إنشاء Spark Session
spark = SparkSession.builder \
    .appName("PySpark Model Evaluation") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Loading test dataset into PySpark DataFrame...")

# 2. قراءة البيانات بـ PySpark
df_spark = spark.read.csv("data/crop_yield.csv", header=True, inferSchema=True)
df_cleaned = df_spark.dropDuplicates().dropna()

df_processed = df_cleaned.withColumn(
    "Fertilizer_Used",
    when(col("Fertilizer_Used") == "TRUE", 1.0).otherwise(0.0)
).withColumn(
    "Irrigation_Used",
    when(col("Irrigation_Used") == "TRUE", 1.0).otherwise(0.0)
)

# أخذ 20% لاختبار الموديل عبر Spark
_, test_df = df_processed.randomSplit([0.8, 0.2], seed=42)

# تحويل بيانات الاختبار لـ Pandas للتنبؤ بالموديل المحفوظ
test_pd = test_df.toPandas()
X_test = test_pd.drop(columns=["Yield_tons_per_hectare"])

# تحميل الموديل المحفوظ
model = joblib.load("models/rf_crop_yield_model.joblib")
test_pd["prediction"] = model.predict(X_test)

# 3. تحويل النتائج مرة أخرى إلى PySpark DataFrame لاستخدام Spark RegressionEvaluator
predictions_spark = spark.createDataFrame(test_pd)

# 4. حساب المقاييس باستخدام PySpark MLlib Evaluators حصراً
evaluator_rmse = RegressionEvaluator(labelCol="Yield_tons_per_hectare", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="Yield_tons_per_hectare", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="Yield_tons_per_hectare", predictionCol="prediction", metricName="mae")

rmse_val = evaluator_rmse.evaluate(predictions_spark)
r2_val = evaluator_r2.evaluate(predictions_spark)
mae_val = evaluator_mae.evaluate(predictions_spark)

print("=" * 60)
print("Model Evaluation Results (Calculated via PySpark MLlib Evaluator):")
print("=" * 60)
print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
print(f"R-squared (R2 Score):            {r2_val:.4f}")
print(f"Mean Absolute Error (MAE):       {mae_val:.4f}")
print("=" * 60)

spark.stop()