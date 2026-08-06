import os
import joblib
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from sklearn.ensemble import RandomForestRegressor as SklearnRF

# 1. إعداد Spark Session
spark = (
    SparkSession.builder.appName("Smart Agri AI - PySpark MLlib")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

print("Loading data into PySpark DataFrame...")
df = spark.read.csv("data/crop_yield.csv", header=True, inferSchema=True)
df_cleaned = df.dropDuplicates().dropna()
df_sampled = df_cleaned.sample(withReplacement=False, fraction=0.1, seed=42)

df_processed = df_sampled.withColumn(
    "Fertilizer_Used",
    when(col("Fertilizer_Used") == "TRUE", 1.0).otherwise(0.0),
).withColumn(
    "Irrigation_Used",
    when(col("Irrigation_Used") == "TRUE", 1.0).otherwise(0.0),
)

print("Training PySpark MLlib Pipeline...")
# تحويل PySpark DataFrame إلى Pandas بعد معالجتها بـ Spark
pandas_df = df_processed.toPandas()

# فصل Features و Target
X = pandas_df.drop(columns=["Yield_tons_per_hectare"])
y = pandas_df["Yield_tons_per_hectare"]

# Preprocessing بـ Scikit-Learn للحفظ السلس
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder as SkOneHot

categorical_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
numeric_cols = [
    "Rainfall_mm",
    "Temperature_Celsius",
    "Days_to_Harvest",
    "Fertilizer_Used",
    "Irrigation_Used",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", SkOneHot(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

final_pipeline = SkPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", SklearnRF(n_estimators=20, max_depth=8, n_jobs=-1)),
    ]
)

final_pipeline.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(final_pipeline, "models/rf_crop_yield_model.joblib")
print("Model saved successfully as joblib for Streamlit deployment!")

spark.stop()