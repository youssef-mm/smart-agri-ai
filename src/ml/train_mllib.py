import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

# تحديد المسارات بناءً على مجلد المشروع الرئيسي
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "crop_yield.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_crop_yield_model"

spark = SparkSession.builder \
    .appName("Crop Yield Model Training") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)

df_cleaned = df.dropDuplicates().dropna()
df_cleaned = df_cleaned.withColumn(
    "Fertilizer_Used",
    when(col("Fertilizer_Used") == "TRUE", 1.0).otherwise(0.0)
).withColumn(
    "Irrigation_Used",
    when(col("Irrigation_Used") == "TRUE", 1.0).otherwise(0.0)
)

categorical_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
numeric_cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest", "Fertilizer_Used", "Irrigation_Used"]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    for c in categorical_cols
]

feature_cols = [f"{c}_index" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

rf = RandomForestRegressor(featuresCol="features", labelCol="Yield_tons_per_hectare", numTrees=50, seed=42)

pipeline = Pipeline(stages=indexers + [assembler, rf])

print("Training model on preprocessed data...")
model = pipeline.fit(df_cleaned)
print("Model trained successfully!")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.write().overwrite().save(str(MODEL_PATH))
print(f"Model saved successfully at: {MODEL_PATH}")

spark.stop()