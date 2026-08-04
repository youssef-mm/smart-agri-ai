from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("Crop Yield Prediction Training") \
    .getOrCreate()

# 2. Read Dataset
data_path = "data/crop_yield.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# 3. Categorical Columns Indexing
categorical_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
    for col in categorical_cols
]

# 4. Feature Assembly
feature_cols = [f"{col}_index" for col in categorical_cols] + [
    'Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 
    'Irrigation_Used', 'Days_to_Harvest'
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 5. Model Definition
rf = RandomForestRegressor(featuresCol="features", labelCol="Yield_tons_per_hectare")

# 6. Pipeline Setup
pipeline = Pipeline(stages=indexers + [assembler, rf])

# 7. Train Test Split & Fitting
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print("Training model...")
model = pipeline.fit(train_data)

# 8. Save Model
model_save_path = "models/crop_yield_model"
model.write().overwrite().save(model_save_path)

print(f"Model trained and saved successfully at '{model_save_path}'!")
spark.stop()
