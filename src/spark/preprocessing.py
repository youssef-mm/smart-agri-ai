from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType
)

spark = SparkSession.builder \
    .appName("Smart Agriculture Streaming") \
    .master("local[*]") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
    ) \
    .getOrCreate()

schema = StructType([
    StructField("Region", StringType(), True),
    StructField("Soil_Type", StringType(), True),
    StructField("Crop", StringType(), True),
    StructField("Rainfall_mm", DoubleType(), True),
    StructField("Temperature_Celsius", DoubleType(), True),
    StructField("Fertilizer_Used", StringType(), True),
    StructField("Irrigation_Used", StringType(), True),
    StructField("Weather_Condition", StringType(), True),
    StructField("Days_to_Harvest", IntegerType(), True),
    StructField("Yield_tons_per_hectare", DoubleType(), True)
])

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "crop_yield_topic") \
    .option("startingOffsets", "latest") \
    .load()

json_df = kafka_df.selectExpr("CAST(value AS STRING) AS json")

df = json_df.select(
    from_json(col("json"), schema).alias("data")
).select("data.*")

df = df.dropDuplicates()
df = df.dropna()

df = df.withColumn(
    "Fertilizer_Used",
    when(col("Fertilizer_Used") == "TRUE", True).otherwise(False)
)

df = df.withColumn(
    "Irrigation_Used",
    when(col("Irrigation_Used") == "TRUE", True).otherwise(False)
)

query = df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()