from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Spark SQL Analytics") \
    .getOrCreate()

df = spark.read.csv(
    "data/crop_yield.csv",
    header=True,
    inferSchema=True
)

df.createOrReplaceTempView("crop_data")

print("=" * 70)
print("1. Average Yield by Crop")
print("=" * 70)

spark.sql("""
SELECT Crop,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Crop
ORDER BY Avg_Yield DESC
""").show()


print("=" * 70)
print("2. Average Yield by Region")
print("=" * 70)

spark.sql("""
SELECT Region,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Region
ORDER BY Avg_Yield DESC
""").show()


print("=" * 70)
print("3. Average Yield by Weather")
print("=" * 70)

spark.sql("""
SELECT Weather_Condition,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Weather_Condition
ORDER BY Avg_Yield DESC
""").show()


print("=" * 70)
print("4. Effect of Fertilizer on Yield")
print("=" * 70)

spark.sql("""
SELECT Fertilizer_Used,
COUNT(*) AS Total_Records,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Fertilizer_Used
ORDER BY Avg_Yield DESC
""").show()


print("=" * 70)
print("5. Effect of Irrigation on Yield")
print("=" * 70)

spark.sql("""
SELECT Irrigation_Used,
COUNT(*) AS Total_Records,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Irrigation_Used
ORDER BY Avg_Yield DESC
""").show()


print("=" * 70)
print("6. Top 10 Highest Yield Records")
print("=" * 70)

spark.sql("""
SELECT Crop,
Region,
Weather_Condition,
Yield_tons_per_hectare
FROM crop_data
ORDER BY Yield_tons_per_hectare DESC
LIMIT 10
""").show()


print("=" * 70)
print("7. Best Crop in Each Region")
print("=" * 70)

spark.sql("""
SELECT Crop,
Region,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Crop, Region
ORDER BY Avg_Yield DESC
LIMIT 10
""").show()


print("=" * 70)
print("8. Average Temperature & Rainfall per Crop")
print("=" * 70)

spark.sql("""
SELECT Crop,
ROUND(AVG(Temperature_Celsius),2) AS Avg_Temperature,
ROUND(AVG(Rainfall_mm),2) AS Avg_Rainfall
FROM crop_data
GROUP BY Crop
ORDER BY Avg_Temperature DESC
""").show()


print("=" * 70)
print("9. Number of Records by Weather Condition")
print("=" * 70)

spark.sql("""
SELECT Weather_Condition,
COUNT(*) AS Total_Records
FROM crop_data
GROUP BY Weather_Condition
ORDER BY Total_Records DESC
""").show()


print("=" * 70)
print("10. Weather & Fertilizer Effect on Yield")
print("=" * 70)

spark.sql("""
SELECT Weather_Condition,
Fertilizer_Used,
ROUND(AVG(Yield_tons_per_hectare),2) AS Avg_Yield
FROM crop_data
GROUP BY Weather_Condition, Fertilizer_Used
ORDER BY Avg_Yield DESC
""").show()


spark.stop()