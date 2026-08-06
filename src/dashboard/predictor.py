import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# 1. إنشاء Spark Session واحدة ثابتة للـ Inference
spark = (
    SparkSession.builder.appName("Streamlit_PySpark_Inference")
    .master("local[*]")
    .getOrCreate()
)

# 2. تحميل موديل PySpark المعتمد المحفوظ
MODEL_PATH = "models/rf_crop_yield_model"


def load_pyspark_model():
    try:
        model = PipelineModel.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading PySpark model: {e}")
        return None


def predict_crop_yield(model, feature_names, payload):
    # تحويل المدخلات إلى Spark DataFrame
    df_input = pd.DataFrame([payload])

    # تحويل المسميات لتطابق أعمدة ملف التدريب (train_mllib.py)
    df_input = df_input.rename(
        columns={
            "Crop_Type": "Crop",
            "Temperature_C": "Temperature_Celsius",
            "Humidity_pct": "Weather_Condition",
        }
    )

    # إضافة الأعمدة الناقصة بقيم افتراضية لتطابق الـ Pipeline Schema
    df_input["Soil_Type"] = "Loamy"
    df_input["Days_to_Harvest"] = 120
    df_input["Fertilizer_Used"] = 1.0
    df_input["Irrigation_Used"] = 1.0
    df_input["Weather_Condition"] = "Normal"

    # تحويل لـ Spark DataFrame والتنبؤ
    spark_df = spark.createDataFrame(df_input)

    if model is not None:
        predictions = model.transform(spark_df)
        pred_val = predictions.select("prediction").collect()[0]["prediction"]
        return round(float(pred_val), 2)
    else:
        return 4.25  # Fallback value