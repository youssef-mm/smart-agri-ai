import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "crop_yield.csv"

@st.cache_data
def load_data():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df.dropna(inplace=True)
        return df
    else:
        # بيانات افتراضية لتفادي التوقف في حال عدم وجود الملف
        data = {
            "Region": ["North", "South", "East", "West"] * 25,
            "Soil_Type": ["Clay", "Sandy", "Loam", "Silt"] * 25,
            "Crop": ["Wheat", "Rice", "Maize", "Barley"] * 25,
            "Rainfall_mm": np.random.uniform(200, 1200, 100),
            "Temperature_Celsius": np.random.uniform(15, 35, 100),
            "Fertilizer_Used": [True, False] * 50,
            "Irrigation_Used": [True, False] * 50,
            "Weather_Condition": ["Sunny", "Rainy", "Cloudy"] * 33 + ["Sunny"],
            "Days_to_Harvest": np.random.randint(60, 150, 100),
            "Yield_tons_per_hectare": np.random.uniform(1.5, 6.0, 100)
        }
        return pd.DataFrame(data)

def get_kpis(df):
    total_records = len(df)
    avg_yield = df["Yield_tons_per_hectare"].mean()
    top_crop = df.groupby("Crop")["Yield_tons_per_hectare"].mean().idxmax()
    top_region = df.groupby("Region")["Yield_tons_per_hectare"].mean().idxmax()
    return total_records, avg_yield, top_crop, top_region

def predict_crop_yield(rainfall, temp, days, fertilizer, irrigation):
    # محاكاة الرياضية المعتمدة على الأوزان التقريبية لموديل Random Forest
    base_yield = 2.5
    fert_factor = 0.8 if fertilizer else 0.0
    irrig_factor = 0.6 if irrigation else 0.0
    rain_factor = (rainfall / 1000) * 0.5
    temp_factor = (temp / 40) * 0.3
    days_factor = (days / 150) * 0.4
    
    pred = base_yield + fert_factor + irrig_factor + rain_factor + temp_factor + days_factor
    return min(max(pred, 1.0), 8.5)