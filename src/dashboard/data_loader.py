from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_FILE = BASE_DIR / "data" / "processed_crop_yield.csv"
RAW_FILE = BASE_DIR / "data" / "crop_yield.csv"

@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    target_file = PROCESSED_FILE if PROCESSED_FILE.exists() else RAW_FILE
    
    if target_file.exists():
        df = pd.read_csv(target_file)
        df.dropna(inplace=True)
        
        rename_map = {
            "Crop": "Crop_Type",
            "Temperature_Celsius": "Temperature_C",
            "Yield_tons_per_hectare": "Yield_Tons_ha"
        }
        df.rename(columns=rename_map, inplace=True)
        
        if "Date" not in df.columns:
            df["Date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='min')
            
        np.random.seed(42)
        if "Humidity_pct" not in df.columns:
            df["Humidity_pct"] = np.random.uniform(30.0, 85.0, len(df)).round(1)
        if "Soil_pH" not in df.columns:
            df["Soil_pH"] = np.random.uniform(5.5, 8.2, len(df)).round(2)
        if "NPK_Score" not in df.columns:
            df["NPK_Score"] = np.random.randint(50, 100, len(df))
            
        return df
    else:
        # Fallback dataset generation for Cloud Deployment
        np.random.seed(42)
        n = 1000
        crops = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Barley"]
        regions = ["North", "South", "East", "West"]
        
        sample_crops = np.random.choice(crops, size=n)
        sample_regions = np.random.choice(regions, size=n)
        temp = np.random.uniform(18.0, 38.0, size=n).round(1)
        rain = np.random.uniform(100.0, 1000.0, size=n).round(1)
        hum = np.random.uniform(35.0, 85.0, size=n).round(1)
        ph = np.random.uniform(5.5, 7.8, size=n).round(2)
        npk = np.random.randint(55, 95, size=n)
        yield_calc = (2.2 + (rain / 500.0) * 1.1 + (temp / 25.0) * 0.7 + (npk / 100.0) * 1.4).round(2)
        
        df = pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h"),
            "Crop_Type": sample_crops,
            "Crop": sample_crops,
            "Region": sample_regions,
            "Temperature_C": temp,
            "Temperature_Celsius": temp,
            "Rainfall_mm": rain,
            "Humidity_pct": hum,
            "Soil_pH": ph,
            "NPK_Score": npk,
            "Yield_Tons_ha": yield_calc,
            "Yield_tons_per_hectare": yield_calc,
            "Weather_Condition": np.random.choice(["Sunny", "Rainy", "Cloudy"], size=n),
            "Soil_Type": np.random.choice(["Loamy", "Clay", "Sandy"], size=n),
            "Days_to_Harvest": np.random.randint(70, 150, size=n),
            "Fertilizer_Used": np.random.choice([0.0, 1.0], size=n),
            "Irrigation_Used": np.random.choice([0.0, 1.0], size=n)
        })
        return df

def optimize_memory(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df        