from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "crop_yield.csv"

@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df.dropna(inplace=True)
        
        rename_map = {
            "Crop": "Crop_Type",
            "Temperature_Celsius": "Temperature_C",
            "Yield_tons_per_hectare": "Yield_Tons_ha"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # إنشاء نطاق زمني منطقي وسريع (السنة الأخيرة فقط بدلاً من 100 سنة)
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
        raise FileNotFoundError(f"لم يتم العثور على الملف: {DATA_FILE}")