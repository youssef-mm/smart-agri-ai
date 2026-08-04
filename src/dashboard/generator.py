import pandas as pd
import numpy as np
import requests
import streamlit as st
from typing import Tuple, Dict, Any

@st.cache_data(ttl=3600, show_spinner=False)
def generate_agri_data(n_samples: int = 400) -> pd.DataFrame:
    np.random.seed(42)
    end_date = pd.Timestamp.now()
    dates = pd.date_range(end=end_date, periods=n_samples, freq='D')
    
    crops = ["Wheat", "Rice", "Corn", "Soybeans", "Cotton"]
    regions = ["Delta", "Upper Egypt", "Alexandria", "Fayoum"]
    
    crop_choices = np.random.choice(crops, n_samples)
    region_choices = np.random.choice(regions, n_samples)
    rainfall = np.random.randint(40, 350, n_samples)
    temp = np.random.uniform(18.0, 42.0, n_samples).round(1)
    humidity = np.random.uniform(30.0, 85.0, n_samples).round(1)
    soil_ph = np.random.uniform(5.5, 8.2, n_samples).round(2)
    npk_score = np.random.randint(50, 100, n_samples)
    
    yield_val = (
        2.0 + 
        (rainfall * 0.008) + 
        (npk_score * 0.03) + 
        ((7.0 - np.abs(soil_ph - 6.5)) * 0.25) - 
        (np.abs(temp - 28.0) * 0.07) + 
        np.random.normal(0, 0.35, n_samples)
    )
    yield_val = np.clip(yield_val, 1.2, 9.5).round(2)
    
    return pd.DataFrame({
        "Date": dates,
        "Crop_Type": crop_choices,
        "Region": region_choices,
        "Rainfall_mm": rainfall,
        "Temperature_C": temp,
        "Humidity_pct": humidity,
        "Soil_pH": soil_ph,
        "NPK_Score": npk_score,
        "Yield_Tons_ha": yield_val
    })

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather(city: str, api_key: str = "") -> Tuple[Dict[str, Any], bool]:
    if not api_key:
        return _synthetic_weather_fallback(city), False
        
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},EG&units=metric&appid={api_key}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return {
                "Temperature_C": round(data["main"]["temp"], 1),
                "Humidity_pct": round(data["main"]["humidity"], 1),
                "City": city,
                "Source": "OpenWeatherMap Live API"
            }, True
    except Exception:
        pass
        
    return _synthetic_weather_fallback(city), False

def _synthetic_weather_fallback(city: str) -> Dict[str, Any]:
    np.random.seed(hash(city) % 1000)
    return {
        "Temperature_C": round(np.random.uniform(22.0, 38.0), 1),
        "Humidity_pct": round(np.random.uniform(35.0, 75.0), 1),
        "City": city,
        "Source": "Synthetic Engine"
    }