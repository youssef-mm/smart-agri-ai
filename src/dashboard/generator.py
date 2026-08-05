import random
import time

def generate_live_sensor_data():
    """محاكاة قراءة بيانات الحساسات بشكل لحظي"""
    return {
        "Temperature_Celsius": round(random.uniform(15.0, 40.0), 2),
        "Rainfall_mm": round(random.uniform(100.0, 1200.0), 2),
        "Humidity_pct": round(random.uniform(40.0, 90.0), 2),
        "Soil_pH": round(random.uniform(5.5, 7.5), 2),
        "Timestamp": time.strftime("%H:%M:%S")
    }

def fetch_live_weather(city: str = "Cairo") -> dict:
    """جلب أو محاكاة بيانات الطقس الحية للـ Dashboard"""
    return {
        "city": city,
        "temperature": round(random.uniform(20.0, 35.0), 1),
        "Temperature_C": round(random.uniform(20.0, 35.0), 1),
        "humidity": round(random.uniform(40.0, 80.0), 1),
        "Humidity_pct": round(random.uniform(40.0, 80.0), 1),
        "rainfall": round(random.uniform(0.0, 15.0), 1),
        "wind_speed": round(random.uniform(5.0, 25.0), 1),
        "condition": random.choice(["Sunny", "Partly Cloudy", "Clear", "Rainy"]),
        "timestamp": time.strftime("%H:%M:%S")
    }