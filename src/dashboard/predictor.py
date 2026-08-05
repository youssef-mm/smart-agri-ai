import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Any, Tuple

@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(df: pd.DataFrame) -> Tuple[Any, Dict[str, float], pd.DataFrame, list]:
    # أخذ عينة 10,000 صف للأداء اللحظي المبتكر
    if len(df) > 10000:
        df_ml = df.sample(n=10000, random_state=42)
    else:
        df_ml = df.copy()

    features = ["Rainfall_mm", "Temperature_C", "Humidity_pct", "Soil_pH", "NPK_Score", "Crop_Type", "Region"]
    X = pd.get_dummies(df_ml[features], columns=["Crop_Type", "Region"], drop_first=False)
    y = df_ml["Yield_Tons_ha"]
    
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_preds = rf.predict(X_test)
    
    metrics = {
        "Model": "RandomForest Regressor",
        "R2": round(float(r2_score(y_test, rf_preds)), 3),
        "MAE": round(float(mean_absolute_error(y_test, rf_preds)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, rf_preds))), 3)
    }
    
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return rf, metrics, fi_df, feature_names

def predict_crop_yield(model: Any, feature_names: list, input_data: Dict[str, Any]) -> float:
    input_df = pd.DataFrame([input_data])
    encoded_df = pd.get_dummies(input_df, columns=["Crop_Type", "Region"], drop_first=False)
    aligned_df = encoded_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(aligned_df)[0]
    return float(round(prediction, 2))