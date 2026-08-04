import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from typing import Dict, Any, Tuple

def train_and_evaluate_models(df: pd.DataFrame) -> Tuple[Any, Dict[str, float], pd.DataFrame, list]:
    features = ["Rainfall_mm", "Temperature_C", "Humidity_pct", "Soil_pH", "NPK_Score", "Crop_Type", "Region"]
    X = pd.get_dummies(df[features], columns=["Crop_Type", "Region"], drop_first=False)
    y = df["Yield_Tons_ha"]
    
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    
    rf_preds = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)
    
    rf_r2 = r2_score(y_test, rf_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    
    if xgb_r2 > rf_r2:
        best_model = xgb
        best_preds = xgb_preds
        model_name = "XGBoost Regressor"
    else:
        best_model = rf
        best_preds = rf_preds
        model_name = "RandomForest Regressor"
        
    metrics = {
        "Model": model_name,
        "R2": round(float(r2_score(y_test, best_preds)), 3),
        "MAE": round(float(mean_absolute_error(y_test, best_preds)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, best_preds))), 3)
    }
    
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return best_model, metrics, fi_df, feature_names

def predict_crop_yield(model: Any, feature_names: list, input_data: Dict[str, Any]) -> float:
    input_df = pd.DataFrame([input_data])
    encoded_df = pd.get_dummies(input_df, columns=["Crop_Type", "Region"], drop_first=False)
    aligned_df = encoded_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(aligned_df)[0]
    return float(round(prediction, 2))