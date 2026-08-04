import sys
from pathlib import Path
import datetime
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import folium
from streamlit_folium import st_folium

# ==============================================================================
# 1. PAGE CONFIGURATION & INITIAL SETUP
# ==============================================================================
st.set_page_config(
    page_title="Smart Agri AI — Executive Operations Dashboard",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. LOCALIZATION DICTIONARY & STYLING ENGINE (CSS / RTL)
# ==============================================================================
LOCALIZATION = {
    "ar": {
        "app_title": "🌾 Smart Agri AI — منصة التحليلات والإدارة الزراعية",
        "app_subtitle": "نظام دعم اتخاذ القرار بالذكاء الاصطناعي لمتابعة العمليات والتنبؤ بإنتاجية المحاصيل.",
        "nav_overview": "المراقبة العامة",
        "nav_analytics": "التحليلات والمؤشرات",
        "nav_prediction": "محرك التنبؤ (AI)",
        "nav_pipeline": "مراقبة البث المباشر",
        "export_btn": "📥 تصدير البيانات (CSV)",
        "kpi_yield": "متوسط الإنتاجية",
        "kpi_temp": "متوسط الحرارة",
        "kpi_npk": "مؤشر التربة (NPK)",
        "kpi_logs": "إجمالي القراءات",
        "crop_select": "اختر المحاصيل:",
        "region_select": "اختر المناطق:",
        "date_select": "النطاق الزمني:",
        "temp_select": "درجة الحرارة (°C):",
        "aggregation_select": "تجميع اتجاه الإنتاجية:",
        "agg_daily": "يومي (Daily)",
        "agg_weekly": "أسبوعي (Weekly)",
        "agg_monthly": "شهري (Monthly)",
        "weather_title": "🌐 الطقس المباشر للمحافظات",
        "map_title": "🗺️ خريطة توزيع المزارع الإقليمية",
        "model_eval": "📊 كفاءة نموذج التنبؤ (ML Evaluation)",
        "best_model": "النموذج المعتمد",
        "predict_btn": "🚀 حساب التنبؤ بالإنتاجية",
        "pred_result": "الإنتاجية المتوقعة",
        "crop_type": "نوع المحصول",
        "region": "المنطقة",
        "rainfall": "الأمطار (mm)",
        "npk": "مغذيات التربة (NPK)",
        "ph": "حموضة التربة (pH)",
        "humidity": "الرطوبة (%)",
        "temp": "الحرارة (°C)",
        "data_preview": "📋 سجلات البيانات المباشرة",
        "no_data": "⚠️ لا توجد بيانات مطابقة للفلاتر المحددة."
    },
    "en": {
        "app_title": "🌾 Smart Agri AI — Executive Operations Dashboard",
        "app_subtitle": "Real-time AI decision support system & crop yield forecasting platform.",
        "nav_overview": "Overview",
        "nav_analytics": "Data Analytics",
        "nav_prediction": "AI Prediction Engine",
        "nav_pipeline": "Pipeline Monitor",
        "export_btn": "📥 Export Data (CSV)",
        "kpi_yield": "Avg Yield",
        "kpi_temp": "Avg Temp",
        "kpi_npk": "Soil NPK Score",
        "kpi_logs": "Total Logs",
        "crop_select": "Select Crops:",
        "region_select": "Select Regions:",
        "date_select": "Date Range Filter:",
        "temp_select": "Temperature Range (°C):",
        "aggregation_select": "Yield Trend Aggregation:",
        "agg_daily": "Daily",
        "agg_weekly": "Weekly",
        "agg_monthly": "Monthly",
        "weather_title": "🌐 Live Provincial Weather",
        "map_title": "🗺️ Regional Farm Distribution Map",
        "model_eval": "📊 AI Model Performance Metrics",
        "best_model": "Selected Model",
        "predict_btn": "🚀 Run AI Yield Predictor",
        "pred_result": "Predicted Yield",
        "crop_type": "Crop Type",
        "region": "Region",
        "rainfall": "Rainfall (mm)",
        "npk": "Soil NPK",
        "ph": "Soil pH",
        "humidity": "Humidity (%)",
        "temp": "Temperature (°C)",
        "data_preview": "📋 Filtered Live Logs",
        "no_data": "⚠️ No records matching the selected filter criteria."
    }
}

def inject_enterprise_css(lang_code: str):
    """Dynamically injects custom CSS to support RTL/LTR layout and enterprise dark aesthetics."""
    direction = "rtl" if lang_code == "ar" else "ltr"
    text_align = "right" if lang_code == "ar" else "left"
    
    st.markdown(f"""
        <style>
        /* Enterprise Directionality & Typography */
        .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {{
            direction: {direction};
            text-align: {text_align};
        }}
        
        /* High-Performance Card Containers */
        div[data-testid="stMetric"] {{
            background-color: #1A2026;
            border: 1px solid #2D3748;
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: #4CAF50;
            transform: translateY(-2px);
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.75rem !important;
            font-weight: 700;
            color: #4CAF50 !important;
        }}
        
        /* Sidebar Navigation & Controls Customization */
        [data-testid="stSidebar"] {{
            background-color: #12161A;
            border-right: 1px solid #2D3748;
        }}
        
        /* Custom Download Button Bar */
        .stDownloadButton button {{
            width: 100%;
            background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.55rem 1rem !important;
        }}
        .stDownloadButton button:hover {{
            background: linear-gradient(135deg, #388E3C 0%, #2E7D32 100%) !important;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }}
        
        /* Table Custom Fixes */
        [data-testid="stDataFrame"] {{
            background-color: #1A2026;
            border-radius: 8px;
        }}
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. CACHED DATA PIPELINE & API SERVICES
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_or_generate_agri_data(n_samples: int = 400) -> pd.DataFrame:
    """Generates clean operational data with strict bounds and consistent schema."""
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
    """Fetches real-time weather data with robust timeout fallback."""
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
                "Source": "OpenWeatherMap API"
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

@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(df: pd.DataFrame):
    """Trains regression models and returns evaluation metrics and feature importances."""
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
        best_model, best_preds, model_name = xgb, xgb_preds, "XGBoost Regressor"
    else:
        best_model, best_preds, model_name = rf, rf_preds, "RandomForest Regressor"
        
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

# ==============================================================================
# 4. PLOTLY VISUALS BUILDERS (DARK THEME & SMOOTHING)
# ==============================================================================
DARK_PAPER_BG = "rgba(0,0,0,0)"
DARK_PLOT_BG = "#1A2026"

def build_smoothed_timeline_chart(df: pd.DataFrame, time_freq: str = "W", title: str = "") -> go.Figure:
    if df.empty:
        return go.Figure()
        
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "يومي": "D", "أسبوعي": "W", "شهري": "M"}
    selected_freq = freq_map.get(time_freq, "W")
    
    df_sorted = df.sort_values("Date")
    grouped = df_sorted.groupby([pd.Grouper(key="Date", freq=selected_freq), "Crop_Type"])["Yield_Tons_ha"].mean().reset_index()
    
    fig = px.line(
        grouped, 
        x="Date", 
        y="Yield_Tons_ha", 
        color="Crop_Type", 
        title=title, 
        template="plotly_dark",
        markers=True
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=5))
    fig.update_layout(
        height=380, 
        paper_bgcolor=DARK_PAPER_BG, 
        plot_bgcolor=DARK_PLOT_BG, 
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def build_scatter_chart(df: pd.DataFrame, title: str = "Rainfall vs Yield"):
    fig = px.scatter(df, x="Rainfall_mm", y="Yield_Tons_ha", size="NPK_Score", color="Crop_Type", title=title, template="plotly_dark")
    fig.update_layout(height=380, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_boxplot_chart(df: pd.DataFrame, title: str = "Yield Variabilities"):
    fig = px.box(df, x="Crop_Type", y="Yield_Tons_ha", color="Crop_Type", points="all", title=title, template="plotly_dark")
    fig.update_layout(height=380, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_heatmap_chart(df: pd.DataFrame, title: str = "Correlation Matrix"):
    num_cols = ["Rainfall_mm", "Temperature_C", "Humidity_pct", "Soil_pH", "NPK_Score", "Yield_Tons_ha"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title=title, template="plotly_dark")
    fig.update_layout(height=380, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_feature_importance_chart(fi_df: pd.DataFrame, title: str = "Top Drivers of Yield"):
    fig = px.bar(fi_df.head(8), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Greens", title=title, template="plotly_dark")
    fig.update_layout(height=320, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, yaxis=dict(autorange="reversed"))
    return fig

def build_gauge_chart(yield_value: float, title: str = "Predicted Yield"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=yield_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': '#E0E0E0'}},
        gauge={
            'axis': {'range': [0, 10], 'tickcolor': '#E0E0E0'},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 3], 'color': "#3e2723"},
                {'range': [3, 6], 'color': "#33691e"},
                {'range': [6, 10], 'color': "#1b5e20"}
            ],
        }
    ))
    fig.update_layout(height=280, paper_bgcolor=DARK_PAPER_BG, font=dict(color="#E0E0E0"), margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ==============================================================================
# 5. APPLICATION CONTROLLER & LAYOUT RENDERER
# ==============================================================================
def main():
    # Load Session Data
    if "df" not in st.session_state:
        st.session_state.df = load_or_generate_agri_data(n_samples=400)

    # Sidebar Controls & Language Switcher
    with st.sidebar:
        st.markdown("## 🌾 **Agri Intelligence**")
        
        lang_choice = st.radio("🌐 Language / اللغة", options=["العربية", "English"], horizontal=True)
        lang_code = "ar" if lang_choice == "العربية" else "en"
        t = LOCALIZATION[lang_code]
        
        st.markdown("---")
        
        # Navigation Options using modern controls
        nav_items = [t["nav_overview"], t["nav_analytics"], t["nav_prediction"], t["nav_pipeline"]]
        
        if hasattr(st, "segmented_control"):
            selected_nav = st.segmented_control(label="Navigation", options=nav_items, default=nav_items[0])
        else:
            selected_nav = st.radio(label="Navigation", options=nav_items)
            
        st.markdown("---")
        
        # Data Filtering Widgets
        all_crops = list(st.session_state.df["Crop_Type"].unique())
        selected_crops = st.multiselect(t["crop_select"], options=all_crops, default=all_crops)
        
        all_regions = list(st.session_state.df["Region"].unique())
        selected_regions = st.multiselect(t["region_select"], options=all_regions, default=all_regions)
        
        min_date = st.session_state.df["Date"].min().date()
        max_date = st.session_state.df["Date"].max().date()
        date_range = st.date_input(t["date_select"], value=(min_date, max_date), min_value=min_date, max_value=max_date)
        
        temp_range = st.slider(t["temp_select"], float(st.session_state.df["Temperature_C"].min()), float(st.session_state.df["Temperature_C"].max()), (18.0, 42.0))

    # Apply CSS Injector based on chosen language
    inject_enterprise_css(lang_code=lang_code)

    # Filter Application
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start_dt, end_dt = pd.Timestamp(min_date), pd.Timestamp(max_date)

    filtered_df = st.session_state.df[
        (st.session_state.df["Crop_Type"].isin(selected_crops)) &
        (st.session_state.df["Region"].isin(selected_regions)) &
        (st.session_state.df["Date"].between(start_dt, end_dt)) &
        (st.session_state.df["Temperature_C"].between(temp_range[0], temp_range[1]))
    ]

    # Model Training
    dataset_hash = hash(len(st.session_state.df))
    model, metrics, fi_df, feature_names = train_and_evaluate_models(st.session_state.df)

    # Header & Title Bar
    st.title(t["app_title"])
    st.caption(t["app_subtitle"])

    # Executive KPI Metric Bar
    def calculate_deltas(filtered: pd.DataFrame, full: pd.DataFrame):
        if len(filtered) == 0:
            return 0.0, 0.0, 0.0
        yield_diff = filtered["Yield_Tons_ha"].mean() - full["Yield_Tons_ha"].mean()
        temp_diff = filtered["Temperature_C"].mean() - full["Temperature_C"].mean()
        npk_diff = filtered["NPK_Score"].mean() - full["NPK_Score"].mean()
        return round(yield_diff, 2), round(temp_diff, 1), round(npk_diff, 1)

    d_yield, d_temp, d_npk = calculate_deltas(filtered_df, st.session_state.df)

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    avg_y = filtered_df["Yield_Tons_ha"].mean() if not filtered_df.empty else 0
    avg_t = filtered_df["Temperature_C"].mean() if not filtered_df.empty else 0
    avg_npk = filtered_df["NPK_Score"].mean() if not filtered_df.empty else 0

    col_k1.metric(t["kpi_yield"], f"{avg_y:.2f} T/ha", delta=f"{d_yield:+.2f} T/ha vs avg")
    col_k2.metric(t["kpi_temp"], f"{avg_t:.1f} °C", delta=f"{d_temp:+.1f} °C vs avg")
    col_k3.metric(t["kpi_npk"], f"{avg_npk:.0f} / 100", delta=f"{d_npk:+.1f} pts")
    col_k4.metric(t["kpi_logs"], f"{len(filtered_df)}", delta="Filtered Logs")

    st.markdown("---")

    # Tab/Navigation Routing
    if selected_nav in [t["nav_overview"], "Overview", None]:
        c1, c2 = st.columns([2, 1])
        with c1:
            # Aggregation Granularity Selector
            agg_choice = st.radio(t["aggregation_select"], options=[t["agg_daily"], t["agg_weekly"], t["agg_monthly"]], horizontal=True)
            st.plotly_chart(build_smoothed_timeline_chart(filtered_df, time_freq=agg_choice, title=t["nav_overview"]), use_container_width=True)
            
            st.subheader(t["map_title"])
            m = folium.Map(location=[26.8206, 30.8025], zoom_start=6, tiles="CartoDB dark_matter")
            coords = {"Delta": [30.5, 31.0], "Fayoum": [29.3, 30.8], "Upper Egypt": [25.5, 32.7], "Alexandria": [31.2, 29.9]}
            for name, loc in coords.items():
                folium.Marker(loc, popup=name, icon=folium.Icon(color="green", icon="leaf")).add_to(m)
            st_folium(m, use_container_width=True, height=320)
            
        with c2:
            st.subheader(t["weather_title"])
            city = st.selectbox("City / المحافظة", ["Cairo", "Alexandria", "Fayoum", "Asyut"])
            w_info, _ = fetch_live_weather(city)
            
            # Weather Metric Cards with Icons
            w_col1, w_col2 = st.columns(2)
            w_col1.metric("🌡️ Temp", f"{w_info['Temperature_C']} °C")
            w_col2.metric("💧 Humidity", f"{w_info['Humidity_pct']} %")
            
            st.markdown("---")
            st.subheader(t["data_preview"])
            st.dataframe(filtered_df[["Date", "Crop_Type", "Region", "Yield_Tons_ha"]].tail(6), use_container_width=True)
            
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(label=t["export_btn"], data=csv_data, file_name=f"agri_data_{datetime.date.today()}.csv", mime="text/csv")

    elif selected_nav in [t["nav_analytics"], "Data Analytics"]:
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(build_scatter_chart(filtered_df), use_container_width=True)
        with a2:
            st.plotly_chart(build_boxplot_chart(filtered_df), use_container_width=True)
        st.plotly_chart(build_heatmap_chart(filtered_df), use_container_width=True)

    elif selected_nav in [t["nav_prediction"], "AI Prediction Engine"]:
        st.subheader(t["model_eval"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["best_model"], metrics["Model"])
        m2.metric("R² Score", metrics["R2"])
        m3.metric("MAE", metrics["MAE"])
        m4.metric("RMSE", metrics["RMSE"])

        with st.form("pred_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                in_crop = st.selectbox(t["crop_type"], all_crops)
                in_region = st.selectbox(t["region"], all_regions)
                in_temp = st.slider(t["temp"], 10.0, 50.0, 28.0)
            with f2:
                in_rain = st.number_input(t["rainfall"], 0, 500, 180)
                in_npk = st.slider(t["npk"], 0, 100, 75)
            with f3:
                in_ph = st.slider(t["ph"], 4.0, 9.0, 6.5)
                in_hum = st.slider(t["humidity"], 10.0, 100.0, 55.0)
                
            submit = st.form_submit_button(t["predict_btn"])
            
        if submit:
            payload = {"Crop_Type": in_crop, "Region": in_region, "Rainfall_mm": in_rain, "Temperature_C": in_temp, "Humidity_pct": in_hum, "Soil_pH": in_ph, "NPK_Score": in_npk}
            pred_val = predict_crop_yield(model, feature_names, payload)
            
            res1, res2 = st.columns([1, 2])
            with res1:
                st.metric(t["pred_result"], f"{pred_val} Tons/ha")
                st.plotly_chart(build_gauge_chart(pred_val, title=t["pred_result"]), use_container_width=True)
            with res2:
                st.plotly_chart(build_feature_importance_chart(fi_df), use_container_width=True)

    elif selected_nav in [t["nav_pipeline"], "Pipeline Monitor"]:
        st.subheader("⚡ Live Ingestion Stream Monitor")
        st.info("📌 Simulated stream state linked to Apache Kafka & Spark Analytics.")
        
        @st.fragment(run_every="5s")
        def render_stream():
            rate = int(np.random.normal(1250, 45))
            latency = int(np.random.normal(38, 4))
            
            s1, s2, s3 = st.columns(3)
            s1.metric("Kafka Topic Status", "Active 🟢", "agri_sensor_stream")
            s2.metric("Ingestion Rate", f"{rate:,} msg/sec")
            s3.metric("Spark Processing Latency", f"{latency} ms")
            
            st.code("[IoT Emulator] ──> (Kafka Topic: sensor_stream) ──> [Spark Engine] ──> [Streamlit UI]", language="text")

        render_stream()

if __name__ == "__main__":
    main()