import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

DARK_PAPER_BG = "rgba(0,0,0,0)"
DARK_PLOT_BG = "#1A2026"

def get_sampled_df(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """خافض حمولة عالي الكفاءة لضمان فتح المخططات بسرعة"""
    if len(df) > max_points:
        return df.sample(n=max_points, random_state=42)
    return df

def build_smoothed_timeline_chart(df: pd.DataFrame, time_freq: str = "W", title: str = "Yield Trend Over Time") -> go.Figure:
    """رسم بياني زمني ناعم مع تجميع أسبوعي لمنع تداخل البيانات"""
    if df.empty:
        return go.Figure()
        
    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    
    # تجميع البيانات أسبوعياً
    grouped = df_plot.groupby([pd.Grouper(key="Date", freq=time_freq), "Crop_Type"])["Yield_Tons_ha"].mean().reset_index()
    
    fig = px.line(
        grouped, 
        x="Date", 
        y="Yield_Tons_ha", 
        color="Crop_Type", 
        title=title, 
        template="plotly_dark",
        render_mode="svg"
    )
    # تنعيم الخطوط وزيادة سمكها
    fig.update_traces(line=dict(width=2.5, shape='spline'))
    fig.update_layout(
        height=330, 
        paper_bgcolor=DARK_PAPER_BG, 
        plot_bgcolor=DARK_PLOT_BG, 
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def build_scatter_chart(df: pd.DataFrame, title: str = "Rainfall vs Yield"):
    df_plot = get_sampled_df(df, max_points=1500)
    fig = px.scatter(
        df_plot, 
        x="Rainfall_mm", 
        y="Yield_Tons_ha", 
        color="Crop_Type", 
        title=title, 
        template="plotly_dark",
        render_mode="svg"
    )
    fig.update_layout(height=320, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_boxplot_chart(df: pd.DataFrame, title: str = "Yield Variabilities"):
    df_plot = get_sampled_df(df, max_points=3000)
    fig = px.box(
        df_plot, 
        x="Crop_Type", 
        y="Yield_Tons_ha", 
        color="Crop_Type", 
        points=False, 
        title=title, 
        template="plotly_dark"
    )
    fig.update_layout(height=320, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_heatmap_chart(df: pd.DataFrame, title: str = "Correlation Matrix"):
    df_plot = get_sampled_df(df, max_points=5000)
    num_cols = ["Rainfall_mm", "Temperature_C", "Humidity_pct", "Soil_pH", "NPK_Score", "Yield_Tons_ha"]
    corr = df_plot[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title=title, template="plotly_dark")
    fig.update_layout(height=320, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_feature_importance_chart(fi_df: pd.DataFrame, title: str = "Top Yield Drivers"):
    fig = px.bar(fi_df.head(8), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Greens", title=title, template="plotly_dark")
    fig.update_layout(height=300, paper_bgcolor=DARK_PAPER_BG, plot_bgcolor=DARK_PLOT_BG, yaxis=dict(autorange="reversed"))
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
    fig.update_layout(height=260, paper_bgcolor=DARK_PAPER_BG, font=dict(color="#E0E0E0"), margin=dict(l=20, r=20, t=40, b=20))
    return fig

def build_distribution_chart(df: pd.DataFrame, title: str = "Crop Yield Distribution"):
    df_plot = get_sampled_df(df, max_points=3000)
    fig = px.histogram(
        df_plot, 
        x="Yield_Tons_ha", 
        nbins=25, 
        title=title, 
        template="plotly_dark",
        color_discrete_sequence=["#10B981"]
    )
    fig.update_layout(
        height=320, 
        paper_bgcolor=DARK_PAPER_BG, 
        plot_bgcolor=DARK_PLOT_BG, 
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig