import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

DARK_PAPER_BG = "rgba(0,0,0,0)"
DARK_PLOT_BG = "#1A2026"

def build_timeline_chart(df: pd.DataFrame, title: str = "Yield Trend Over Time"):
    if df.empty:
        return go.Figure()
        
    # تجميع البيانات أسبوعياً لتقليل الضوضاء ومنع تداخل الخطوط
    df_sorted = df.sort_values("Date")
    grouped = df_sorted.groupby([pd.Grouper(key="Date", freq="W"), "Crop_Type"])["Yield_Tons_ha"].mean().reset_index()
    
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

def build_feature_importance_chart(fi_df: pd.DataFrame, title: str = "Top Yield Drivers"):
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