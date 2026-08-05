import streamlit as st

def inject_enterprise_css(lang_code: str = "ar"):
    direction = "rtl" if lang_code == "ar" else "ltr"
    text_align = "right" if lang_code == "ar" else "left"
    
    st.markdown(f"""
        <style>
        .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {{
            direction: {direction};
            text-align: {text_align};
        }}
        
        div[data-testid="stMetric"] {{
            background-color: #1A2026;
            border: 1px solid #2D3748;
            border-radius: 10px;
            padding: 12px 16px;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.7rem !important;
            font-weight: 700;
            color: #10B981 !important;
        }}
        
        /* Power BI Card Container Styling */
        div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {{
            border-radius: 12px;
        }}
        
        .stDownloadButton button {{
            width: 100%;
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            color: white !important;
            font-weight: 600;
            border-radius: 8px;
            border: none;
        }}
        </style>
    """, unsafe_allow_html=True)