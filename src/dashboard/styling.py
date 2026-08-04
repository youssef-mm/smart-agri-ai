import streamlit as st

def inject_custom_css(lang: str = "ar"):
    direction = "rtl" if lang == "ar" else "ltr"
    text_align = "right" if lang == "ar" else "left"
    
    st.markdown(f"""
        <style>
        /* RTL & Language Alignment Fixes */
        .stApp, [data-testid="stSidebar"] {{
            direction: {direction};
            text-align: {text_align};
        }}
        
        /* Metric Cards Styling */
        div[data-testid="stMetric"] {{
            background-color: #1A2026;
            border: 1px solid #2D3748;
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        div[data-testid="stMetricValue"] {{
            font-size: 1.7rem !important;
            font-weight: 700;
            color: #4CAF50 !important;
        }}

        /* Clean Sidebar & Buttons */
        .stDownloadButton button {{
            width: 100%;
            background-color: #2E7D32 !important;
            color: white !important;
            border-radius: 8px;
            border: none;
        }}
        
        .stDownloadButton button:hover {{
            background-color: #388E3C !important;
        }}
        </style>
    """, unsafe_allow_html=True)