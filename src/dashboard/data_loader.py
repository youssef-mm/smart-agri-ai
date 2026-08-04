import pandas as pd
from generator import generate_agri_data

# ⚙️ Toggle Flag: غير هذه القيمة لـ True لما التيم يخلص أجزاءه
USE_LIVE_PIPELINE = False  

def load_dashboard_data() -> pd.DataFrame:
    if USE_LIVE_PIPELINE:
        # 🟢 هنا هنكتب كود القراءة من Spark/Database لما التيم يخلص
        # example: return pd.read_sql("SELECT * FROM agri_analytics", db_engine)
        pass
    else:
        # 🟡 وضع المحاكاة المؤقت حتى انتهاء التيم
        return generate_agri_data(n_samples=400)