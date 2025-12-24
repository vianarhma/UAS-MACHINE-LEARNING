import streamlit as st

# Import halaman dari folder views
from views import dashboard, preprocessing, machine_learning, algoritma, about, contact

# === Konfigurasi Page ===
st.set_page_config(
    page_title="Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding-top: 2rem;
        background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Feature Card */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .feature-card h4 {
        color: #667eea;
        margin-bottom: 15px;
        font-size: 18px;
    }
    
    /* Step Card */
    .step-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 20px;
        border: 1px solid #e0e0e0;
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        flex-shrink: 0;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .step-content {
        flex: 1;
    }
    
    /* Footer Section */
    .footer-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-top: 40px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .footer-section h3 {
        margin: 0;
        font-size: 24px;
    }
    
    .footer-section p {
        margin: 10px 0;
    }
    
    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stat-card h3 {
        margin: 0;
        font-size: 36px;
        font-weight: bold;
    }
    
    /* Label Cards */
    .label-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #667eea;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .label-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .label-card h4 {
        color: #667eea;
        margin: 0 0 10px 0;
        font-weight: 600;
    }
    
    .label-card .cluster-num {
        font-weight: bold;
        font-size: 24px;
        margin: 5px 0;
        color: #333;
    }
    
    /* Prediction Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        margin: 20px 0;
    }
    
    .prediction-card h2 {
        margin: 0;
        font-size: 48px;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-card p {
        margin: 10px 0 0 0;
        font-size: 20px;
        opacity: 0.95;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: white !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
    }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border: none !important;
    }
    
    /* Success/Error/Warning/Info Box */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# === Navigation Sidebar ===
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center;'>🧭 Navigasi</h2>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🏠 Dashboard", "🔧 Preprocessing", "🤖 Machine Learning", "ℹ️ About", "📖 Tahapan Algoritma", "📧 Contact Me"],
        label_visibility="collapsed"
    )
    st.markdown("---")

# === Routing (Pengarah Halaman) ===
if page == "🏠 Dashboard":
    dashboard.show()
elif page == "🔧 Preprocessing":
    preprocessing.show()
elif page == "🤖 Machine Learning":
    machine_learning.show()
elif page == "📖 Tahapan Algoritma":
    algoritma.show()
elif page == "ℹ️ About":
    about.show()
elif page == "📧 Contact Me":
    contact.show()