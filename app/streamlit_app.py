import streamlit as st
import pandas as pd
from app.model.predictor import predict_single, predict_batch
from app.utils import load_params

# Load configuration
params = load_params()
BATCH_OUTPUT_PATH = params["BATCH_OUTPUT_PATH"]

# Set up page configuration
st.set_page_config(page_title="🍷 Wine Class Predictor", layout="centered")

# Apply custom CSS for aesthetics
st.markdown("""
    <style>
        .main {
            background-color: #f8f4f0;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #7b2c25;
            font-size: 2.8em;
        }
        .stButton button {
            background-color: #7b2c25;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 2em;
        }
        .stDownloadButton button {
            background-color: #1d6f42;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stNumberInput label {
            font-weight: bold;
        }
        .stSidebar {
            background-color: #f3e6e3;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.title("🍇 Wine Class Predictor")
st.markdown("**Predict the type of wine based on its chemical properties**")

# Sidebar for mode selection
st.sidebar.header("🔧 Inference Mode")
mode = st.sidebar.radio("Choose a prediction mode:", ["🔍 Single Prediction", "📂 Batch Prediction"])

# Feature names for input
feature_names = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

# === Single Prediction Mode ===
if mode == "🔍 Single Prediction":
    st.subheader("🧪 Enter Wine Features")
    features = []

    col1, col2 = st.columns(2)
    for i, name in enumerate(feature_names):
        with (col1 if i % 2 == 0 else col2):
            value = st.number_input(f"{name}", value=0.0)
            features.append(value)

    if st.button("🔮 Predict"):
        prediction = predict_single(features)
        st.success(f"✅ **Predicted Wine Class:** `{prediction}`")

# === Batch Prediction Mode ===
else:
    st.subheader("📤 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with wine features", type="csv")

    if uploaded_file is not None:
        st.markdown("### 📊 Input Preview")
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df.head())

        result_df = predict_batch(input_df)

        st.markdown("### ✅ Prediction Results")
        st.dataframe(result_df)

        # Download results
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "batch_predictions.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("By Herman Tcheneghon ")
