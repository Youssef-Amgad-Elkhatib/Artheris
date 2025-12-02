import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import base64
import io
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import load_model

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# --- 1. Define Feature Lists ---
# Must match the order used during model training
NUM_COLS = ['age_years', 'BMI', 'ap_hi', 'ap_lo']
CAT_COLS = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
ML_FEATURES = ['age_years', 'gender', 'BMI', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
# --- 2. Load Assets ---
@st.cache_resource
def load_production_assets():
    """Loads and caches all necessary models and scaler."""
    try:
        # Load Scaler (fitted only on NUM_COLS)
        scaler = joblib.load('scaler.joblib')
        # Load Ensemble Model (scikit-learn object)
        ensemble_model = joblib.load('final_ensemble_model.joblib')
        # Load DL Model (Keras/TensorFlow SavedModel directory)
        # Ensure the SavedModel format ('final_dl_model' folder) is used to avoid version errors
        dl_model = load_model('final_dl_model.h5')
        # Load K-Means Model
        kmeans = joblib.load('kmeans_model.joblib')
        return scaler, ensemble_model, dl_model, kmeans
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure all assets are correctly saved and present.")
        st.stop()
    except Exception as e:
        # Catches the Keras serialization error if the file format is wrong/corrupt
        st.error(f"Failed to load a model. **Root Cause Check:** Did you successfully re-save the DL model using `final_dl_model.save('final_dl_model')` (SavedModel format) in your notebook?")
        st.error(f"Details: {e}")
        st.stop()


# Load all assets once at app startup
scaler, ensemble_model, dl_model, kmeans = load_production_assets()

# --- 3. Helper Functions ---

def preprocess_input(input_df):
    """
    Separates numerical and categorical features, scales numerical features,
    and recombines them into a single DataFrame in the correct order (ML_FEATURES).
    """
    # 1. Select Numerical Features and Scale them
    input_num = input_df[NUM_COLS]
    scaled_data = scaler.transform(input_num)
    scaled_df = pd.DataFrame(scaled_data, columns=NUM_COLS)

    # 2. Select Categorical/Binary Features (already coded)
    input_cat = input_df[CAT_COLS].reset_index(drop=True)

    # 3. Recombine in the correct ML_FEATURES order
    # Using pd.concat for efficient merging of scaled and unscaled features
    processed_df = pd.concat([scaled_df, input_cat], axis=1)

    # Final check on column order before prediction
    processed_df = processed_df[ML_FEATURES]
    return processed_df


# Function to get cluster profile description (simplified example)
def get_cluster_profile(cluster_label):
    if cluster_label == 0:
        return (
            "**Profile 0: Lower Risk / Younger Profile.** Typically features lower BMI, healthier blood pressure, and better cholesterol/glucose levels. Focus on preventative health maintenance.")
    elif cluster_label == 1:
        return (
            "**Profile 1: Higher Risk / Established Risk Profile.** Exhibits higher average BMI, elevated blood pressure (ap_hi/ap_lo), and often higher cholesterol. Requires immediate lifestyle intervention.")
    else:
        return "Cluster profile unavailable."


# Function to download dataframe as CSV
def get_table_download_link(df, filename="batch_predictions.csv", text="Download Batch Predictions"):
    """Generates a link allowing the data in a DataFrame to be downloaded."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# --- 4. Streamlit App Layout ---

st.set_page_config(page_title="Cardiovascular Risk Assessment Platform", layout="wide")

st.title("🫀 Patient Cardiovascular Risk Assessment")
st.markdown("---")

# Navigation (Sidebar)
st.sidebar.header("Application Mode")
analysis_mode = st.sidebar.radio("Select Prediction Mode:",
                                 ["Single Patient Assessment", "Batch Processing (Upload CSV)"])

st.sidebar.markdown("---")
st.sidebar.header("Prediction Model Selection")
# Allow user to select the prediction model
model_choice = st.sidebar.selectbox(
    "Choose Prediction Engine:",
    ("Ensemble Model (XGB+LR)", "Deep Learning Model (DNN)")
)

if model_choice == "Ensemble Model (XGB+LR)":
    predictor = ensemble_model
else:
    predictor = dl_model

# SHAP explainer initialization (Using Ensemble for performance/stability)
try:
    # Use a minimal background dataset for SHAP Explainer
    explainer = shap.Explainer(ensemble_model.predict_proba,
                               pd.DataFrame(scaler.transform(pd.DataFrame(np.zeros((1, 4)), columns=NUM_COLS)),
                                            columns=NUM_COLS))
except Exception as e:
    st.warning(f"SHAP Explainer initialization warning: {e}. SHAP explanation might be unavailable.")

# --- 5. Single Patient Assessment (Input Form) ---
if analysis_mode == "Single Patient Assessment":
    st.header(f"Real-Time Risk Assessment using: {model_choice}")

    # Input columns setup
    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.number_input("Age (in years)", min_value=18, max_value=100, value=55)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female (1)" if x == 1 else "Male (2)")
        # Note: BMI is the feature name used in the saved models
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=90, max_value=200, value=120, step=1)

    with col2:
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=60, max_value=130, value=80, step=1)
        cholesterol = st.selectbox("Cholesterol", options=[1, 2, 3], format_func=lambda x:
        {1: 'Normal (1)', 2: 'Above Normal (2)', 3: 'Well Above Normal (3)'}[x])
        gluc = st.selectbox("Glucose", options=[1, 2, 3],
                            format_func=lambda x: {1: 'Normal (1)', 2: 'Above Normal (2)', 3: 'Well Above Normal (3)'}[
                                x])
        smoke = st.selectbox("Smoker?", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

    with col3:
        alco = st.selectbox("Alcohol Consumer?", options=[0, 1],
                            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        active = st.selectbox("Physically Active?", options=[0, 1],
                              format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

    if st.button("Calculate Risk & Insight"):

        # Create input DataFrame with columns matching ML_FEATURES
        data = [[age_years, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
        input_df = pd.DataFrame(data, columns=ML_FEATURES)

        # Preprocess using the corrected function
        scaled_input = preprocess_input(input_df)

        # Real-Time Risk Prediction
        if model_choice == "Ensemble Model (XGB+LR)":
            proba = predictor.predict_proba(scaled_input)[:, 1][0]
        else:  # DL Model
            proba = predictor.predict(scaled_input).flatten()[0]

        risk_percentage = proba * 100

        # --- Output: Risk Prediction ---
        st.subheader("Cardiovascular Disease Risk Prediction")

        risk_color = "red" if risk_percentage >= 50 else ("orange" if risk_percentage >= 30 else "green")
        st.markdown(f"#### Predicted Risk: <span style='color:{risk_color};'>**{risk_percentage:.2f}%**</span>",
                    unsafe_allow_html=True)

        # Simple Confidence Interval
        std_dev = 0.05
        ci_low = max(0, proba - 2 * std_dev) * 100
        ci_high = min(1, proba + 2 * std_dev) * 100
        st.markdown(f"Confidence Interval (95% CI): **{ci_low:.2f}% to {ci_high:.2f}%**")

        st.markdown("---")

        # --- Output: Cluster Assignment ---
        st.subheader("Interactive Cluster Assignment")
        cluster_label = kmeans.predict(scaled_input)[0]
        st.markdown(f"##### Assigned Patient Profile Cluster: **Cluster {cluster_label}**")
        st.info(get_cluster_profile(cluster_label))

        st.markdown("---")

        # --- Output: SHAP Explanation ---
        st.subheader("SHAP Force Plot Explanation")

        # Calculate SHAP values for the single patient instance using the Ensemble explainer
        shap_values = explainer(scaled_input.iloc[0])

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.force_plot(
            shap_values[0],
            matplotlib=True,
            show=False,
            out_names=["Predicted Risk"],
            feature_names=ML_FEATURES
        ), bbox_inches='tight')
        st.write(f"The plot shows feature contributions for the **Ensemble Model** (used for explanation).")


# --- 6. Batch Processing (File Upload) ---
elif analysis_mode == "Batch Processing (Upload CSV)":
    st.header(f"Batch Patient Risk Assessment using: {model_choice}")
    st.markdown("Upload a CSV file containing patient data. The CSV must have the following **10 columns**:")
    st.code(", ".join(ML_FEATURES))

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Successfully loaded {len(batch_df)} records.")

            # Validation check
            if not all(feature in batch_df.columns for feature in ML_FEATURES):
                missing = [f for f in ML_FEATURES if f not in batch_df.columns]
                st.error(f"Missing required columns in CSV: {', '.join(missing)}")
                st.stop()

            # Preprocess using the corrected function
            scaled_batch = preprocess_input(batch_df.copy())

            # 1. Prediction
            if model_choice == "Ensemble Model (XGB+LR)":
                batch_proba = predictor.predict_proba(scaled_batch)[:, 1]
            else:  # DL Model
                batch_proba = predictor.predict(scaled_batch).flatten()

            # 2. Clustering
            batch_clusters = kmeans.predict(scaled_batch)

            # 3. Compile Results
            batch_df['Risk_Prediction_Prob'] = batch_proba
            batch_df['Cluster_Assignment'] = batch_clusters

            st.subheader("Batch Prediction Results")
            st.dataframe(batch_df[['Risk_Prediction_Prob', 'Cluster_Assignment']].head(10))

            # Download capability
            st.markdown(get_table_download_link(batch_df, f"risk_assessment_{len(batch_df)}_patients.csv"),
                        unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during batch processing: {e}")