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

warnings.filterwarnings('ignore')

# --- Global Constants ---
NUM_COLS = ['age_years', 'BMI', 'ap_hi', 'ap_lo']
CAT_COLS = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
ML_FEATURES = ['age_years', 'gender', 'BMI', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']


# --- Asset Loading ---
@st.cache_resource
def load_production_assets():
    try:
        scaler = joblib.load('scaler.joblib')
        ensemble_model = joblib.load('final_ensemble_model.joblib')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dl_model = load_model('final_dl_model.h5')

        kmeans = joblib.load('kmeans.joblib')
        return scaler, ensemble_model, dl_model, kmeans

    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Ensure all assets exist.")
        st.stop()

    except Exception as e:
        st.error("Failed to load a model. Did you re-save the DL model in SavedModel format?")
        st.error(f"Details: {e}")
        st.stop()


scaler, ensemble_model, dl_model, kmeans = load_production_assets()


# --- Preprocessing ---
def preprocess_input(input_df):
    input_num = input_df[NUM_COLS]
    scaled_data = scaler.transform(input_num)
    scaled_df = pd.DataFrame(scaled_data, columns=NUM_COLS)

    input_cat = input_df[CAT_COLS].reset_index(drop=True)

    processed_df = pd.concat([scaled_df, input_cat], axis=1)
    processed_df = processed_df[ML_FEATURES]

    return processed_df


# --- SHAP Explainer Initialization ---
def initialize_explainer(model_choice, ensemble_model, dl_model, preprocess_func):
    background_df = pd.DataFrame(np.zeros((1, 10)), columns=ML_FEATURES)
    background_scaled = preprocess_func(background_df)

    try:
        if model_choice == "Ensemble Model (XGB+LR)":
            return shap.Explainer(ensemble_model.predict_proba, background_scaled)
        else:
            return shap.Explainer(dl_model.predict, background_scaled)

    except Exception as e:
        st.warning(f"SHAP initialization failed: {e}")
        return None


# --- Cluster Profiles ---
def get_cluster_profile(cluster_label):
    profiles = {
        0: "**Profile 0: Lower Risk / Younger Profile.** Low BMI, healthier BP, good metabolic levels.",
        1: "**Profile 1: Higher Risk / Established Risk Profile.** Higher BMI, BP, cholesterol.",
        2: "**Profile 2: Mid-Age, Low Activity.** Moderate risk, low physical activity.",
        3: "**Profile 3: Extremely High BP Risk.** Very elevated systolic/diastolic pressure.",
        4: "**Profile 4: Mixed Lifestyle Risks.** High smoking/alcohol rates.",
        5: "**Profile 5: Young, Pre-Diabetic.** Higher glucose/cholesterol early in life."
    }
    return profiles.get(cluster_label, f"Cluster profile unavailable for label {cluster_label}.")


# --- Batch Download Helper ---
def get_table_download_link(df, filename="batch_predictions.csv", text="Download Batch Predictions"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'


# --- Streamlit UI ---
st.set_page_config(page_title="Cardiovascular Risk Assessment Platform", layout="wide")

st.title("🫀 Patient Cardiovascular Risk Assessment")
st.markdown("---")

st.sidebar.header("Application Mode")
analysis_mode = st.sidebar.radio("Select Prediction Mode:", ["Single Patient Assessment", "Batch Processing (Upload CSV)"])

st.sidebar.markdown("---")
st.sidebar.header("Prediction Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Engine:",
    ("Ensemble Model (XGB+LR)", "Deep Learning Model (DNN)")
)

predictor = ensemble_model if model_choice == "Ensemble Model (XGB+LR)" else dl_model
explainer = initialize_explainer(model_choice, ensemble_model, dl_model, preprocess_input)


# =======================
# === SINGLE PATIENT ====
# =======================
if analysis_mode == "Single Patient Assessment":

    st.header(f"Real-Time Risk Assessment using: {model_choice}")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.number_input("Age (in years)", 18, 100, 55)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female (1)" if x == 1 else "Male (2)")
        bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 25.0, step=0.1)
        ap_hi = st.number_input("Systolic BP (ap_hi)", -150, 17000, 1000)

    with col2:
        ap_lo = st.number_input("Diastolic BP (ap_lo)", -70, 12000, 1000)
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {1:'Normal',2:'Above',3:'Well Above'}[x])
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {1:'Normal',2:'Above',3:'Well Above'}[x])
        smoke = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        alco = st.selectbox("Alcohol Consumer?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physically Active?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    if st.button("Calculate Risk & Insight"):

        data = [[age_years, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
        input_df = pd.DataFrame(data, columns=ML_FEATURES)
        scaled_input = preprocess_input(input_df)

        if model_choice == "Ensemble Model (XGB+LR)":
            proba = predictor.predict_proba(scaled_input)[:, 1][0]
        else:
            proba = predictor.predict(scaled_input).flatten()[0]

        risk_percentage = proba * 100

        st.subheader("Cardiovascular Disease Risk Prediction")

        risk_color = "red" if risk_percentage >= 50 else ("orange" if risk_percentage >= 30 else "green")
        st.markdown(
            f"#### Predicted Risk: <span style='color:{risk_color};'>**{risk_percentage:.2f}%**</span>",
            unsafe_allow_html=True,
        )

        std_dev = 0.05
        ci_low = max(0, proba - 2 * std_dev) * 100
        ci_high = min(1, proba + 2 * std_dev) * 100

        st.markdown(f"Confidence Interval (95% CI): **{ci_low:.2f}% to {ci_high:.2f}%**")

        st.markdown("---")
        st.subheader("Interactive Cluster Assignment")

        cluster_label = kmeans.predict(scaled_input)[0]
        st.markdown(f"##### Assigned Patient Profile Cluster: **Cluster {cluster_label}**")
        st.info(get_cluster_profile(cluster_label))

        st.markdown("---")
        st.subheader("SHAP Force Plot Explanation")


        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            st.components.v1.html(shap_html, height=height)


        if explainer:

            X_sample = scaled_input
            shap_values = explainer(X_sample)


            base_vals = shap_values.base_values

            if np.ndim(base_vals) == 1 and base_vals.size == 1:
                base_value = float(base_vals[0])

            elif np.ndim(base_vals) == 2 and base_vals.shape[1] >= 2:
                base_value = float(base_vals[0][1])

            else:
                base_value = float(np.ravel(base_vals)[0])

            phi = shap_values.values
            if phi.ndim == 3:
                phi = phi[0, :, 0]
            elif phi.ndim == 2:
                phi = phi[0]

            if isinstance(X_sample, pd.DataFrame):
                features = X_sample.iloc[0]
            else:
                features = input_df.iloc[0]

            force_plot = shap.force_plot(
                base_value,
                phi,
                features,
                matplotlib=False,
            )

            st_shap(force_plot, height=320)

        else:
            st.warning("SHAP explainer unavailable.")



# =======================
# ====== BATCH MODE =====
# =======================
elif analysis_mode == "Batch Processing (Upload CSV)":

    st.header(f"Batch Patient Risk Assessment using: {model_choice}")
    st.markdown("Upload a CSV containing these 10 columns:")
    st.code(", ".join(ML_FEATURES))

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records.")

            if not all(f in batch_df.columns for f in ML_FEATURES):
                missing = [f for f in ML_FEATURES if f not in batch_df.columns]
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()

            scaled_batch = preprocess_input(batch_df.copy())

            if model_choice == "Ensemble Model (XGB+LR)":
                batch_proba = predictor.predict_proba(scaled_batch)[:, 1]
            else:
                batch_proba = predictor.predict(scaled_batch).flatten()

            batch_clusters = kmeans.predict(scaled_batch)

            batch_df['Risk_Prediction_Prob'] = batch_proba
            batch_df['Cluster_Assignment'] = batch_clusters

            st.subheader("Batch Prediction Results")
            st.dataframe(batch_df[['Risk_Prediction_Prob', 'Cluster_Assignment']].head(10))

            st.markdown(
                get_table_download_link(
                    batch_df,
                    filename=f"risk_assessment_{len(batch_df)}_patients.csv"
                ),
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Batch processing error: {e}")
