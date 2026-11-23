
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="centered"
)

# ==========================================
# 2. DARK THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

div[data-testid="stForm"] {
    background-color: #142238;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #30445f;
}

.stButton button,
.stFormSubmitButton button {
    background: #00bfa6;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    height: 48px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #009e8b;
    color: white;
}

div[data-testid="stMetric"] {
    background: #142238;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_models():

    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    return model, scaler, columns


try:
    model, scaler, columns = load_models()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# 4. HEADER
# ==========================================

st.title("❤️ CARDIOPREDICT AI")

st.subheader("Heart Disease Prediction System")

st.caption(
    "Machine Learning Based Heart Health Analysis"
)

st.divider()


# ==========================================
# 5. PATIENT INFORMATION FORM
# ==========================================

st.header("👤 Patient Information")

with st.form("patient_form"):

    st.subheader("Basic Health Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            1, 120, 45
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            50, 250, 120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            50, 700, 200
        )

    with col2:

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            50, 250, 150
        )

        oldpeak = st.number_input(
            "Oldpeak",
            0.0, 10.0, 1.0, step=0.1
        )

    st.divider()

    st.subheader("Patient Demographics")

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    st.divider()

    st.subheader("Chest Pain Information")

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"],
        format_func=lambda x: {
            "ASY": "Asymptomatic",
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "TA": "Typical Angina"
        }[x]
    )

    st.divider()

    st.subheader("ECG Information")

    col1, col2 = st.columns(2)

    with col1:

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

    with col2:

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

    st.divider()

    st.subheader("Exercise Information")

    st_slope = st.selectbox(
        "ST Slope",
        ["Down", "Flat", "Up"]
    )

    st.divider()

    predict = st.form_submit_button(
        "❤️ PREDICT HEART DISEASE",
        use_container_width=True
    )


# ==========================================
# 6. PREDICTION
# ==========================================

if predict:

    try:

        patient_data = {

            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,

            "Sex_M": int(sex == "Male"),

            "ChestPainType_ATA": int(chest_pain == "ATA"),
            "ChestPainType_NAP": int(chest_pain == "NAP"),
            "ChestPainType_TA": int(chest_pain == "TA"),

            "RestingECG_Normal": int(resting_ecg == "Normal"),
            "RestingECG_ST": int(resting_ecg == "ST"),

            "ExerciseAngina_Y": int(exercise_angina == "Yes"),

            "ST_Slope_Flat": int(st_slope == "Flat"),
            "ST_Slope_Up": int(st_slope == "Up")
        }

        # Create DataFrame in exact model column order

        input_df = pd.DataFrame(
            [[patient_data[col] for col in columns]],
            columns=columns
        )

        # Apply saved scaler

        scaled_data = scaler.transform(input_df)

        # Predict

        prediction = model.predict(scaled_data)[0]

        st.divider()

        st.header("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Heart Disease Detected "
                "according to the KNN model."
            )

        elif prediction == 0:

            st.success(
                "✅ No Heart Disease Detected "
                "according to the KNN model."
            )

        else:

            st.warning(f"Unexpected prediction: {prediction}")

        # Model probabilities

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(scaled_data)[0]

            st.subheader("Model Prediction Probabilities")

            for label, probability in zip(
                model.classes_, probabilities
            ):

                st.write(
                    f"Class {label}: {probability * 100:.2f}%"
                )

                st.progress(float(probability))

        st.info(
            "Medical Disclaimer: This application is an "
            "educational machine learning project, not a "
            "medical diagnostic tool. Consult a qualified "
            "healthcare professional for medical assessment."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "❤️ CardioPredict AI | Python | Scikit-learn | Streamlit"
)
