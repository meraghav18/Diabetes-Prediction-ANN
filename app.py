import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = load_model("diabetes_model.keras")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🩺 Diabetes Prediction using ANN")
st.write("Enter the patient's details below and click **Predict**.")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📚 Input Guide")

st.sidebar.write("""
### Feature Information

**Pregnancies**
- Number of times pregnant

**Glucose**
- Blood glucose level (mg/dL)

**Blood Pressure**
- Diastolic Blood Pressure (mmHg)

**Skin Thickness**
- Triceps skin fold thickness (mm)

**Insulin**
- 2-Hour Serum Insulin (mu U/ml)

**BMI**
- Body Mass Index

**Diabetes Pedigree Function (DPF)**
- Indicates the influence of family history on diabetes risk.
- If you don't know the value, keep it **0.5**.

**Age**
- Age in years
""")

# -------------------------------
# User Inputs
# -------------------------------
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose (mg/dL)",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure (mmHg)",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness (mm)",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin (mu U/ml)",
    min_value=0.0,
    value=79.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function (DPF)",
    min_value=0.0,
    value=0.5,
    help="If you don't know this value, keep the default value (0.5)."
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write(f"**Prediction Probability:** {probability:.2%}")

    st.progress(float(probability))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "⚠️ This application is for educational purposes only and is not a substitute for professional medical advice."
)