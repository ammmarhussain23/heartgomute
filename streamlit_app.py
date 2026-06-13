import streamlit as st
from predict import HeartDiseasePredictor

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.title("Heart Disease Risk Predictor")
st.markdown("Enter patient clinical features to predict heart disease probability.")

predictor = HeartDiseasePredictor("models")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=54)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Typical Angina", 2: "2 - Atypical Angina", 3: "3 - Non-Anginal Pain", 4: "4 - Asymptomatic"}[x],
    )
    bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=300, value=130)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ekg = st.selectbox(
        "EKG Results",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 - Normal", 1: "1 - ST-T Abnormality", 2: "2 - LV Hypertrophy"}[x],
    )

with col2:
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=700, value=245)
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox(
        "Slope of ST",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}[x],
    )
    vessels = st.selectbox("Number of Vessels (Fluoroscopy)", options=[0, 1, 2, 3])
    thallium = st.selectbox(
        "Thallium Stress Test",
        options=[3, 6, 7],
        format_func=lambda x: {3: "3 - Normal", 6: "6 - Fixed Defect", 7: "7 - Reversible Defect"}[x],
    )

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    result = predictor.predict({
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thallium,
    })

    prob = result["probability"]
    risk = result["risk_level"]

    if risk == "low":
        st.success(f"**Probability: {prob:.1%}** — Low Risk")
    elif risk == "moderate":
        st.warning(f"**Probability: {prob:.1%}** — Moderate Risk")
    else:
        st.error(f"**Probability: {prob:.1%}** — High Risk")

    st.progress(prob)

    st.caption("This tool is for educational purposes only and is not a substitute for professional medical advice.")
