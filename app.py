# app.py

import streamlit as st
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("best_svm_model_8features.pkl")
scaler = joblib.load("scaler_8features.pkl")

st.title("🩺 Diabetes Prediction App")

st.write("Please enter the following details:")

# Input form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Collect inputs into a DataFrame
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Show result
    if prediction[0] == 1:
        st.success("✅ The person is likely **Diabetic**.")
    else:
        st.success("✅ The person is likely **NOT Diabetic**.")
