# app.py

import streamlit as st
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("best_svm_model_8features.pkl")
scaler = joblib.load("scaler_8features.pkl")

st.title("🩺 Diabetes Prediction App")

st.write("Please enter the following details:")

# Gender input to determine if Pregnancies should be shown
gender = st.selectbox("Gender", ["Female", "Male"])

# Input form
if gender == "Female":
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
else:
    pregnancies = 0  # Hidden and defaulted for males

glucose = st.number_input("Glucose", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Predict button
if st.button("Predict"):
    # Check if all fields are zero
    if (
        pregnancies == 0 and glucose == 0 and blood_pressure == 0 and
        skin_thickness == 0 and insulin == 0 and bmi == 0.0 and
        dpf == 0.0 and age == 0
    ):
        st.warning("⚠️ All fields are required for prediction. Please enter valid values.")
    else:
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
