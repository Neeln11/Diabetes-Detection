import streamlit as st
import numpy as np
import joblib

# Load the trained model & scaler
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("### Enter the patient details below:")

# Collect user input
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=30)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Prediction button
if st.button("🔍 Predict Diabetes"):
    # Convert input data into array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Get prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Display the result
    if prediction == 1:
        st.error("⚠️ The model predicts that the patient **has diabetes**.")
    else:
        st.success("✅ The model predicts that the patient **does not have diabetes**.")

# Footer
st.markdown("---")
st.markdown("🛠 **Built with Streamlit & Machine Learning**")
