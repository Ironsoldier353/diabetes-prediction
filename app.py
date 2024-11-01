# diabetes_app.py

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")  # Load your logistic regression model
    scaler = joblib.load("scaler.pkl")          # Load the scaler
    return model, scaler

model, scaler = load_model()

# Load dataset to get mean values for each feature to handle missing values
data = pd.read_csv("diabetes.csv")
feature_means = data.drop(columns='Outcome').mean()

# Streamlit app layout
st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of diabetes based on patient data.")

# Input fields for user data, leaving them blank by default
pregnancies = st.text_input("Pregnancies", "")
glucose = st.text_input("Glucose Level", "")
blood_pressure = st.text_input("Blood Pressure", "")
skin_thickness = st.text_input("Skin Thickness", "")
insulin = st.text_input("Insulin", "")
bmi = st.text_input("BMI", "")
diabetes_pedigree = st.text_input("Diabetes Pedigree Function", "")
age = st.text_input("Age", "")

# Prepare data for prediction, using mean values for any empty input
user_data = np.array([
    int(pregnancies) if pregnancies else feature_means['Pregnancies'],
    float(glucose) if glucose else feature_means['Glucose'],
    float(blood_pressure) if blood_pressure else feature_means['BloodPressure'],
    float(skin_thickness) if skin_thickness else feature_means['SkinThickness'],
    float(insulin) if insulin else feature_means['Insulin'],
    float(bmi) if bmi else feature_means['BMI'],
    float(diabetes_pedigree) if diabetes_pedigree else feature_means['DiabetesPedigreeFunction'],
    int(age) if age else feature_means['Age']
]).reshape(1, -1)

# Scale the input data
user_data_scaled = scaler.transform(user_data)

# Predict diabetes outcome
if st.button("Predict Diabetes"):
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)[0]

    # Display results
    if prediction[0] == 1:
        st.write(f"The model predicts you **may have diabetes** with a confidence of {prediction_proba[1]:.2f}.")
    else:
        st.write(f"The model predicts you **do not have diabetes** with a confidence of {prediction_proba[0]:.2f}.")
