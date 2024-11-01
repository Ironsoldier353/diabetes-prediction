import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Feature means for handling optional fields
feature_means = {
    'Pregnancies': 3.8,
    'Glucose': 120.9,
    'BloodPressure': 69.1,
    'SkinThickness': 20.5,
    'Insulin': 79.8,
    'BMI': 31.9,
    'DiabetesPedigreeFunction': 0.47,
    'Age': 33.2
}

# Streamlit app layout
st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of diabetes based on patient data.")

# Input fields for each feature
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=int(feature_means['Pregnancies']))
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=200.0, value=float(feature_means['Glucose']))
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=float(feature_means['BloodPressure']))
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=float(feature_means['SkinThickness']))
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=float(feature_means['Insulin']))
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=float(feature_means['BMI']))
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=float(feature_means['DiabetesPedigreeFunction']))
age = st.number_input("Age", min_value=0, max_value=120, value=int(feature_means['Age']))

# Organize inputs into a numpy array, replacing zero values with means if optional
user_data = np.array([
    pregnancies if pregnancies != 0 else feature_means['Pregnancies'],
    glucose if glucose != 0 else feature_means['Glucose'],
    blood_pressure if blood_pressure != 0 else feature_means['BloodPressure'],
    skin_thickness if skin_thickness != 0 else feature_means['SkinThickness'],
    insulin if insulin != 0 else feature_means['Insulin'],
    bmi if bmi != 0 else feature_means['BMI'],
    diabetes_pedigree if diabetes_pedigree != 0 else feature_means['DiabetesPedigreeFunction'],
    age if age != 0 else feature_means['Age']
]).reshape(1, -1)

# Scale user data
user_data_scaled = scaler.transform(user_data)

# Predict diabetes outcome
if st.button("Predict Diabetes"):
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0]

    # Display results with confidence
    if prediction == 1:
        st.write(f"The model predicts you **may have diabetes** with a confidence of {prediction_proba[1]:.2f}.")
    else:
        st.write(f"The model predicts you **do not have diabetes** with a confidence of {prediction_proba[0]:.2f}.")
