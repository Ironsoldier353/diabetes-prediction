import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")  
    scaler = joblib.load("scaler.pkl")        
    return model, scaler

model, scaler = load_model()

# Load dataset to get mean values for each feature to handle missing values
data = pd.read_csv("diabetes.csv")
feature_means = data.drop(columns='Outcome').mean()

# Custom CSS to style the app
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica', sans-serif;
        background-color: #f7f7f7;
    }
    .title {
        text-align: center;
        color: #3e8e41;
    }
    .description {
        text-align: center;
        color: #333;
        font-size: 1.1em;
        margin-bottom: 40px;
    }
    .input-container {
        background-color: #fff;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .input-label {
        font-weight: bold;
    }
    .btn {
        background-color: #3e8e41;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .btn:hover {
        background-color: #2c6f34;
    }
    .result {
        font-size: 1.2em;
        font-weight: bold;
    }
    .result-positive {
        color: red;
    }
    .result-negative {
        color: green;
    }
    .required {
        color: red;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Diabetes Prediction App</h1>", unsafe_allow_html=True)

st.markdown("<p class='description'>This app predicts the likelihood of diabetes based on patient data such as age, BMI, glucose levels, and more.</p>", unsafe_allow_html=True)

#Input layout
col1, col2 = st.columns([1, 1])

# Input fields for user data, marking the necessary fields with a star
with col1:
    pregnancies = st.text_input("Pregnancies (count) *", "", placeholder="e.g., 0 - 20")
    glucose = st.text_input("Glucose Level (mg/dL) *", "", placeholder="e.g., 70 - 200 mg/dL")
    blood_pressure = st.text_input("Blood Pressure (Systolic in mmHg) *", "", placeholder="e.g., 60 - 180 mmHg")
    skin_thickness = st.text_input("Skin Thickness (mm)", "", placeholder="e.g., 10 - 99 mm")

with col2:
    insulin = st.text_input("Insulin (µU/mL)", "", placeholder="e.g., 0 - 500 µU/mL")
    bmi = st.text_input("BMI (kg/m²)", "", placeholder="e.g., 15.0 - 45.0 kg/m²")
    diabetes_pedigree = st.text_input("Diabetes Pedigree Function", "", placeholder="e.g., 0.1 - 2.5")
    age = st.text_input("Age (years) *", "", placeholder="e.g., 10 - 120 years")

# Validate input to ensure required fields are not empty
if st.button("Predict Diabetes", key="predict", help="Click to predict the likelihood of diabetes based on the entered data", use_container_width=True):
    # Check if the necessary fields are filled
    if not pregnancies or not glucose or not blood_pressure or not age:
        st.markdown("<p class='required'>Please fill all the required fields marked with a star (*).</p>", unsafe_allow_html=True)
    else:
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

        prediction = model.predict(user_data_scaled)
        prediction_proba = model.predict_proba(user_data_scaled)[0]

        # Display result 
        if prediction[0] == 1:
            st.markdown(f"<p class='result result-positive'>The model predicts you <strong>may have diabetes</strong> with a confidence of {prediction_proba[1]:.2f}.</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='result result-negative'>The model predicts you <strong>do not have diabetes</strong> with a confidence of {prediction_proba[0]:.2f}.</p>", unsafe_allow_html=True)
