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
        background-color: #f4f7fa;
        margin: 0;
        padding: 0;
    }
    .title {
        text-align: center;
        color: #3e8e41;
        font-size: 2.5em;
        margin-top: 50px;
        font-weight: 700;
    }
    .description {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-top: 10px;
        margin-bottom: 40px;
        padding: 0 10%;
    }
    .input-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
        max-width: 800px;
        margin: 0 auto;
    }
    .input-label {
        font-weight: 600;
        font-size: 1.1em;
    }
    .btn {
        background-color: #3e8e41;
        color: white;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1.2em;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .btn:hover {
        background-color: #2c6f34;
    }
    .result {
        font-size: 1.5em;
        font-weight: 700;
        margin-top: 30px;
        padding: 10px;
        text-align: center;
    }
    .result-positive {
        color: #e74c3c;
    }
    .result-negative {
        color: #2ecc71;
    }
    .required {
        color: #e74c3c;
        font-weight: bold;
    }
    .input-container input {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
        border: 2px solid #ddd;
        font-size: 1.1em;
        width: 100%;
        transition: border-color 0.3s ease;
    }
    .input-container input:focus {
        outline: none;
        border-color: #3e8e41;
    }
    .input-container input[placeholder] {
        color: #aaa;
    }
    .columns {
        display: flex;
        justify-content: space-between;
        gap: 20px;
    }
    .col {
        flex: 1;
    }
    .error-msg {
        color: #e74c3c;
        font-weight: bold;
        margin-top: 10px;
    }
    .error-msg-text {
        font-size: 0.9em;
    }
    .form-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    .form-group {
        flex: 1;
        min-width: 250px;
        padding: 10px;
    }
    .form-group label {
        display: block;
        margin-bottom: 5px;
    }
    .form-group input {
        width: 100%;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #ddd;
        font-size: 1em;
        transition: border-color 0.3s ease;
    }
    .form-group input:focus {
        border-color: #3e8e41;
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
    missing_fields = []
    if not pregnancies:
        missing_fields.append('Pregnancies')
    if not glucose:
        missing_fields.append('Glucose Level')
    if not blood_pressure:
        missing_fields.append('Blood Pressure')
    if not age:
        missing_fields.append('Age')

    if missing_fields:
        st.markdown(f"<p class='error-msg'>Please fill all the required fields marked with a star (*): {', '.join(missing_fields)}</p>", unsafe_allow_html=True)
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
        
