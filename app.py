import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import random
import math

# Load Model & Scaler
model = joblib.load('/content/diabetes_xgb_model.pkl')
scaler = joblib.load('/content/scaler.pkl')

# Dictionaries for categorical variables
gender_dict = {1: "Male", 2: "Female"}
ethnicity_dict = {1: "Mexican American",2: "Other Hispanic",3: "Non-Hispanic White",
                  4: "Non-Hispanic Black",6: "Non-Hispanic Asian",7: "Other Race - Including Multi-Racial"}
education_dict = {1: "Less than 9th grade",2: "9-11th grade",3: "High school/GED",
                  4: "Some college",5: "College graduate or above"}
marital_dict = {1: "Married",2: "Widowed",3: "Divorced",
                4: "Separated",5: "Never married",6: "Living with partner"}
birth_country_dict = {1: "USA",2: "Other country"}

# Streamlit App
st.title("Diabetes Risk Prediction Dashboard")
st.write("Enter your details to estimate the probability of diabetes.")

# User inputs:

# Categorical Inputs
gender = st.selectbox("Select Gender", options=[None]+list(gender_dict.keys()),
                      format_func=lambda x: "Select..." if x is None else gender_dict[x])
ethnicity = st.selectbox("Select Ethnicity", options=[None]+list(ethnicity_dict.keys()),
                         format_func=lambda x: "Select..." if x is None else ethnicity_dict[x])
education = st.selectbox("Select Education Level", options=[None]+list(education_dict.keys()),
                         format_func=lambda x: "Select..." if x is None else education_dict[x])
marital_status = st.selectbox("Select Marital Status", options=[None]+list(marital_dict.keys()),
                              format_func=lambda x: "Select..." if x is None else marital_dict[x])
birth_country = st.selectbox("Select Birth Country", options=[None]+list(birth_country_dict.keys()),
                             format_func=lambda x: "Select..." if x is None else birth_country_dict[x])

# Numerical Inputs
age = st.number_input("Age", min_value=20, max_value=80)
bmi = st.number_input("BMI", min_value=0.0)
weight = st.number_input("Weight (kg)", min_value=0.0)
height = st.number_input("Height (cm)", min_value=0.0)
waist = st.number_input("Waist (cm)", min_value=0.0)
systolic_bp = st.number_input("Systolic BP", min_value=0)
diastolic_bp = st.number_input("Diastolic BP", min_value=0)
glucose = st.number_input("Glucose", min_value=0.0)
hba1c = st.number_input("HbA1c", min_value=0.0)


# Manual Prediction Button

if st.button("Predict Diabetes Risk"):
    input_data = pd.DataFrame({
        'Age': [age], 'BMI': [bmi], 'weight': [weight], 'height': [height],
        'waist': [waist], 'systolic_bp': [systolic_bp], 'diastolic_bp': [diastolic_bp],
        'Glucose': [glucose], 'HbA1c': [hba1c], 'Gender': [gender],
        'Ethnicity': [ethnicity], 'Education': [education],
        'Marital_Status': [marital_status], 'Birth_Country': [birth_country]
    })

    # Encode categorical
    categorical_cols = ['Gender','Ethnicity','Education','Marital_Status','Birth_Country']
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Align with training
    for col in model.feature_names_in_:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model.feature_names_in_]

    # Scale numeric
    numeric_cols = ['Age', 'BMI', 'weight', 'height', 'waist',
                    'systolic_bp', 'diastolic_bp', 'Glucose', 'HbA1c']
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    # Predict
    pred_proba = model.predict_proba(input_encoded)[:,1]
    st.subheader("Prediction Result")
    st.write(f"Estimated probability of diabetes: {pred_proba[0]*100:.2f}%")
    if pred_proba[0] >= 0.4:
        st.warning("High risk of diabetes ⚠️")
    else:
        st.success("Low risk of diabetes ✅")


# Live Real-Time Patient Monitoring Simulation

# "Generate New Patient" button that simulates a new patient arriving
def generate_live_patient():
    return pd.DataFrame([{
        'Age': random.randint(20, 80),
        'BMI': round(random.uniform(18, 45),1),
        'weight': round(random.uniform(50, 120),1),
        'height': round(random.uniform(150, 200),1),
        'waist': round(random.uniform(70, 120),1),
        'systolic_bp': random.randint(90, 180),
        'diastolic_bp': random.randint(60, 120),
        'Glucose': round(random.uniform(70, 180),1),
        'HbA1c': round(random.uniform(4.5, 12.0),1),
        'Gender': random.choice([1,2]),
        'Ethnicity': random.choice(list(ethnicity_dict.keys())),
        'Education': random.choice(list(education_dict.keys())),
        'Marital_Status': random.choice(list(marital_dict.keys())),
        'Birth_Country': random.choice(list(birth_country_dict.keys()))
    }])


st.subheader("Simulated Live Patient Data")
if st.button("Generate Live Patient"):
    # Generate random patient
    live_patient = generate_live_patient()

    # Display all patient details
    st.write("### Patient Details Entered / Generated")
    st.dataframe(live_patient)  # This prints all features in a table

    # One-hot encode categorical columns
    categorical_cols = ['Gender','Ethnicity','Education','Marital_Status','Birth_Country']
    live_encoded = pd.get_dummies(live_patient, columns=categorical_cols, drop_first=True)

    # Align with training columns
    for col in model.feature_names_in_:
        if col not in live_encoded.columns:
            live_encoded[col] = 0
    live_encoded = live_encoded[model.feature_names_in_]

    # Scale numeric columns
    numeric_cols = ['Age', 'BMI', 'weight', 'height', 'waist',
                    'systolic_bp', 'diastolic_bp', 'Glucose', 'HbA1c']
    live_encoded[numeric_cols] = scaler.transform(live_encoded[numeric_cols])

    # Make prediction
    live_pred = model.predict_proba(live_encoded)[:,1][0]

    st.subheader("Live Prediction")
    st.write(f"Estimated probability of diabetes: {live_pred*100:.2f}%")
    if live_pred >= 0.4:
        st.warning("High risk of diabetes ⚠️")
    else:
        st.success("Low risk of diabetes ✅")


    # Resource Optimization
    # Initialize session state counter for high-risk patients
    if 'high_risk_count' not in st.session_state:
        st.session_state.high_risk_count = 0

    # Update high-risk count
    if live_pred >= 0.4:
        st.session_state.high_risk_count += 1


    # Simple hospital resource rules
    high_risk = st.session_state.high_risk_count
    doctors_needed = math.ceil(high_risk / 10)  # +1 doctor per 10 high-risk patients
    test_kits_needed = high_risk * 2            # 2 test kits per high-risk patient

    st.subheader("Resource Recommendations")
    st.write(f"High-risk patients today: {high_risk}")
    st.write(f"Doctors needed: {doctors_needed}")
    st.write(f"Test kits needed: {test_kits_needed}")
