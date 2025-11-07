import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

#  Load Model & Scaler
model = joblib.load('/content/diabetes_xgb_model.pkl')
scaler = joblib.load('/content/scaler.pkl')

#  Dictionaries for categorical variables
gender_dict = {1: "Male",2: "Female"}

ethnicity_dict = {1: "Mexican American",2: "Other Hispanic",3: "Non-Hispanic White",
                  4: "Non-Hispanic Black",6: "Non-Hispanic Asian",
                  7: "Other Race - Including Multi-Racial"}

education_dict = {1: "Less than 9th grade",2: "9-11th grade",3: "High school/GED",
                  4: "Some college",5: "College graduate or above"}

marital_dict = {1: "Married",2: "Widowed",3: "Divorced",
                4: "Separated",5: "Never married",6: "Living with partner"}

birth_country_dict = {1: "USA",2: "Other country"}

#  Streamlit App 
st.title("Diabetes Risk Prediction Dashboard")
st.write("Enter your details to estimate the probability of diabetes.")

# User inputs
# Categorical Inputs
gender = st.selectbox("Select Gender", options=[None] + list(gender_dict.keys()),
                      format_func=lambda x: "Select..." if x is None else gender_dict[x])
ethnicity = st.selectbox("Select Ethnicity", options=[None] + list(ethnicity_dict.keys()),
                         format_func=lambda x: "Select..." if x is None else ethnicity_dict[x])
education = st.selectbox("Select Education Level", options=[None] + list(education_dict.keys()),
                         format_func=lambda x: "Select..." if x is None else education_dict[x])
marital_status = st.selectbox("Select Marital Status", options=[None] + list(marital_dict.keys()),
                              format_func=lambda x: "Select..." if x is None else marital_dict[x])
birth_country = st.selectbox("Select Birth Country", options=[None] + list(birth_country_dict.keys()),
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


 
# Prediction button 
if st.button("Predict Diabetes Risk"):
    # Prepare input dataframe
    input_data = pd.DataFrame({
        'Age': [age], 'BMI': [bmi], 'weight': [weight], 'height': [height],
        'waist': [waist], 'systolic_bp': [systolic_bp], 'diastolic_bp': [diastolic_bp],
        'Glucose': [glucose], 'HbA1c': [hba1c], 'Gender': [gender],
        'Ethnicity': [ethnicity], 'Education': [education],
        'Marital_Status': [marital_status], 'Birth_Country': [birth_country]
    })

    # One-hot encode categorical columns (same as training)
    categorical_cols = ['Gender','Ethnicity','Education','Marital_Status','Birth_Country']
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Align columns with training dataset
    for col in model.feature_names_in_:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model.feature_names_in_]

    # Scale numeric columns
    numeric_cols = ['Age', 'BMI', 'weight', 'height', 'waist', 
                    'systolic_bp', 'diastolic_bp', 'Glucose', 'HbA1c']
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    # Make prediction
    pred_proba = model.predict_proba(input_encoded)[:,1]  # Probability of diabetes

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"Estimated probability of diabetes: {pred_proba[0]*100:.2f}%")
    if pred_proba[0] >= 0.4:   # threshold we tuned
        st.warning("High risk of diabetes ⚠️")
    else:
        st.success("Low risk of diabetes ✅")
