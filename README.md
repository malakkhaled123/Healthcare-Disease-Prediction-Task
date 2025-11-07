# Diabetes Risk Prediction Model

## Overview

This project predicts the risk of diabetes using the NHANES dataset (August 2021 – August 2023). The goal is to create a machine learning model and a Streamlit dashboard that allows users to input their health details and receive an estimated probability of diabetes.

**Dataset:** [NHANES 2021-2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)

---

## Dataset
I first collected .xpt files from 4 places [demographics , questionnarres , examinations , lab tests] , then cleaned & merged each one of them in a notebook , These notebooks are provided here in the repository if you want to refer to them.


The Final dataset contains **7720 samples** with the following **features**:

| Feature              | Description                            |
| -------------------- | -------------------------------------- |
| ID                   | Unique identifier for each participant |
| Gender               | Male / Female                          |
| Age                  | Years                                  |
| Ethnicity            | Race/Hispanic origin                   |
| Education            | Education level                        |
| Marital_Status       | Marital status                         |
| Income_Ratio         | Family income to poverty ratio         |
| Birth_Country        | Country of birth                       |
| BMI                  | Body Mass Index                        |
| weight               | Weight (kg)                            |
| height               | Height (cm)                            |
| waist                | Waist circumference (cm)               |
| systolic_bp          | Systolic blood pressure                |
| diastolic_bp         | Diastolic blood pressure               |
| HbA1c                | Glycated hemoglobin                    |
| Glucose              | Blood glucose                          |
| **Target: diabetes** | 0 = No diabetes, 1 = Diabetes          |

---

## Exploratory Data Analysis (EDA)

The following visualizations were created to understand the data and relationships between features and diabetes:

* Age distribution by diabetes status
* BMI vs diabetes
* Comparison of Glucose and HbA1c between diabetic and non-diabetic participants
* Correlation heatmap of features
* Age group analysis



<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/99d7b61d-aa65-4e62-9748-39dcd71890b3" />

The histogram clearly shows that age is a very important predictor for diabetes as it increases the risk of diabetes increase especially in 40+ ages , but there is a severe class imbalance so this means that I must prioritize metrics like Precision , Recall , F1-score and AUC-ROC curve over accuracy to build a reliable model.

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/7abc235d-2187-473c-927c-54af90473bde" />

Here the box plot shows that the median BMI for the diabetic group is visibly higher than the non-diabetic group , also the boxplot confirms that higher BMI strongly predicts diabetes, meaning it will be a key feature for the model, especially in identifying high-risk patients.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/e558c5ac-b769-4b6b-9471-cf6c4bf54539" />

Here we can see that diabetes patients have higher average glucose , HbA1c values than the non-diabetic people ,making it also a strong predictor for our model.


<img width="584" height="504" alt="image" src="https://github.com/user-attachments/assets/babc7dd6-6450-4034-ae6d-4fc98a30bdcb" />

It shows that BMI , waist , weight , are highly correlated

---

## Feature Engineering

* Merged all relevant tables using the **SEQN** column
* Selected medically relevant features
* One-hot encoded categorical variables
* Standardized numeric features using **StandardScaler**
* Saved cleaned dataset: `data_processed/diabetes_clean.csv`

---

## Machine Learning Models

Three models were trained and evaluated:

| Model                                   | Accuracy | Precision (Diabetic) | Recall (Diabetic) | F1-Score (Diabetic) | ROC-AUC |
| --------------------------------------- | -------- | -------------------- | ----------------- | ------------------- | ------- |
| Logistic Regression                     | 0.84     | 0.45                 | 0.75              | 0.56                | 0.802   |
| Random Forest (class_weight='balanced') | 0.91     | 0.79                 | 0.48              | 0.60                | 0.77    |
| XGBoost (Tuned, threshold=0.4)          | 0.88     | 0.55                 | 0.67              | 0.60                | 0.891   |

**Top Features in Feature Importance (XGBoost):**

1. HbA1c
2. Glucose
3. BMI
4. Age

---

## Threshold Tuning

* XGBoost model threshold was adjusted to **0.4** to improve recall for diabetic class
* This balances detecting high-risk patients while maintaining reasonable precision

---

## Streamlit Dashboard

The dashboard allows a user to enter health details and get a predicted probability of diabetes:

**Features in Dashboard:**

* Age, Gender, Ethnicity, Education, Marital Status, Birth Country
* BMI, Weight, Height, Waist, Systolic BP, Diastolic BP, Glucose, HbA1c

**How to run locally:**

```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-risk-dashboard.git
cd diabetes-risk-dashboard

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

*Dashboard screenshots:*

<img width="1918" height="961" alt="image" src="https://github.com/user-attachments/assets/329a53bd-c25b-4010-8feb-59502826ee79" />
<img width="1912" height="967" alt="image" src="https://github.com/user-attachments/assets/97780439-c7bf-4746-86d5-dd0537e2fb33" />


---

## Repository Structure

```
diabetes-risk-dashboard/
│
├── data/
│   ├── cleaned/
│   │   ├── demographics.csv
│   │   ├── diabetes_clean.csv
│   │   ├── exam.csv
│   │   └── lab_tests.csv
│   ├── raw/
│   │   ├── demographics/
│   │   │   └── DEMO_L.xpt
│   │   ├── examination/
│   │   │   ├── BMX_L.xpt
│   │   │   └── BPXO_L.xpt
│   │   ├── lab tests/
│   │   │   ├── GHB_L.xpt
│   │   │   └── GLU_L.xpt
│   │   └── questionnares/
│   │       └── DIQ_L.xpt
│
├── models/
│   ├── diabetes_xgb_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_clean_demo.ipynb
│   ├── 02_clean_labs.ipynb
│   ├── 03_clean_exams.ipynb
│   └── Healthcare_Prediction_Final.ipynb
│
├── app.py                # Streamlit dashboard
├── README.md             # Project documentation
└── requirements.txt      # Dependencies for local setup / Streamlit

```

---

## Dependencies

* Python 3.8+
* pandas, numpy
* scikit-learn
* xgboost
* streamlit
* joblib

---

## Conclusion:
This project illustrates how machine learning models can be applied in real-world healthcare scenarios to enhance patient care, reduce risks, and support proactive medical decision-making. It aligns with the broader vision of predictive analytics in healthcare, enabling hospitals to combine data-driven insights with clinical expertise for improved outcomes.

--- 

## Author

Malak Khaled – malakkhaleds2mm@gmail.com / https://www.linkedin.com/in/malak-khaled-056432259/ https://github.com/malakkhaled123

---
