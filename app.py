import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("final_xgboost_model.pkl")

# UI
st.title("Employee Attrition Prediction")
st.sidebar.header("Enter Input Features")

# Inputs matching trained model's features
age = st.sidebar.slider("Age", 18, 60, 30)
monthly_rate = st.sidebar.slider("Monthly Rate", 1000, 20000, 10000)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 3)

# Encode JobRole
job_role_dict = {"Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2, "Manager": 3}
job_role_encoded = job_role_dict.get(job_role, 0)

# Create dataframe for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyRate": [monthly_rate],
    "MonthlyIncome": [monthly_income],
    "JobRole": [job_role_encoded],
    "WorkLifeBalance": [work_life_balance]
})

# Prediction
prediction = model.predict(input_data)

# Display result
if prediction[0] == 1:
    st.subheader("Prediction: ❌ Employee is likely to leave.")
else:
    st.subheader("Prediction: ✅ Employee is likely to stay.")
