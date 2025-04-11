import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = joblib.load("final_xgboost_model.pkl")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="centered")

# -------------------- TITLE --------------------
st.title("📉 Employee Attrition Prediction")
st.markdown("""
A smart tool that predicts if an employee is likely to leave the company based on 5 key workplace factors.

- Powered by XGBoost
- Focused on simplicity & clarity
- Ideal for quick HR insights
""")

# -------------------- SIDEBAR --------------------
st.sidebar.header("🎛️ Enter Employee Details")
age = st.sidebar.slider("Age", 18, 60, 30)
monthly_rate = st.sidebar.slider("Monthly Rate", 1000, 20000, 10000)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
work_life_balance = st.sidebar.slider("Work Life Balance (1=Bad, 4=Great)", 1, 4, 3)

# -------------------- INPUT PREP --------------------
job_role_dict = {"Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2, "Manager": 3}
job_role_encoded = job_role_dict.get(job_role, 0)

input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyRate": [monthly_rate],
    "MonthlyIncome": [monthly_income],
    "JobRole": [job_role_encoded],
    "WorkLifeBalance": [work_life_balance]
})

# -------------------- PREDICTION --------------------
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]  # Probability of leaving

# -------------------- OUTPUT --------------------
st.markdown("### 📊 Prediction Result")
if prediction == 1:
    st.error(f"❌ The employee is **likely to leave**.")
else:
    st.success(f"✅ The employee is **likely to stay**.")

# Confidence level
st.metric(label="Attrition Risk Probability", value=f"{probability*100:.1f}%")

# Confidence advice
if probability > 0.85:
    st.warning("⚠️ High confidence: Consider proactive retention steps.")
elif probability > 0.65:
    st.info("🔎 Moderate risk: Monitor engagement.")
else:
    st.success("✅ Low risk: No action needed.")

# -------------------- INPUT SUMMARY --------------------
st.markdown("### 🧾 Summary of Input")
st.table(pd.DataFrame({
    "Feature": ["Age", "Monthly Rate", "Monthly Income", "Job Role", "Work Life Balance"],
    "Value": [age, monthly_rate, monthly_income, job_role, work_life_balance]
}))

# -------------------- DOWNLOAD OPTION --------------------
result = input_data.copy()
result["Prediction"] = ["Leave" if prediction == 1 else "Stay"]
result["Probability"] = [f"{probability:.2%}"]
csv = result.to_csv(index=False)
st.download_button("⬇️ Download This Result as CSV", csv, "attrition_prediction.csv", "text/csv")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("👨‍💻 Built by **Sharath Reddy** | [GitHub Repo](https://github.com/SharathReddyKottam/emp-attrition-pred)")
