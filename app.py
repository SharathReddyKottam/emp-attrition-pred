import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("final_xgboost_model.pkl")

# -------------------- UI Header --------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("📉 Employee Attrition Prediction Dashboard")
st.markdown(
    """
    This tool predicts whether an employee is likely to leave the company based on key features like salary, role, and work-life balance.
    
    _Use the sidebar to enter employee details and view the prediction below._  
    """
)

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("🔧 Input Features")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_rate = st.sidebar.slider("Monthly Rate", 1000, 20000, 10000)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 3)

# Encode job role
job_role_dict = {"Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2, "Manager": 3}
job_role_encoded = job_role_dict.get(job_role, 0)

# -------------------- Prepare Input Data --------------------
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyRate": [monthly_rate],
    "MonthlyIncome": [monthly_income],
    "JobRole": [job_role_encoded],
    "WorkLifeBalance": [work_life_balance]
})

# -------------------- Prediction --------------------
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]  # Probability of leaving

# -------------------- Output Section --------------------
st.markdown("### 🧾 Prediction Result")
if prediction == 1:
    st.error(f"❌ The employee is **likely to leave**.\n\n💡 Attrition probability: **{probability:.0%}**")
else:
    st.success(f"✅ The employee is **likely to stay**.\n\n🧠 Attrition probability: **{probability:.0%}**")

# -------------------- Display Input Summary --------------------
st.markdown("### 📋 Summary of Your Input")
input_display = {
    "Age": age,
    "Monthly Rate": monthly_rate,
    "Monthly Income": monthly_income,
    "Job Role": job_role,
    "Work-Life Balance": work_life_balance
}
st.table(pd.DataFrame(input_display.items(), columns=["Feature", "Value"]))

# -------------------- Optional: Download Button --------------------
result_data = input_data.copy()
result_data["Prediction"] = ["Leave" if prediction == 1 else "Stay"]
result_data["Probability"] = [f"{probability:.2%}"]
csv = result_data.to_csv(index=False)
st.download_button("⬇️ Download Prediction as CSV", csv, "prediction_result.csv", "text/csv")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("👨‍💻 Built by **Sharath Reddy** | [GitHub Repo](https://github.com/SharathReddyKottam/emp-attrition-pred)")
