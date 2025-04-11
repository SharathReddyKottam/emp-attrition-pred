# 📉 Employee Attrition Prediction Dashboard

A smart and interactive Streamlit dashboard that predicts whether an employee is likely to leave the company based on key workplace factors. Powered by a custom-trained XGBoost model focused on simplicity and impact.

---

## 🚀 Features

- ✅ Predicts attrition likelihood with confidence level
- 📊 Uses top 5 features only: Age, Monthly Rate, Monthly Income, Job Role, Work-Life Balance
- 🔁 Real-time input + prediction output
- 📈 Shows risk probability (percent)
- 📋 Recap of selected inputs in a summary table
- 💾 Download prediction + input as CSV
- 👤 Personalized footer and GitHub link

---

## 📥 How to Use

```bash
# Clone the repo
https://github.com/SharathReddyKottam/emp-attrition-pred

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🧠 Model Info

- **Algorithm**: XGBoost Classifier
- **Trained On**: Instacart HR dataset
- **SMOTE**: Used to handle class imbalance
- **Feature Encoding**: `JobRole` encoded using `LabelEncoder`
- **Final Model Features**:
  - Age
  - MonthlyRate
  - MonthlyIncome
  - JobRole (encoded)
  - WorkLifeBalance

---

## 📸 Dashboard Preview

> Add a screenshot here: <img width="1440" alt="emp-att-pred_OUTPUT" src="https://github.com/user-attachments/assets/d77adf7e-168f-471c-aed2-8b817e39fac8" />

---

## 🛠 Tech Stack

- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas
- imblearn
- seaborn / matplotlib

---

## 📬 Contact

👨‍💻 Built with ❤️ by **Sharath Reddy**  
🔗 [LinkedIn](https://www.linkedin.com/in/sharathreddykottam) • [GitHub](https://github.com/SharathReddyKottam)

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/SharathReddyKottam/emp-attrition-pred/main/app.py)

---

## ✨ Future Ideas

- SHAP visualizations
- Light/dark theme toggle
- Upload CSV for batch prediction
- PDF download of prediction report
