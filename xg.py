import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("dataset.csv")

# Drop invalid rows BEFORE defining X and y
data = data[data['Attrition'].isin(['Yes', 'No'])]
data['Attrition'] = data['Attrition'].map({'No': 0, 'Yes': 1})

# Encode categorical column
label_encoder = LabelEncoder()
data['JobRole'] = label_encoder.fit_transform(data['JobRole'])

# Select only top 5 features
selected_features = ['Age', 'MonthlyRate', 'MonthlyIncome', 'JobRole', 'WorkLifeBalance']
X = data[selected_features]
y = data['Attrition']

# SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
xgb.plot_importance(model)
plt.title("Feature importance")
plt.show()

# Save model
joblib.dump(model, "/Users/sharathkottam/Desktop/EA&P_project/final_xgboost_model.pkl")
