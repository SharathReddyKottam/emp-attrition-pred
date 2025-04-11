import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv("dataset.csv")

# Step 2: Drop irrelevant columns
data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18'])

# Step 3: Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Step 4: Split data into features (X) and target variable (y)
X = data.drop(columns=['Attrition'])  # All columns except 'Attrition' are features
y = data['Attrition']  # 'Attrition' is the target variable

# Step 5: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 6: Split the resampled data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 7: Define the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Step 8: Define the hyperparameters for Grid Search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l2'],  # 'l1' for Lasso or 'l2' for Ridge regularization
    'solver': ['liblinear', 'saga']  # solvers compatible with l1 and l2
}

# Step 9: Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Step 10: Fit GridSearchCV on the resampled data
grid_search.fit(X_train, y_train)

# Step 11: Get the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Step 12: Train the model with the best hyperparameters
best_log_reg = grid_search.best_estimator_

# Step 13: Predict on the test set
y_pred = best_log_reg.predict(X_test)

# Step 14: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Accuracy and Classification Report
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{class_report}")

# Step 15: Confusion Matrix Visualization
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
