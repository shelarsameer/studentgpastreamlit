import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
data = pd.read_csv('f:/datasets/Student_performance_data _.csv')

# Prepare features and target
X = data.drop(['StudentID', 'GPA', 'GradeClass'], axis=1)
y = data['GPA']

# Encode categorical variables
X = pd.get_dummies(X, columns=['Gender', 'Ethnicity', 'ParentalEducation'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the model and scaler
joblib.dump(model, 'student_gpa_model.joblib')
joblib.dump(scaler, 'student_gpa_scaler.joblib')