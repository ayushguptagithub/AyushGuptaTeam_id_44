import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the merged data from the Excel sheet
df = pd.read_excel('merged_file_avu.xlsx')

# Assume a threshold for power generation
threshold = 100  # Adjust this threshold as needed

# Create a binary target variable based on the threshold
df['Power_above_threshold'] = (df['Power generated by system'] > threshold).astype(int)

# Extract features (inputs) and binary target variable (output)
X = df[['Air temperature', 'Pressure', 'Wind speed']]
y = df['Power_above_threshold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(report)

# Save predictions to a new CSV file
df_test_with_predictions = X_test.copy()
df_test_with_predictions['Actual Power'] = df_test_with_predictions.apply(lambda row: 'Above Threshold' if row['Power generated by system'] > threshold else 'Below Threshold', axis=1)
df_test_with_predictions['Predicted Power'] = ['Above Threshold' if pred == 1 else 'Below Threshold' for pred in y_pred]
df_test_with_predictions.to_csv('test_data_with_predictions_logistic_regression.csv', index=False)
