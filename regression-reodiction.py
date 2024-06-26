import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_excel('merged_file_avu.xlsx')
X = df[['Air temperature', 'Pressure', 'Wind speed']]
y = df['Power generated by system']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error (Random Forest):', mse)

# Load new data for prediction
new_data = pd.read_excel("wind_power_gen_3months_validation_data.xlsx")
X_new = new_data[['Air temperature', 'Pressure', 'Wind speed']]

# Scale the new data
X_new_scaled = scaler.transform(X_new)

# Predict on the new data
predicted_power = model.predict(X_new_scaled)

# Add predictions to the new data
new_data['Predicted Power'] = predicted_power

# Save the new data with predictions
new_data.to_csv('test_data_with_predictions_random_forest.csv', index=False)
