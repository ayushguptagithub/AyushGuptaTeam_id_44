import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib  # Import joblib for saving the model

# Load the dataset
df = pd.read_csv('merged_file.csv')
X = df[['Air temperature', 'Pressure', 'Wind speed']]
y = df['Power generated by system']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print('Mean Squared Error (Neural Network):', mse)