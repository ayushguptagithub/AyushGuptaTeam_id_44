import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('merged_csv_final2.csv')

# Drop the 'date' column
data.drop(['date'], axis=1, inplace=True)

# Separate features (X) and target variable (y)
X = data.drop('stability', axis=1)
y = data['stability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create a neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training sets
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Make predictions on the testing set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Compare actual response values (y_test) with predicted response values (y_pred)
print("Neural Network model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
