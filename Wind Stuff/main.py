from flask import Flask, render_template, request  # Add Flask, render_template, and request imports
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('trained_model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        air_temperature = float(request.form['air_temperature'])
        pressure = float(request.form['pressure'])
        wind_speed = float(request.form['wind_speed'])
        
        # Make a prediction using the pre-trained model
        prediction = model.predict([[air_temperature, pressure, wind_speed]])
        
        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
