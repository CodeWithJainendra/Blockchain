from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the trained machine learning model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the request
    gender = int(request.form.get('gender'))
    age = int(request.form.get('age'))
    height = float(request.form.get('height'))
    weight = float(request.form.get('weight'))
    ap_hi = int(request.form.get('ap_hi'))
    ap_lo = int(request.form.get('ap_lo'))
    cholesterol = int(request.form.get('cholesterol'))
    gluc = int(request.form.get('gluc'))
    smoke = int(request.form.get('smoke'))
    alco = int(request.form.get('alco'))
    active = int(request.form.get('active'))

    # Create an input array from the extracted features
    input_query = np.array([[gender, age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

    # Make prediction using the trained model
    result = model.predict(input_query)[0]

    # Determine if cardiovascular disease is predicted or not
    if result == 1:
        prediction = "Cardiovascular disease detected."
    else:
        prediction = "No cardiovascular disease detected."

    # Return the prediction result
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
