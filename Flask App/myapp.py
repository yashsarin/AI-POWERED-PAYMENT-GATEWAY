from flask import Flask, request, jsonify
import pickle
import pandas as pd

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load the model
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json(force=True)
    print(data)

    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]

    # Return prediction as JSON
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
