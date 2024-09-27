from flask import Flask, request, jsonify
import argparse
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--trained_model_path", type=str, help="trained model file path") 
args = parser.parse_args()
model = joblib.load(args.trained_model_path)

@app.route('/')
def home():
    return "Dream ML Is Serving Predictions"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction using the loaded model.
    Expects a POST request with JSON input.
    """

    data = request.get_json(force=True)
    
    try:
        row = pd.Series(data).to_frame().T
        prediction = model.predict(row)
        response = {
            'prediction': int(prediction[0])
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
