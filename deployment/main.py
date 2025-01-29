import os
import sys
import logging
import pandas as pd
from flask_cors import CORS
from flask import Flask, request, jsonify
from app.prediction import make_prediction
from app.schemas import CreditScoresPredictionInput, CreditScoresPredictionOutput

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from scripts.utils.logger import setup_logger
except ImportError as e:
    logging(f"Import error: {e}. Please check the module path.")

# Setup logger for deployement
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
logger = setup_logger("flask_deployement", log_dir)

logger.info("Starting process...")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Define a route to test the API
@app.route("/", methods=["GET"])
async def home():
    return jsonify({"message": "Welcome to the Credit Scoring API! The server is running!"})

# Define a route for predictions
@app.route("/predict_credit_scores", methods=["POST"])
async def predict():
    """
    Endpoint to predict Credit scores using the model.
    """
    try:
        input_dict = input_data.model_dump()

        # Parse the input JSON data
        input_data = request.json 
        raw_data = pd.DataFrame(input_data)

        # Generate predictions
        predictions = make_prediction(input_dict)
        predicted_credit_scores = predictions.flatten().tolist()

        # Return predictions as JSON
        return jsonify({"predicted_credit_scores": predicted_credit_scores})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Run Flask server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7777)

# python app.py