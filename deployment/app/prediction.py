import os
import sys
import joblib
import logging
from typing import List, Union, Optional

import pandas as pd
import tensorflow.keras.backend as K
from app.utils import preprocess_input
from tensorflow.keras.models import load_model 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from scripts.utils.logger import setup_logger
except ImportError as e:
    logging(f"Import error: {e}. Please check the module path.")

# Setup logger for deployement
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
logger = setup_logger("deployement", log_dir)  

resources_dir = os.path.join('resources')
# model_path = os.path.join(resources_dir, 'credit_scoring_model.h5')
# model_path = os.path.join(resources_dir, 'credit_scoring_model.pkl')
model_path = os.path.join(resources_dir, 'model.pkl')

# Load the trained model
try:
    logger.info("Starting to load the trained model...")
    model = joblib.load(model_path)
    # model = load_model(model_path, custom_objects=None)
    logger.info("Successfully loaded the trained model.")
except Exception as e:
    logger.error(f"Error initializing application: {e}")
    raise

def make_prediction(input_data: Union[dict, List[dict]]) -> Union[dict, List[dict]]:

    """
    Generate credit score predictions for single or batch inputs using the model.

    Args:
    - input_data: A single dictionary, a list of dictionaries or a DataFrame-like 
      dictionary of input features or a list of such dictionaries.

    Returns:
    - Single or batch predictions
    - predicted_credit_scores: Predicted credit score value(s).
    """
    logger.info("Starting prediction...")
    
    # Convert input(s) to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)
    elif "columns" in input_data and "data" in input_data:
        input_df = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    else:
        raise ValueError("Invalid input format")
    
    # Preprocess inputs
    processed_input = preprocess_input(input_df)  # Preprocess all rows at once
    
    # Generate predictions
    predictions = model.predict(processed_input)
    logger.info("Prediction made successfully.")
    
    # Return predictions as list
    predicted_credit_scores = predictions.flatten().tolist()
    logger.info(predicted_credit_scores)
    return predicted_credit_scores
