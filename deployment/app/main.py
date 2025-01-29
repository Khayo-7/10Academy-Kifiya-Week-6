import os
import sys
import logging
from typing import List, Union

from fastapi import FastAPI, HTTPException
from app.prediction import make_prediction
from app.schemas import CreditScoresPredictionInput, CreditScoresPredictionOutput

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from scripts.utils.logger import setup_logger
except ImportError as e:
    logging(f"Import error: {e}. Please check the module path.")

# Setup logger for deployement
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
logger = setup_logger("fastapi_deployement", log_dir)

logger.info("Starting process...")

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Starting root function...")
    response = {"message": "Welcome to the Credit scoring API! The server is running!"}
    logger.info("Ending root function...")
    return response

@app.post("/predict_credit_scores", response_model=Union[CreditScoresPredictionOutput, List[CreditScoresPredictionOutput]])
async def predict_credit_scores(input_data: Union[CreditScoresPredictionInput, List[CreditScoresPredictionInput], dict]):
    """
    Unified endpoint to predict Credit Score for various input formats / single or batch inputs:
    - Single instance of PredictionInput.
    - List of PredictionInput instances.
    - Dictionary with "columns" and "data" keys (DataFrame-like input).
    """

    logger.info("Starting unified prediction endpoint...")

    try:
        if isinstance(input_data, dict):
            if "columns" in input_data and "data" in input_data:
                # DataFrame-like batch input
                columns, data = input_data["columns"], input_data["data"]
                input_dicts = [dict(zip(columns, row)) for row in data]
                predictions = make_prediction(input_dicts)
            else:
                # Single dictionary input
                predictions = make_prediction(input_data)
        elif isinstance(input_data, list):
            input_dicts = [entry.model_dump() for entry in input_data]
            predictions = make_prediction(input_dicts)
        else:
            predictions = make_prediction(input_data.model_dump())

        # Format the response
        if isinstance(predictions, list) and len(predictions) > 1:
            logger.info("Successfully completed batch prediction.")
            return {"PredictedCreditScores": predictions}
        else:
            logger.info("Successfully completed single prediction.")
            return {"PredictedCreditScores": predictions[0]}

    except Exception as e:
        logger.error("Error in prediction endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error.")

logger.info("Ending process...")

# uvicorn app.main:app --reload --host 0.0.0.0 --port 7777