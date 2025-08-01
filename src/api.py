# src/api.py
import uvicorn
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, conlist
from typing import List

from .model_training import load_model
from .data_preprocessing import load_iris # Use iris data to get feature names

# Define the request body format
class IrisData(BaseModel):
    data: List[conlist(float, min_length=4, max_length=4)]

# Initialize the FastAPI app
app = FastAPI()

# Global variables for the model and feature names
model = None
FEATURE_NAMES = None

@app.on_event("startup")
async def load_trained_model():
    """
    Loads the model and feature names on application startup.
    """
    global model, FEATURE_NAMES
    model_path = "models/iris_classifier.pkl"
    try:
        model = load_model(model_path)
        # Get feature names from the original dataset for consistency
        FEATURE_NAMES = load_iris(as_frame=True).frame.drop(columns=['target']).columns.tolist()
        print("Model and feature names loaded successfully.")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found. Please train the model first.")

@app.get("/")
def home():
    return {"message": "Iris Classifier API is running!"}

@app.post("/predict")
def predict(iris_data: IrisData):
    """
    Makes predictions on new data.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Create a DataFrame from the input data
        df = pd.DataFrame(iris_data.data, columns=FEATURE_NAMES)
        predictions = model.predict(df)
        
        # Convert predictions to a list and return
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data provided: {e}")

if __name__ == "__main__":
    # This block is for local development and testing
    uvicorn.run(app, host="0.0.0.0", port=8000)