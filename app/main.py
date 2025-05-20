import joblib
import os
import logging
import pandas as pd

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Literal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s" 
)

# Creating the logger
logger = logging.getLogger(__name__)

app = FastAPI()


def load_model_and_transformer(model_path, transformer_path):
    if not os.path.exists(model_path):
        logger.error("Model file not found!")
        raise FileNotFoundError(f"{model_path} not found")

    if not os.path.exists(transformer_path):
        logger.error("Transformer file not found!")
        raise FileNotFoundError(f"{transformer_path} not found")
        
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    
    logger.info("Model loaded successfully.")
    
    return model, transformer

model, transformer = load_model_and_transformer("model.pkl", "col_transformer.pkl")
cols = ["CreditScore", "Geography", "Gender", "Age", 
        "Tenure", "Balance", "NumOfProducts", "HasCrCard", 
        "IsActiveMember", "EstimatedSalary",
]

class InputData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1] 
    EstimatedSalary: float
    
@app.get("/")
def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Welcome to the ML prediction API"}

@app.get("/health")
def health_check():
    logger.info("Health check endpoind accessed.")
    return  {"status": "ok"}

@app.post("/predict")
def predict(input:  InputData):
    try:
        logger.info(f"Received input ...")
        
        data = [[getattr(input, col) for col in cols]]
        data = pd.DataFrame(data, columns = cols)
        
        logging.info("Transforming the data...")
        data = transformer.transform(data)
        logging.info("Data transformed Successfully!") 
        
        logging.info("Predicting ...")
        prediction = model.predict(data)
        
        logger.info(f"Prediction: {prediction}")
        
        return {"prediciton": int(prediction[0])}
    
    except Exception as e:
        logger.error(f"Prediciton failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
        
