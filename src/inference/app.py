from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

pipeline = joblib.load("heart_model_pipeline.joblib")

@app.post("/predict")
def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0].max()

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
