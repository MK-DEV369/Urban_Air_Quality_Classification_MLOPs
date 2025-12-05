from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import time

# PROMETHEUS IMPORTS
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response


# ---------------------
# LOAD MODEL
# ---------------------
model = joblib.load("models/best_pm25_model.pkl")


# ---------------------
# PROMETHEUS METRICS
# ---------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint"]
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions",
    ["model_name"]
)

LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of prediction endpoint"
)


# ---------------------
# FASTAPI APP
# ---------------------

app = FastAPI(
    title="PM2.5 Air Quality Prediction API",
    description="Predict PM2.5 using trained ML model",
    version="1.0"
)


# ---------------------
# INPUT Schema
# ---------------------

class AirQualityInput(BaseModel):
    PM10: float
    O3: float
    CO: float
    hour: int
    dayofweek: int
    month: int


# ---------------------
# ROUTES
# ---------------------

@app.get("/")
def home():
    REQUEST_COUNT.labels("GET", "/").inc()
    return {"message": "PM2.5 Prediction API is running!"}


@app.post("/predict")
def predict(input_data: AirQualityInput):

    REQUEST_COUNT.labels("POST", "/predict").inc()
    start_time = time.time()

    # Prepare features
    features = np.array([[
        input_data.PM10,
        input_data.O3,
        input_data.CO,
        input_data.hour,
        input_data.dayofweek,
        input_data.month
    ]])

    # Predict
    prediction = model.predict(features)[0]

    # PROMETHEUS LOGS
    LATENCY.observe(time.time() - start_time)
    PREDICTION_COUNT.labels(type(model).__name__).inc()

    return {
        "PM25_prediction": float(prediction),
        "model_used": type(model).__name__
    }


# ---------------------
# PROMETHEUS ENDPOINT
# ---------------------

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
