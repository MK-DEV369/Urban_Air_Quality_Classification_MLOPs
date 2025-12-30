from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import time
import logging
import os

# PROMETHEUS IMPORTS
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response


# ---------------------
# LOAD MODEL
# ---------------------
model_path = "models/best_pm25_model.pkl"
if not os.path.exists(model_path):
    print(f"⚠️  Warning: Model file not found at {model_path}")
    print("   Run: python scripts/train_with_comet.py")
    model = None
else:
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")


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

UP = Counter(
    "up",
    "Service health (1 = up)"
)


# ---------------------
# FASTAPI APP
# ---------------------

app = FastAPI(
    title="PM2.5 Air Quality Prediction API",
    description="Predict PM2.5 using trained ML model",
    version="1.0"
)


logger = logging.getLogger("audit")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def startup_event():
    """Log startup."""
    print("🚀 FastAPI server starting on http://0.0.0.0:8000")
    print("📊 Metrics available at http://0.0.0.0:8000/metrics")
    print("📝 API docs at http://0.0.0.0:8000/docs")
    UP.inc()


@app.middleware("http")
async def audit_logging_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


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


@app.get("/health")
def health():
    """Health check endpoint."""
    REQUEST_COUNT.labels("GET", "/health").inc()
    status = "healthy" if model is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(input_data: AirQualityInput):
    """Predict PM2.5 levels."""
    REQUEST_COUNT.labels("POST", "/predict").inc()
    start_time = time.time()

    if model is None:
        return {
            "error": "Model not loaded. Train model first: python scripts/train_with_comet.py",
            "status": "unavailable"
        }, 503

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


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")

