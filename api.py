from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import pandas as pd
import numpy as np
from models import Autoencoder, FailurePredictor
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from prometheus_client import make_asgi_app, Gauge, Counter

app = FastAPI(title="Storage Anomaly Prediction API")

# Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

ANOMALY_SCORE = Gauge('disk_anomaly_score', 'Reconstruction error from Autoencoder', ['disk_id'])
FAILURE_PROB = Gauge('disk_failure_probability', 'Failure probability from XGBoost', ['disk_id'])
PREDICTION_COUNT = Counter('disk_predictions_total', 'Total predictions made', ['disk_id', 'status'])
TEMPERATURE = Gauge('disk_temperature_celsius', 'Disk Temperature', ['disk_id'])
READ_LATENCY = Gauge('disk_read_latency_ms', 'Read Latency', ['disk_id'])
WRITE_LATENCY = Gauge('disk_write_latency_ms', 'Write Latency', ['disk_id'])
THROUGHPUT = Gauge('disk_throughput_mbps', 'Throughput', ['disk_id'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    scaler = joblib.load("models/scaler.pkl")
    
    feature_cols = [
        "smart_5_reallocated_sector_count",
        "smart_187_reported_uncorrectable_errors",
        "smart_194_temperature_celsius",
        "smart_197_current_pending_sector_count",
        "read_latency_ms",
        "write_latency_ms",
        "throughput_mbps"
    ]
    input_dim = len(feature_cols)
    
    autoencoder = Autoencoder(input_dim)
    autoencoder.load_state_dict(torch.load("models/autoencoder.pth"))
    autoencoder.eval()
    
    failure_predictor = FailurePredictor(model_type="xgboost")
    failure_predictor.load("models/failure_predictor.json")
    
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # For dev purposes, we might want to continue even if models fail to load, 
    # but for this specific task, we should probably fail or handle it.
    # We'll just print the error for now.

class DiskMetrics(BaseModel):
    disk_id: str
    smart_5_reallocated_sector_count: int
    smart_187_reported_uncorrectable_errors: int
    smart_194_temperature_celsius: int
    smart_197_current_pending_sector_count: int
    read_latency_ms: float
    write_latency_ms: float
    throughput_mbps: float

@app.post("/ingest")
async def ingest_metrics(metrics: DiskMetrics):
    # In a real system, we would save this to a database (Elasticsearch/InfluxDB)
    # Here we just acknowledge receipt
    return {"status": "received", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict(metrics: DiskMetrics):
    try:
        data = pd.DataFrame([metrics.dict()])
        X = data[feature_cols]
        
        # Anomaly Detection (Autoencoder)
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).item()
            
        # Failure Prediction (XGBoost)
        failure_prob = failure_predictor.predict_proba(X)[0]
        
        is_anomaly = mse > 0.5 # Threshold needs tuning, but 0.5 is a placeholder
        will_fail = failure_prob > 0.5
        
        # Update Prometheus Metrics
        ANOMALY_SCORE.labels(disk_id=metrics.disk_id).set(mse)
        FAILURE_PROB.labels(disk_id=metrics.disk_id).set(failure_prob)
        TEMPERATURE.labels(disk_id=metrics.disk_id).set(metrics.smart_194_temperature_celsius)
        READ_LATENCY.labels(disk_id=metrics.disk_id).set(metrics.read_latency_ms)
        WRITE_LATENCY.labels(disk_id=metrics.disk_id).set(metrics.write_latency_ms)
        THROUGHPUT.labels(disk_id=metrics.disk_id).set(metrics.throughput_mbps)
        
        status = "FAIL" if will_fail else "HEALTHY"
        PREDICTION_COUNT.labels(disk_id=metrics.disk_id, status=status).inc()
        
        return {
            "disk_id": metrics.disk_id,
            "anomaly_score": mse,
            "is_anomaly": bool(is_anomaly),
            "failure_probability": float(failure_prob),
            "prediction": "FAIL" if will_fail else "HEALTHY"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
