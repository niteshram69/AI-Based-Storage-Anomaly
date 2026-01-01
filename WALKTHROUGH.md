# Walkthrough: AI-Based Storage Anomaly & Failure Prediction System

I have built a complete prototype for predicting disk failures and detecting anomalies using AI. The system consists of a synthetic data generator, an Autoencoder for anomaly detection, an XGBoost model for failure prediction, a FastAPI backend, and a real-time dashboard.

## Prerequisites

Ensure you have Python 3.9+ installed.

## 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train Models

Generate synthetic data and train the AI models (Autoencoder & XGBoost):

```bash
python3 train.py
```
This will create a `models/` directory containing `autoencoder.pth`, `failure_predictor.json`, and `scaler.pkl`.

## 3. Start the API

Launch the FastAPI backend:

```bash
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```
The API exposes:
- `POST /ingest`: For receiving disk metrics.
- `POST /predict`: For real-time inference.
- `GET /health`: Health check.

## 4. Run the Dashboard

Open the dashboard file in your browser:

```
file:///Users/niteshram/Documents/project/dashboard/index.html
```

The dashboard will automatically connect to `http://127.0.0.1:8000`, simulate disk metrics, and display:
- **Status**: HEALTHY or FAIL.
- **Anomaly Score**: Higher values indicate abnormal behavior.
- **Failure Probability**: Probability of imminent disk failure.
- **Alerts**: A warning banner appears if any disk is critical.

## 5. Verification Results

### API Verification
I ran `test_api.py` which successfully:
- Sent normal data -> Received "HEALTHY" prediction with low anomaly score.
- Sent anomalous data -> Received "FAIL" prediction with high anomaly score.

### Model Performance
- **Autoencoder**: Successfully trained to reconstruct normal data patterns.
- **Failure Predictor**: Achieved high accuracy on synthetic validation data.

## Next Steps
- Connect to real Elasticsearch/Prometheus data sources.
- Deploy models using Docker/Kubernetes.
- Refine thresholds for anomaly detection based on real hardware data.

## 6. Pushing to GitHub

Since the `gh` CLI is not installed, you need to create the repository manually:

1.  **Create Repository**: Go to [github.com/new](https://github.com/new) and create a repository named `storage-anomaly-prediction`. **Do not** initialize with README, .gitignore, or License.
2.  **Push Code**: Run the following commands in your terminal:

```bash
git remote add origin https://github.com/YOUR_USERNAME/storage-anomaly-prediction.git
git branch -M main
git push -u origin main
```

## 7. Real-World Usage (Local System)

To analyze your local system's storage health:

1.  **Start the API** (if not running):
    ```bash
    python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
    ```

2.  **Run the Real Data Collector**:
    This script collects real-time I/O metrics using `psutil` and basic health status from macOS System Profiler.
    ```bash
    ./venv/bin/python3 real_collector.py
    ```

3.  **View Dashboard**:
    Open `dashboard/index.html`. You will see a new card for `local_disk_0` updating with real-time latency and throughput from your Mac.

## 8. Grafana & Prometheus Setup

I have set up a **portable, project-contained** Grafana instance for you.

1.  **Start Prometheus**:
    ```bash
    prometheus --config.file=prometheus.yml
    ```

2.  **Start Grafana (Local Instance)**:
    Use the custom configuration file included in the project:
    ```bash
    grafana server --config project_grafana.ini --homepath /opt/homebrew/opt/grafana/share/grafana
    ```

3.  **View Dashboard**:
    *   Open **[http://localhost:3000/d/storage-anomaly/storage-anomaly-dashboard](http://localhost:3000/d/storage-anomaly/storage-anomaly-dashboard)**.
    *   The dashboard is already imported and configured!
    *   **Login**: No login required (Anonymous Admin enabled for this local instance).

You will see live graphs for Failure Probability, Anomaly Scores, and Throughput.
