import requests
import json
import time
import random

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # Wait for API to start
    print("Waiting for API to start...")
    for _ in range(10):
        try:
            requests.get(f"{base_url}/health")
            print("API is up!")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        print("API failed to start.")
        return

    # Test Prediction Endpoint
    print("\nTesting Prediction Endpoint...")
    
    # Normal Data
    normal_payload = {
        "disk_id": "disk_test_1",
        "smart_5_reallocated_sector_count": 0,
        "smart_187_reported_uncorrectable_errors": 0,
        "smart_194_temperature_celsius": 35,
        "smart_197_current_pending_sector_count": 0,
        "read_latency_ms": 5.0,
        "write_latency_ms": 10.0,
        "throughput_mbps": 100.0
    }
    
    response = requests.post(f"{base_url}/predict", json=normal_payload)
    print(f"Normal Data Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Anomalous Data
    anom_payload = {
        "disk_id": "disk_test_2",
        "smart_5_reallocated_sector_count": 10,
        "smart_187_reported_uncorrectable_errors": 5,
        "smart_194_temperature_celsius": 60,
        "smart_197_current_pending_sector_count": 5,
        "read_latency_ms": 100.0,
        "write_latency_ms": 150.0,
        "throughput_mbps": 10.0
    }
    
    response = requests.post(f"{base_url}/predict", json=anom_payload)
    print(f"\nAnomalous Data Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()
