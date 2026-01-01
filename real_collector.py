import psutil
import subprocess
import time
import requests
import json
import re

API_URL = "http://127.0.0.1:8000/predict"
DISK_ID = "local_disk_0"

def get_disk_health():
    """
    Uses system_profiler to get basic S.M.A.R.T. status.
    Returns a dict with simulated SMART metrics based on status.
    """
    try:
        result = subprocess.run(['system_profiler', 'SPStorageDataType'], capture_output=True, text=True)
        output = result.stdout
        
        # Check for S.M.A.R.T. Status
        if "S.M.A.R.T. Status: Verified" in output:
            # Healthy
            return {
                "smart_5_reallocated_sector_count": 0,
                "smart_187_reported_uncorrectable_errors": 0,
                "smart_197_current_pending_sector_count": 0,
                "smart_194_temperature_celsius": 40 # Default assumption
            }
        elif "S.M.A.R.T. Status: Failing" in output:
             return {
                "smart_5_reallocated_sector_count": 100,
                "smart_187_reported_uncorrectable_errors": 50,
                "smart_197_current_pending_sector_count": 20,
                "smart_194_temperature_celsius": 60
            }
        else:
            # Unknown, assume healthy
            return {
                "smart_5_reallocated_sector_count": 0,
                "smart_187_reported_uncorrectable_errors": 0,
                "smart_197_current_pending_sector_count": 0,
                "smart_194_temperature_celsius": 40
            }
    except Exception as e:
        print(f"Error getting disk health: {e}")
        return {
            "smart_5_reallocated_sector_count": 0,
            "smart_187_reported_uncorrectable_errors": 0,
            "smart_197_current_pending_sector_count": 0,
            "smart_194_temperature_celsius": 40
        }

def get_disk_io():
    """
    Uses psutil to get disk I/O stats.
    Calculates latency and throughput.
    """
    try:
        # Get initial stats
        io1 = psutil.disk_io_counters()
        time.sleep(1) # Wait 1 second to calculate rate
        io2 = psutil.disk_io_counters()
        
        read_bytes = io2.read_bytes - io1.read_bytes
        write_bytes = io2.write_bytes - io1.write_bytes
        read_count = io2.read_count - io1.read_count
        write_count = io2.write_count - io1.write_count
        read_time = io2.read_time - io1.read_time # ms
        write_time = io2.write_time - io1.write_time # ms
        
        throughput_mbps = (read_bytes + write_bytes) / (1024 * 1024)
        
        read_latency = read_time / read_count if read_count > 0 else 0
        write_latency = write_time / write_count if write_count > 0 else 0
        
        return {
            "read_latency_ms": read_latency,
            "write_latency_ms": write_latency,
            "throughput_mbps": throughput_mbps
        }
    except Exception as e:
        print(f"Error getting disk I/O: {e}")
        return {
            "read_latency_ms": 0,
            "write_latency_ms": 0,
            "throughput_mbps": 0
        }

def collect_and_send():
    print(f"Starting Real Data Collector for {DISK_ID}...")
    while True:
        health_metrics = get_disk_health()
        io_metrics = get_disk_io()
        
        payload = {
            "disk_id": DISK_ID,
            **health_metrics,
            **io_metrics
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            print(f"Sent: {json.dumps(payload)} | Response: {response.json()['prediction']}")
        except Exception as e:
            print(f"Failed to send data: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    collect_and_send()
