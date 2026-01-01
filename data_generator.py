import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

class DiskDataGenerator:
    def __init__(self, num_disks=5):
        self.num_disks = num_disks
        self.disk_ids = [f"disk_{i}" for i in range(num_disks)]
        
    def generate_normal_data(self, num_samples=100):
        data = []
        for _ in range(num_samples):
            for disk_id in self.disk_ids:
                record = {
                    "timestamp": datetime.now(),
                    "disk_id": disk_id,
                    "smart_5_reallocated_sector_count": int(np.random.normal(0, 1)), # Mostly 0
                    "smart_187_reported_uncorrectable_errors": int(np.random.normal(0, 0.5)),
                    "smart_194_temperature_celsius": int(np.random.normal(35, 5)),
                    "smart_197_current_pending_sector_count": int(np.random.normal(0, 0.5)),
                    "read_latency_ms": np.random.normal(5, 2),
                    "write_latency_ms": np.random.normal(10, 3),
                    "throughput_mbps": np.random.normal(100, 20),
                    "label": 0 # Normal
                }
                # Clip negative values
                for k, v in record.items():
                    if isinstance(v, (int, float)) and k != "label":
                        record[k] = max(0, v)
                data.append(record)
        return pd.DataFrame(data)

    def generate_anomalous_data(self, num_samples=20):
        data = []
        for _ in range(num_samples):
            disk_id = random.choice(self.disk_ids)
            record = {
                "timestamp": datetime.now(),
                "disk_id": disk_id,
                "smart_5_reallocated_sector_count": int(np.random.normal(5, 2)),
                "smart_187_reported_uncorrectable_errors": int(np.random.normal(2, 1)),
                "smart_194_temperature_celsius": int(np.random.normal(55, 5)), # Overheating
                "smart_197_current_pending_sector_count": int(np.random.normal(2, 1)),
                "read_latency_ms": np.random.normal(50, 10), # High latency
                "write_latency_ms": np.random.normal(80, 15),
                "throughput_mbps": np.random.normal(20, 5), # Low throughput
                "label": 1 # Anomaly/Failure
            }
             # Clip negative values
            for k, v in record.items():
                if isinstance(v, (int, float)) and k != "label":
                    record[k] = max(0, v)
            data.append(record)
        return pd.DataFrame(data)

    def get_stream_data(self):
        # Simulate a single point in time for all disks
        data = []
        for disk_id in self.disk_ids:
            # 5% chance of anomaly
            is_anomaly = random.random() < 0.05
            
            if is_anomaly:
                 record = {
                    "timestamp": datetime.now().isoformat(),
                    "disk_id": disk_id,
                    "smart_5_reallocated_sector_count": int(np.random.normal(5, 2)),
                    "smart_187_reported_uncorrectable_errors": int(np.random.normal(2, 1)),
                    "smart_194_temperature_celsius": int(np.random.normal(55, 5)),
                    "smart_197_current_pending_sector_count": int(np.random.normal(2, 1)),
                    "read_latency_ms": np.random.normal(50, 10),
                    "write_latency_ms": np.random.normal(80, 15),
                    "throughput_mbps": np.random.normal(20, 5),
                }
            else:
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "disk_id": disk_id,
                    "smart_5_reallocated_sector_count": int(np.random.normal(0, 1)),
                    "smart_187_reported_uncorrectable_errors": int(np.random.normal(0, 0.5)),
                    "smart_194_temperature_celsius": int(np.random.normal(35, 5)),
                    "smart_197_current_pending_sector_count": int(np.random.normal(0, 0.5)),
                    "read_latency_ms": np.random.normal(5, 2),
                    "write_latency_ms": np.random.normal(10, 3),
                    "throughput_mbps": np.random.normal(100, 20),
                }
            
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    record[k] = max(0, v)
            data.append(record)
        return data

if __name__ == "__main__":
    gen = DiskDataGenerator()
    df_normal = gen.generate_normal_data(10)
    print("Normal Data Sample:")
    print(df_normal.head())
    
    df_anom = gen.generate_anomalous_data(5)
    print("\nAnomalous Data Sample:")
    print(df_anom.head())
