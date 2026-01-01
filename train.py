import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_generator import DiskDataGenerator
from models import Autoencoder, FailurePredictor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_models():
    os.makedirs("models", exist_ok=True)
    gen = DiskDataGenerator(num_disks=50)
    
    print("Generating training data...")
    # Generate normal data for Autoencoder
    df_normal = gen.generate_normal_data(num_samples=2000)
    
    # Generate mixed data for Failure Predictor
    df_anom = gen.generate_anomalous_data(num_samples=500)
    df_mixed = pd.concat([df_normal, df_anom]).sample(frac=1).reset_index(drop=True)
    
    feature_cols = [
        "smart_5_reallocated_sector_count",
        "smart_187_reported_uncorrectable_errors",
        "smart_194_temperature_celsius",
        "smart_197_current_pending_sector_count",
        "read_latency_ms",
        "write_latency_ms",
        "throughput_mbps"
    ]
    
    # --- Train Autoencoder ---
    print("Training Autoencoder...")
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(df_normal[feature_cols])
    
    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    
    input_dim = len(feature_cols)
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_normal)
    
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = autoencoder(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    torch.save(autoencoder.state_dict(), "models/autoencoder.pth")
    print("Autoencoder saved.")
    
    # --- Train Failure Predictor ---
    print("Training Failure Predictor...")
    X = df_mixed[feature_cols]
    y = df_mixed["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    predictor = FailurePredictor(model_type="xgboost")
    predictor.train(X_train, y_train)
    
    # Evaluate
    probs = predictor.predict_proba(X_test)
    preds = (probs > 0.5).astype(int)
    acc = (preds == y_test).mean()
    print(f"Failure Predictor Accuracy: {acc:.4f}")
    
    predictor.save("models/failure_predictor.json")
    print("Failure Predictor saved.")

if __name__ == "__main__":
    train_models()
