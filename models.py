import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4), # Latent space
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FailurePredictor:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        if self.model_type == "xgboost":
             # XGBoost expects DMatrix or numpy array, but fit handles numpy/pandas
             return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        if self.model_type == "xgboost":
            self.model.save_model(path)
        else:
            import joblib
            joblib.dump(self.model, path)
            
    def load(self, path):
        if self.model_type == "xgboost":
            self.model.load_model(path)
        else:
            import joblib
            self.model = joblib.load(path)
