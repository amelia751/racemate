from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from typing import Dict, Any
import pickle

class FuelConsumptionModel:
    """Gradient Boosting model for fuel burn prediction"""
    
    def __init__(self, params: Dict[str, Any] = None):
        
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'random_state': 42,
            'verbose': 1
        }
        
        if params:
            default_params.update(params)
        
        self.model = GradientBoostingRegressor(**default_params)
        self.feature_names = None
    
    def prepare_features(self, df):
        """Extract features for fuel prediction"""
        
        features = [
            'nmot',  # RPM
            'aps',   # Throttle
            'gear',
            'speed',
            'throttle_variance',
            'on_full_throttle',
            'lap'
        ]
        
        self.feature_names = [f for f in features if f in df.columns]
        
        return df[self.feature_names]
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Gradient Boosting model"""
        
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X):
        """Predict fuel consumption"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importances"""
        return dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
    
    def save(self, path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Load model"""
        with open(path, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    import pandas as pd
    print("Testing Fuel Consumption Model...")
    
    # Create dummy data
    n_samples = 1000
    X_train = pd.DataFrame({
        'nmot': np.random.uniform(2000, 7000, n_samples),
        'aps': np.random.uniform(0, 100, n_samples),
        'gear': np.random.randint(1, 7, n_samples),
        'speed': np.random.uniform(50, 200, n_samples),
        'throttle_variance': np.random.uniform(0, 50, n_samples),
        'on_full_throttle': np.random.randint(0, 2, n_samples),
        'lap': np.random.randint(1, 30, n_samples)
    })
    
    # Simulated fuel consumption (L/lap)
    y_train = 1.2 + 0.0001 * X_train['nmot'] + 0.005 * X_train['aps'] + np.random.normal(0, 0.1, n_samples)
    
    model = FuelConsumptionModel()
    model.feature_names = list(X_train.columns)
    model.train(X_train, y_train)
    
    # Test prediction
    predictions = model.predict(X_train[:10])
    
    print(f"✓ Model trained")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {model.feature_names}")
    print(f"  Sample predictions: {predictions[:5]}")
    print(f"  Feature importance: {model.get_feature_importance()}")

