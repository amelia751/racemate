#!/usr/bin/env python3
"""
Test the trained fuel consumption model
Load from GCS and make predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from google.cloud import storage
from google.oauth2 import service_account
from config.settings import settings
import pickle
import pandas as pd
import numpy as np

def test_fuel_model():
    """Load and test the fuel model from GCS"""
    
    print("="*60)
    print("FUEL MODEL VALIDATION TEST")
    print("="*60)
    
    # Download model from GCS
    print("\n[1/3] Loading model from GCS...")
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    
    bucket = client.bucket(settings.gcs_bucket_models)
    blob = bucket.blob('fuel_consumption/model.pkl')
    
    local_path = "/tmp/fuel_model_test.pkl"
    blob.download_to_filename(local_path)
    
    with open(local_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    print(f"  Features: {model.feature_names}")
    
    # Create test data
    print("\n[2/3] Creating test data...")
    test_data = pd.DataFrame({
        'nmot': [4000, 5000, 6000, 4500, 5500],
        'gear': [3, 4, 5, 3, 4],
        'speed': [100, 120, 140, 110, 130],
        'on_full_throttle': [50, 80, 100, 60, 90],
        'lap': [1, 5, 10, 3, 7]
    })
    
    print("Test scenarios:")
    print(test_data.to_string(index=False))
    
    # Make predictions
    print("\n[3/3] Making predictions...")
    predictions = model.predict(test_data)
    
    print("\nPredicted fuel consumption (L/lap):")
    for i, pred in enumerate(predictions):
        print(f"  Scenario {i+1}: {pred:.3f} L/lap")
    
    # Validate predictions are reasonable
    assert all(0.5 <= p <= 2.5 for p in predictions), "Predictions out of expected range!"
    assert predictions.std() > 0, "Predictions have no variance!"
    
    print("\n✓ Model validation successful!")
    print(f"  Mean prediction: {predictions.mean():.3f} L/lap")
    print(f"  Std deviation: {predictions.std():.3f} L/lap")
    print(f"  Range: [{predictions.min():.3f}, {predictions.max():.3f}] L/lap")
    
    # Load metrics
    print("\n[Bonus] Loading training metrics...")
    blob = bucket.blob('fuel_consumption/metrics.pkl')
    local_metrics_path = "/tmp/fuel_metrics_test.pkl"
    blob.download_to_filename(local_metrics_path)
    
    with open(local_metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    print("\nModel Performance (from training):")
    print(f"  Val MAE: {metrics['val_mae']:.4f}")
    print(f"  Val R²:  {metrics['val_r2']:.4f}")
    
    print("\nTop Feature Importances:")
    for feat, imp in sorted(metrics['feature_importances'].items(), 
                           key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {feat}: {imp:.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_fuel_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

