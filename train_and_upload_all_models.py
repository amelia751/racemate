#!/usr/bin/env python3
"""
Train all 8 models and upload to GCS immediately
"""
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, '/Users/anhlam/hack-the-track/ml-pipeline')
os.chdir('/Users/anhlam/hack-the-track/ml-pipeline')

# Set GCP credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/anhlam/hack-the-track/cognirace-ea0604b3d4db.json'

print("=" * 80)
print("🏋️  TRAINING ALL 8 MODELS AND UPLOADING TO GCS")
print("=" * 80)
print()

# Import training scripts
from train_fuel_consumption import main as train_fuel
from train_lap_time_transformer import main as train_laptime
from train_tire_degradation import main as train_tire
from train_fcy_hazard import main as train_fcy
from train_pit_loss import main as train_pit
from train_anomaly_detector import main as train_anomaly
from train_driver_embedding import main as train_driver
from train_traffic_gnn import main as train_traffic

from google.cloud import storage

def upload_model_to_gcs(model_name: str):
    """Upload trained model to GCS"""
    print(f"\n📤 Uploading {model_name} to GCS...")
    
    try:
        client = storage.Client()
        bucket = client.bucket('cognirace-models')
        
        model_dir = Path(f'models/{model_name}')
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return False
        
        # Upload all files in model directory
        for file_path in model_dir.glob('*'):
            if file_path.is_file():
                blob_name = f"{model_name}/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                print(f"   ✅ Uploaded {file_path.name}")
        
        print(f"✅ {model_name} uploaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload {model_name}: {e}")
        return False

# Train and upload each model
models = [
    ("fuel_consumption", train_fuel),
    ("lap_time_transformer", train_laptime),
    ("tire_degradation", train_tire),
    ("fcy_hazard", train_fcy),
    ("pit_loss", train_pit),
    ("anomaly_detector", train_anomaly),
    ("driver_embedding", train_driver),
    ("traffic_gnn", train_traffic)
]

results = {}

for model_name, train_func in models:
    print(f"\n{'=' * 80}")
    print(f"🏋️  TRAINING: {model_name.upper()}")
    print(f"{'=' * 80}")
    
    try:
        train_func()
        print(f"✅ {model_name} trained successfully")
        
        # Upload to GCS
        upload_success = upload_model_to_gcs(model_name)
        results[model_name] = "✅ Trained & Uploaded" if upload_success else "⚠️  Trained, Upload Failed"
        
    except Exception as e:
        print(f"❌ {model_name} training failed: {e}")
        results[model_name] = f"❌ Failed: {str(e)[:50]}"

print(f"\n{'=' * 80}")
print("📊 FINAL RESULTS")
print(f"{'=' * 80}")

for model_name, status in results.items():
    print(f"{status} {model_name}")

print()
print("🎯 All models processed!")
