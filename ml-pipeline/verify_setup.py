#!/usr/bin/env python3
"""
Verification script to test all components
Run this to verify the ML pipeline is working correctly
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("\n[1/6] Testing Python Imports...")
    try:
        import torch
        import pandas as pd
        import numpy as np
        from google.cloud import storage, aiplatform
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n[2/6] Testing Configuration...")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from config.settings import settings
        print(f"  ✓ GCP Project: {settings.gcp_project_id}")
        print(f"  ✓ Region: {settings.gcp_region}")
        print(f"  ✓ Credentials path: {settings.gcp_service_account_path}")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False

def test_gcp_auth():
    """Test GCP authentication"""
    print("\n[3/6] Testing GCP Authentication...")
    try:
        from gcp_setup.create_buckets import get_storage_client
        client = get_storage_client()
        buckets = list(client.list_buckets())
        print(f"  ✓ Authenticated successfully")
        print(f"  ✓ Found {len(buckets)} buckets")
        return True
    except Exception as e:
        print(f"  ✗ Auth failed: {e}")
        return False

def test_data_processing():
    """Test data processing modules"""
    print("\n[4/6] Testing Data Processing...")
    try:
        from data_processing.csv_parser import TelemetryParser
        from data_processing.feature_engineering import FeatureEngineer
        parser = TelemetryParser()
        engineer = FeatureEngineer()
        print("  ✓ CSV parser initialized")
        print("  ✓ Feature engineer initialized")
        return True
    except Exception as e:
        print(f"  ✗ Data processing failed: {e}")
        return False

def test_models():
    """Test all ML models"""
    print("\n[5/6] Testing ML Models...")
    try:
        import torch
        from models.lap_time_transformer import LapTimeTransformer
        from models.tire_degradation import TireDegradationModel
        from models.fuel_consumption import FuelConsumptionModel
        from models.fcy_hazard import FCYHazardModel
        from models.pit_loss import PitLossModel
        from models.anomaly_detector import AnomalyDetector
        from models.driver_embedding import DriverEmbedding
        
        # Test instantiation
        models = {
            "Lap-Time Transformer": LapTimeTransformer(),
            "Tire Degradation": TireDegradationModel(),
            "Fuel Consumption": FuelConsumptionModel(),
            "FCY Hazard": FCYHazardModel(),
            "Pit Loss": PitLossModel(),
            "Anomaly Detector": AnomalyDetector(),
            "Driver Embedding": DriverEmbedding()
        }
        
        for name, model in models.items():
            if hasattr(model, 'parameters'):
                params = sum(p.numel() for p in model.parameters())
                print(f"  ✓ {name}: {params:,} parameters")
            else:
                print(f"  ✓ {name}: initialized")
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_inference():
    """Test model forward pass"""
    print("\n[6/6] Testing Model Inference...")
    try:
        import torch
        from models.lap_time_transformer import LapTimeTransformer
        
        model = LapTimeTransformer()
        x = torch.randn(2, 200, 16)
        mean, quantiles = model(x)
        
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Output shape: {mean.shape}")
        print(f"  ✓ Quantiles: {len(quantiles)}")
        print("  ✓ Inference successful")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("="*60)
    print("COGNIRACE ML PIPELINE VERIFICATION")
    print("="*60)
    
    tests = [
        test_imports,
        test_config,
        test_gcp_auth,
        test_data_processing,
        test_models,
        test_model_inference
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nYour ML pipeline is ready!")
        print("\nNext steps:")
        print("1. Review /Users/anhlam/hack-the-track/TODO.md for user actions")
        print("2. Enable Vertex AI API in GCP Console")
        print("3. Run: python data_processing/run_pipeline.py")
        return 0
    else:
        print(f"\n✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nPlease check the errors above and:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check GCP credentials are correct")
        print("3. Verify service account has necessary permissions")
        return 1

if __name__ == "__main__":
    sys.exit(main())

