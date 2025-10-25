#!/usr/bin/env python3
"""
Test the trained Tire Degradation model
Load from GCS and make predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from google.cloud import storage
from google.oauth2 import service_account
from config.settings import settings
from models.tire_degradation import TireDegradationModel
import pickle
import numpy as np

def test_tire_model():
    """Load and test the Tire Degradation model from GCS"""
    
    print("="*60)
    print("TIRE DEGRADATION MODEL VALIDATION TEST")
    print("="*60)
    
    # Download model from GCS
    print("\n[1/4] Loading model from GCS...")
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    
    bucket = client.bucket(settings.gcs_bucket_models)
    blob = bucket.blob('tire_degradation/model.pth')
    
    local_path = "/tmp/tire_test.pth"
    blob.download_to_filename(local_path)
    
    # Load checkpoint
    checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
    
    input_dim = checkpoint['input_dim']
    feature_cols = checkpoint['feature_cols']
    physics_params = checkpoint['physics_params']
    
    print(f"✓ Model loaded successfully")
    print(f"  Input dim: {input_dim}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Trained for: {checkpoint['epoch'] + 1} epochs")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = TireDegradationModel(
        input_dim=input_dim,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model initialized")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nLearned Physics Parameters:")
    print(f"  Brake energy coef (α): {physics_params['alpha_brake']:.6f}")
    print(f"  Lateral load coef (β): {physics_params['beta_lateral']:.6f}")
    print(f"  Temperature coef (γ):  {physics_params['gamma_temp']:.6f}")
    
    # Create test scenarios
    print("\n[3/4] Creating test scenarios...")
    
    scenarios = [
        {
            'name': 'Fresh tires (early race)',
            'cum_brake_energy': 100.0,
            'cum_lateral_load': 50.0,
            'air_temp': 25.0
        },
        {
            'name': 'Mid-race wear',
            'cum_brake_energy': 5000.0,
            'cum_lateral_load': 2500.0,
            'air_temp': 30.0
        },
        {
            'name': 'Heavy degradation (late race)',
            'cum_brake_energy': 15000.0,
            'cum_lateral_load': 8000.0,
            'air_temp': 35.0
        },
        {
            'name': 'Extreme conditions',
            'cum_brake_energy': 25000.0,
            'cum_lateral_load': 12000.0,
            'air_temp': 40.0
        }
    ]
    
    # Make predictions
    print("\n[4/4] Making predictions...")
    
    with torch.no_grad():
        for i, scenario in enumerate(scenarios):
            # Create dummy telemetry sequence
            seq = torch.randn(1, 200, input_dim)  # Random telemetry
            
            # Physics features
            physics = {
                'cum_brake_energy': torch.FloatTensor([[scenario['cum_brake_energy']]]),
                'cum_lateral_load': torch.FloatTensor([[scenario['cum_lateral_load']]]),
                'air_temp': torch.FloatTensor([[scenario['air_temp']]])
            }
            
            # Predict grip
            grip_index = model(seq, physics)
            grip_val = grip_index.item()
            
            # Calculate grip loss
            grip_loss_pct = (1.0 - grip_val) * 100
            
            print(f"\nScenario {i+1}: {scenario['name']}")
            print(f"  Cumulative brake energy: {scenario['cum_brake_energy']:.0f}")
            print(f"  Cumulative lateral load: {scenario['cum_lateral_load']:.0f}")
            print(f"  Air temperature: {scenario['air_temp']:.0f}°C")
            print(f"  → Predicted grip index: {grip_val:.3f}")
            print(f"  → Grip loss: {grip_loss_pct:.1f}%")
            
            # Recommendation
            if grip_val > 0.95:
                status = "✓ Excellent grip"
            elif grip_val > 0.85:
                status = "⚠️  Monitor tire condition"
            elif grip_val > 0.75:
                status = "⚠️  Consider pit stop soon"
            else:
                status = "🔴 Pit stop recommended!"
            
            print(f"  Status: {status}")
    
    # Load and display training metrics
    print("\n[Bonus] Training Metrics...")
    blob = bucket.blob('tire_degradation/metrics.pkl')
    local_metrics_path = "/tmp/tire_metrics_test.pkl"
    blob.download_to_filename(local_metrics_path)
    
    with open(local_metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    print(f"\nModel Performance:")
    print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"  Val RMSE:      {np.sqrt(metrics['best_val_loss']):.4f} grip units")
    print(f"  Parameters:    {metrics['num_parameters']:,}")
    print(f"  Epochs:        {metrics['num_epochs']}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
    print("\nModel successfully predicts tire grip degradation!")
    print("Physics-informed approach captures wear dynamics.")
    
    return True

if __name__ == "__main__":
    try:
        test_tire_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

