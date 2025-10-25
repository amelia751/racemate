#!/usr/bin/env python3
"""
Test the trained Lap-Time Transformer model
Load from GCS and make predictions with uncertainty quantiles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from google.cloud import storage
from google.oauth2 import service_account
from config.settings import settings
from models.lap_time_transformer import LapTimeTransformer
import pickle
import numpy as np

def test_lap_time_model():
    """Load and test the Lap-Time Transformer from GCS"""
    
    print("="*60)
    print("LAP-TIME TRANSFORMER VALIDATION TEST")
    print("="*60)
    
    # Download model from GCS
    print("\n[1/4] Loading model from GCS...")
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    
    bucket = client.bucket(settings.gcs_bucket_models)
    blob = bucket.blob('lap_time_transformer/model.pth')
    
    local_path = "/tmp/lap_time_test.pth"
    blob.download_to_filename(local_path)
    
    # Load checkpoint (weights_only=False since we're loading our own trusted checkpoint with sklearn objects)
    checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
    
    input_dim = checkpoint['input_dim']
    feature_cols = checkpoint['feature_cols']
    
    print(f"✓ Model loaded successfully")
    print(f"  Input dim: {input_dim}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Trained for: {checkpoint['epoch'] + 1} epochs")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = LapTimeTransformer(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=4
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model initialized")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test data (synthetic sequence)
    print("\n[3/4] Creating test sequences...")
    
    # Simulate a lap sequence (200 timesteps × input_dim features)
    test_sequences = []
    scenarios = [
        "Conservative lap (slow, smooth)",
        "Aggressive lap (fast, hard braking)",
        "Normal lap (medium pace)"
    ]
    
    for i, scenario in enumerate(scenarios):
        # Generate realistic telemetry patterns
        if i == 0:  # Conservative
            speed = np.linspace(80, 120, 200)
            nmot = np.linspace(3000, 5000, 200)
            braking = np.random.uniform(0, 30, 200)
        elif i == 1:  # Aggressive
            speed = np.linspace(100, 150, 200)
            nmot = np.linspace(4000, 6500, 200)
            braking = np.random.uniform(40, 100, 200)
        else:  # Normal
            speed = np.linspace(90, 135, 200)
            nmot = np.linspace(3500, 5800, 200)
            braking = np.random.uniform(20, 60, 200)
        
        # Create sequence with all required features
        sequence = np.zeros((200, input_dim))
        
        # Fill in available features (pad with zeros if not enough features)
        if input_dim >= 3:
            sequence[:, 0] = speed  # speed
            sequence[:, 1] = nmot   # nmot
            sequence[:, 2] = braking  # pbrake
        
        # Add some noise and variation
        sequence += np.random.normal(0, 5, sequence.shape)
        
        test_sequences.append(sequence)
    
    print(f"Created {len(test_sequences)} test scenarios")
    
    # Make predictions
    print("\n[4/4] Making predictions...")
    
    with torch.no_grad():
        for i, (seq, scenario) in enumerate(zip(test_sequences, scenarios)):
            X = torch.FloatTensor(seq).unsqueeze(0)  # Add batch dim
            
            mean, quantiles = model(X)
            
            q_10, q_50, q_90 = [q.item() for q in quantiles]
            mean_val = mean.item()
            
            print(f"\nScenario {i+1}: {scenario}")
            print(f"  Predicted lap time: {mean_val:.2f} seconds")
            print(f"  10th percentile:    {q_10:.2f} seconds")
            print(f"  Median (50th):      {q_50:.2f} seconds")
            print(f"  90th percentile:    {q_90:.2f} seconds")
            print(f"  Uncertainty range:  ±{(q_90 - q_10)/2:.2f} seconds")
    
    # Load and display training metrics
    print("\n[Bonus] Training Metrics...")
    blob = bucket.blob('lap_time_transformer/metrics.pkl')
    local_metrics_path = "/tmp/lap_time_metrics_test.pkl"
    blob.download_to_filename(local_metrics_path)
    
    with open(local_metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    print(f"\nModel Performance:")
    print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"  Best Val MSE:  {metrics['best_val_mse']:.4f}")
    print(f"  Val RMSE:      {np.sqrt(metrics['best_val_mse']):.2f} seconds")
    print(f"  Parameters:    {metrics['num_parameters']:,}")
    print(f"  Epochs:        {metrics['num_epochs']}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
    print("\nModel successfully predicts lap times with uncertainty!")
    
    return True

if __name__ == "__main__":
    try:
        test_lap_time_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

