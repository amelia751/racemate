#!/usr/bin/env python3
"""
Train Tire Degradation Model (Physics-Informed)
Physics base + TCN residual for grip prediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
from google.oauth2 import service_account
from config.settings import settings
from models.tire_degradation import TireDegradationModel
import pickle

class TireDataset(Dataset):
    """Dataset for tire degradation prediction"""
    
    def __init__(self, df: pd.DataFrame, seq_len=200, scaler=None):
        self.seq_len = seq_len
        
        # Feature columns for sequence
        self.feature_cols = [
            'speed', 'nmot', 'gear', 'pbrake_f', 'pbrake_r',
            'accx_can', 'accy_can', 'Steering_Angle',
            'speed_rolling_mean_5s', 'nmot_rolling_mean_5s',
            'brake_energy', 'lateral_load', 'tire_stress_proxy',
            'steer_rate', 'micro_sector_id', 'acc_magnitude'
        ]
        
        # Only use features that exist
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]
        
        print(f"Using {len(self.feature_cols)} features for sequences")
        
        # Physics features for grip model
        self.physics_features = ['cum_brake_energy', 'cum_lateral_load']
        
        # Prepare sequences and targets
        self.sequences, self.physics_vals, self.targets = self._prepare_data(df)
        
        # Fit or apply scaler
        if scaler is None:
            self.scaler = StandardScaler()
            # Flatten all sequences to fit scaler
            all_data = np.vstack(self.sequences)
            self.scaler.fit(all_data)
        else:
            self.scaler = scaler
        
        # Scale sequences
        self.sequences = [self.scaler.transform(seq) for seq in self.sequences]
        
        print(f"Dataset: {len(self.sequences)} sequences")
        if len(self.targets) > 0:
            print(f"Target range: [{np.min(self.targets):.3f}, {np.max(self.targets):.3f}]")
    
    def _prepare_data(self, df):
        """Extract sequences, physics features, and targets"""
        sequences = []
        physics_vals = []
        targets = []
        
        # Group by vehicle and lap
        grouped = df.groupby(['vehicle_id', 'lap', 'track'])
        
        print(f"  Processing {len(grouped)} laps...")
        
        for (vehicle_id, lap, track), lap_data in grouped:
            if len(lap_data) < self.seq_len:
                continue
            
            # Sort by timestamp
            lap_data = lap_data.sort_values('timestamp')
            
            # Extract sequence features
            try:
                seq = lap_data[self.feature_cols].fillna(0).values[:self.seq_len]
                
                # Extract physics features (use final values from lap)
                physics_dict = {}
                for feat in self.physics_features:
                    if feat in lap_data.columns:
                        val = lap_data[feat].iloc[-1] if not lap_data[feat].isna().all() else 0.0
                        physics_dict[feat] = val
                    else:
                        physics_dict[feat] = 0.0
                
                # Target: Synthetic grip index (since we don't have actual tire data)
                # In production, this would come from tire sensors
                # For now: grip degrades with brake energy and lateral load
                grip_base = 1.0
                grip_loss = (
                    0.00001 * physics_dict.get('cum_brake_energy', 0) +
                    0.00001 * physics_dict.get('cum_lateral_load', 0) +
                    np.random.normal(0, 0.02)  # Noise
                )
                grip_index = np.clip(grip_base - grip_loss, 0.7, 1.0)
                
                sequences.append(seq)
                physics_vals.append(physics_dict)
                targets.append(grip_index)
                
            except Exception as e:
                continue
        
        print(f"  Created {len(sequences)} valid sequences")
        
        return sequences, physics_vals, np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.sequences[idx])
        physics = self.physics_vals[idx]
        y = torch.FloatTensor([self.targets[idx]])
        
        # Convert physics dict to tensors
        physics_tensors = {
            'cum_brake_energy': torch.FloatTensor([physics.get('cum_brake_energy', 0.0)]),
            'cum_lateral_load': torch.FloatTensor([physics.get('cum_lateral_load', 0.0)]),
            'air_temp': torch.FloatTensor([25.0])  # Default temp, would come from weather data
        }
        
        return X, physics_tensors, y


def load_data_from_gcs(bucket_name: str, blob_path: str) -> pd.DataFrame:
    """Load parquet from GCS"""
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    local_path = f"/tmp/{blob_path.split('/')[-1]}"
    blob.download_to_filename(local_path)
    
    df = pd.read_parquet(local_path)
    print(f"Loaded {len(df)} rows from gs://{bucket_name}/{blob_path}")
    
    return df


def train_tire_model():
    """Train the Tire Degradation model"""
    
    print("="*60)
    print("TIRE DEGRADATION MODEL TRAINING")
    print("="*60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[1/6] Loading data from GCS...")
    df_train = load_data_from_gcs(
        settings.gcs_bucket_processed,
        'splits/train.parquet'
    )
    
    # Sample for faster training (use 1M rows)
    sample_size = min(len(df_train), 1000000)
    df_train_sample = df_train.sample(n=sample_size, random_state=42)
    print(f"Using {len(df_train_sample):,} rows for training")
    
    # Create datasets
    print("\n[2/6] Preparing sequences...")
    train_dataset = TireDataset(df_train_sample, seq_len=200)
    
    if len(train_dataset) == 0:
        print("✗ No valid sequences created!")
        return None
    
    # Split for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset_sub, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset_sub, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n[3/6] Initializing Tire Degradation Model...")
    input_dim = len(train_dataset.feature_cols)
    model = TireDegradationModel(
        input_dim=input_dim,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3
    )
    
    model = model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training loop
    print("\n[4/6] Training...")
    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, (X, physics, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # Move physics features to device
            physics_device = {k: v.to(device) for k, v in physics.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            grip_pred = model(X, physics_device)
            
            # Loss
            loss = criterion(grip_pred, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X, physics, y in val_loader:
                X, y = X.to(device), y.to(device)
                physics_device = {k: v.to(device) for k, v in physics.items()}
                
                grip_pred = model(X, physics_device)
                loss = criterion(grip_pred, y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save locally
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'scaler': train_dataset.dataset.scaler if hasattr(train_dataset, 'dataset') else train_dataset.scaler,
                'feature_cols': train_dataset.dataset.feature_cols if hasattr(train_dataset, 'dataset') else train_dataset.feature_cols,
                'input_dim': input_dim,
                'physics_params': {
                    'alpha_brake': model.alpha_brake.item(),
                    'beta_lateral': model.beta_lateral.item(),
                    'gamma_temp': model.gamma_temp.item()
                }
            }
            
            torch.save(checkpoint, '/tmp/tire_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            print(f"    Physics params: α={model.alpha_brake.item():.6f}, β={model.beta_lateral.item():.6f}, γ={model.gamma_temp.item():.6f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n  Early stopping after {epoch+1} epochs")
                break
    
    # Upload to GCS
    print("\n[5/6] Uploading model to GCS...")
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    bucket = client.bucket(settings.gcs_bucket_models)
    
    blob = bucket.blob('tire_degradation/model.pth')
    blob.upload_from_filename('/tmp/tire_best.pth')
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/tire_degradation/model.pth")
    
    # Save metrics
    metrics = {
        'best_val_loss': float(best_val_loss),
        'num_epochs': epoch + 1,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'input_dim': input_dim,
        'feature_cols': train_dataset.dataset.feature_cols if hasattr(train_dataset, 'dataset') else train_dataset.feature_cols,
        'physics_params': checkpoint['physics_params']
    }
    
    metrics_path = "/tmp/tire_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('tire_degradation/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    print("✓ Metrics saved")
    
    print("\n[6/6] Evaluation...")
    print(f"\nFinal Metrics:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  RMSE: {np.sqrt(best_val_loss):.4f} grip units")
    print(f"\nLearned Physics Parameters:")
    print(f"  Brake energy coefficient (α): {checkpoint['physics_params']['alpha_brake']:.6f}")
    print(f"  Lateral load coefficient (β): {checkpoint['physics_params']['beta_lateral']:.6f}")
    print(f"  Temperature coefficient (γ):  {checkpoint['physics_params']['gamma_temp']:.6f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = train_tire_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

