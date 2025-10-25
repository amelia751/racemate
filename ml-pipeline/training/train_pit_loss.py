#!/usr/bin/env python3
"""
Train Pit Loss Model
Predicts pit stop time loss including merge penalties
Physics-based + learned traffic merge penalty
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
from google.oauth2 import service_account
from config.settings import settings
from models.pit_loss import PitLossModel
import pickle

class PitLossDataset(Dataset):
    """Dataset for pit stop time loss prediction"""
    
    def __init__(self, df: pd.DataFrame, scaler=None):
        # Feature columns for traffic state
        self.feature_cols = [
            'speed', 'nmot', 'gear', 'pbrake_f', 'pbrake_r',
            'accx_can', 'accy_can', 'Steering_Angle',
            'speed_rolling_mean_5s', 'nmot_rolling_mean_5s',
            'brake_energy', 'lateral_load', 'tire_stress_proxy',
            'steer_rate', 'acc_magnitude', 'throttle_variance'
        ]
        
        # Only use features that exist
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]
        
        print(f"Using {len(self.feature_cols)} features for traffic state")
        
        # Prepare pit scenarios
        self.scenarios, self.targets = self._prepare_data(df)
        
        # Fit or apply scaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.scenarios)
        else:
            self.scaler = scaler
        
        # Scale features
        self.scenarios = self.scaler.transform(self.scenarios)
        
        print(f"Dataset: {len(self.scenarios)} pit scenarios")
        print(f"Target range: [{self.targets.min():.2f}s, {self.targets.max():.2f}s]")
    
    def _prepare_data(self, df):
        """Create pit stop scenarios from telemetry"""
        scenarios = []
        targets = []
        
        # Sample random moments that could be pit stops
        # In production, this would use actual pit stop timing data
        
        # Group by vehicle and lap
        grouped = df.groupby(['vehicle_id', 'lap', 'track'])
        
        print(f"  Processing {len(grouped)} laps for pit scenarios...")
        
        for (vehicle_id, lap, track), lap_data in grouped:
            if len(lap_data) < 50:
                continue
            
            # Sample a few potential pit moments per lap
            n_samples = min(3, len(lap_data) // 100)
            sample_indices = np.random.choice(len(lap_data), n_samples, replace=False)
            
            for idx in sample_indices:
                # Get traffic state at this moment
                try:
                    row = lap_data.iloc[idx]
                    traffic_features = row[self.feature_cols].astype(float).fillna(0).values
                    
                    # Synthetic pit loss target
                    # Base time: ~20-30 seconds
                    base_time = 25.0
                    
                    # Merge penalty depends on:
                    # 1. Speed of traffic (faster = harder to merge)
                    # 2. Density proxy (number of nearby cars)
                    # 3. Track position
                    
                    speed_penalty = 0.0
                    if 'speed' in row and not pd.isna(row['speed']):
                        # Higher traffic speed = more penalty
                        speed_penalty = max(0, (row['speed'] - 100) / 50) * 3.0
                    
                    # Random traffic density (would come from position data in production)
                    traffic_density = np.random.uniform(0, 1)
                    density_penalty = traffic_density * 5.0
                    
                    # Position penalty (some tracks have longer pit lanes)
                    position_penalty = np.random.uniform(0, 2)
                    
                    total_penalty = speed_penalty + density_penalty + position_penalty
                    total_time = base_time + total_penalty
                    
                    # Realistic range: 18-40 seconds
                    total_time = np.clip(total_time, 18, 40)
                    
                    scenarios.append(traffic_features)
                    targets.append(total_time)
                    
                except Exception as e:
                    continue
        
        print(f"  Created {len(scenarios)} pit scenarios")
        
        return np.array(scenarios), np.array(targets)
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.scenarios[idx])
        y = torch.FloatTensor([self.targets[idx]])
        return X, y


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


def train_pit_loss_model():
    """Train the Pit Loss model"""
    
    print("="*60)
    print("PIT LOSS MODEL TRAINING")
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
    
    # Sample for faster training (use 500K rows)
    sample_size = min(len(df_train), 500000)
    df_train_sample = df_train.sample(n=sample_size, random_state=42)
    print(f"Using {len(df_train_sample):,} rows for training")
    
    # Create datasets
    print("\n[2/6] Preparing pit scenarios...")
    train_dataset = PitLossDataset(df_train_sample)
    
    if len(train_dataset) == 0:
        print("✗ No valid scenarios created!")
        return None
    
    # Split for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset_sub, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset_sub, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n[3/6] Initializing Pit Loss Model...")
    input_dim = len(train_dataset.feature_cols)
    model = PitLossModel(input_dim=input_dim, hidden_dim=64)
    
    model = model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: 64")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    print("\n[4/6] Training...")
    num_epochs = 30
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 8
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X)
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        mae = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                
                pred = model(X)
                loss = criterion(pred, y)
                
                val_loss += loss.item()
                mae += torch.abs(pred - y).mean().item()
        
        val_loss /= len(val_loader)
        mae /= len(val_loader)
        rmse = np.sqrt(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val RMSE:   {rmse:.4f} seconds")
        print(f"  Val MAE:    {mae:.4f} seconds")
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
                'val_rmse': rmse,
                'val_mae': mae,
                'scaler': train_dataset.scaler,
                'feature_cols': train_dataset.feature_cols,
                'input_dim': input_dim
            }
            
            torch.save(checkpoint, '/tmp/pit_loss_best.pth')
            print(f"  ✓ Saved best model (RMSE: {rmse:.4f}s, MAE: {mae:.4f}s)")
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
    
    blob = bucket.blob('pit_loss/model.pth')
    blob.upload_from_filename('/tmp/pit_loss_best.pth')
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/pit_loss/model.pth")
    
    # Save metrics
    best_rmse = np.sqrt(best_val_loss)
    metrics = {
        'best_val_loss': float(best_val_loss),
        'best_val_rmse': float(best_rmse),
        'best_val_mae': float(mae),
        'num_epochs': epoch + 1,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'input_dim': input_dim,
        'feature_cols': train_dataset.feature_cols
    }
    
    metrics_path = "/tmp/pit_loss_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('pit_loss/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    print("✓ Metrics saved")
    
    print("\n[6/6] Evaluation...")
    print(f"\nFinal Metrics:")
    print(f"  Best Val RMSE: {best_rmse:.4f} seconds")
    print(f"  Best Val MAE:  {mae:.4f} seconds")
    print(f"  Model Type: Physics-based + learned merge penalty")
    print(f"  Output: Pit stop time loss (seconds)")
    print(f"  Base pit time: ~25s, Merge penalty: 0-15s")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = train_pit_loss_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

