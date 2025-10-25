#!/usr/bin/env python3
"""
Train Lap-Time Transformer Model
Core predictor for next lap time with uncertainty quantiles
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
from models.lap_time_transformer import LapTimeTransformer, QuantileLoss
import pickle

class LapSequenceDataset(Dataset):
    """Dataset for lap time prediction with sequences"""
    
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
        
        # Prepare sequences and targets
        self.sequences, self.targets = self._prepare_data(df)
        
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
        print(f"Target range: [{np.min(self.targets):.2f}, {np.max(self.targets):.2f}]")
    
    def _prepare_data(self, df):
        """Extract sequences and compute lap time targets"""
        sequences = []
        targets = []
        
        # Compute lap times first (aggregate per lap)
        lap_times = df.groupby(['vehicle_id', 'lap', 'track']).agg({
            'timestamp': ['min', 'max']
        }).reset_index()
        
        lap_times.columns = ['vehicle_id', 'lap', 'track', 'start_time', 'end_time']
        lap_times['lap_time'] = (lap_times['end_time'] - lap_times['start_time']).dt.total_seconds()
        
        # Filter valid lap times (20-200 seconds)
        lap_times = lap_times[(lap_times['lap_time'] > 20) & (lap_times['lap_time'] < 200)]
        
        print(f"  Found {len(lap_times)} valid laps")
        
        # For each lap, extract sequence and target
        for idx, row in lap_times.iterrows():
            vehicle_id = row['vehicle_id']
            lap = row['lap']
            track = row['track']
            lap_time = row['lap_time']
            
            # Get data for this lap
            lap_data = df[
                (df['vehicle_id'] == vehicle_id) & 
                (df['lap'] == lap) &
                (df['track'] == track)
            ].sort_values('timestamp')
            
            if len(lap_data) < self.seq_len:
                continue
            
            # Extract features
            try:
                seq = lap_data[self.feature_cols].fillna(0).values[:self.seq_len]
                
                # Target: lap time (we'll normalize this later)
                target = lap_time
                
                sequences.append(seq)
                targets.append(target)
            except Exception as e:
                continue
        
        return sequences, np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.sequences[idx])
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


def train_lap_time_model():
    """Train the Lap-Time Transformer"""
    
    print("="*60)
    print("LAP-TIME TRANSFORMER TRAINING")
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
    
    # Sample for faster training (use 10% for testing)
    sample_size = min(len(df_train), 1000000)  # 1M rows max for training
    df_train_sample = df_train.sample(n=sample_size, random_state=42)
    print(f"Using {len(df_train_sample):,} rows for training")
    
    # Create datasets
    print("\n[2/6] Preparing sequences...")
    train_dataset = LapSequenceDataset(df_train_sample, seq_len=200)
    
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
    print("\n[3/6] Initializing Lap-Time Transformer...")
    input_dim = len(train_dataset.feature_cols)
    model = LapTimeTransformer(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_seq_len=200,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    model = model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    mse_criterion = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
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
        train_mse = 0
        train_q_loss = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mean, quantiles = model(X)
            
            # Combined loss
            mse_loss = mse_criterion(mean, y)
            q_loss = quantile_criterion(quantiles, y)
            loss = mse_loss + 0.5 * q_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse_loss.item()
            train_q_loss += q_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_q_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_mse = 0
        val_q_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                
                mean, quantiles = model(X)
                
                mse_loss = mse_criterion(mean, y)
                q_loss = quantile_criterion(quantiles, y)
                loss = mse_loss + 0.5 * q_loss
                
                val_loss += loss.item()
                val_mse += mse_loss.item()
                val_q_loss += q_loss.item()
        
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_q_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Q: {train_q_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (MSE: {val_mse:.4f}, Q: {val_q_loss:.4f})")
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
                'input_dim': input_dim
            }
            
            torch.save(checkpoint, '/tmp/lap_time_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
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
    
    blob = bucket.blob('lap_time_transformer/model.pth')
    blob.upload_from_filename('/tmp/lap_time_best.pth')
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/lap_time_transformer/model.pth")
    
    # Save metrics
    metrics = {
        'best_val_loss': float(best_val_loss),
        'best_val_mse': float(val_mse),
        'best_val_q_loss': float(val_q_loss),
        'num_epochs': epoch + 1,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'input_dim': input_dim,
        'feature_cols': train_dataset.dataset.feature_cols if hasattr(train_dataset, 'dataset') else train_dataset.feature_cols
    }
    
    metrics_path = "/tmp/lap_time_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('lap_time_transformer/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    print("✓ Metrics saved")
    
    print("\n[6/6] Evaluation...")
    print(f"\nFinal Metrics:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val MSE: {val_mse:.4f}")
    print(f"  Best Val Q-Loss: {val_q_loss:.4f}")
    print(f"  RMSE: {np.sqrt(val_mse):.4f} seconds")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = train_lap_time_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

