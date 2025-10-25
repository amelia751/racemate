#!/usr/bin/env python3
"""
Train Driver Embedding Model (Transformer with CLS token)
Learns personalized driver representations from clean laps
Multi-task learning with auxiliary supervision
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
from models.driver_embedding import DriverEmbedding
import pickle

class DriverEmbeddingDataset(Dataset):
    """Dataset for driver embedding learning"""
    
    def __init__(self, df: pd.DataFrame, seq_len=200, scaler=None):
        self.seq_len = seq_len
        
        # Feature columns for telemetry
        self.feature_cols = [
            'speed', 'nmot', 'gear', 'pbrake_f', 'pbrake_r',
            'accx_can', 'accy_can', 'Steering_Angle',
            'speed_rolling_mean_5s', 'nmot_rolling_mean_5s',
            'brake_energy', 'lateral_load', 'tire_stress_proxy',
            'steer_rate', 'acc_magnitude', 'throttle_variance'
        ]
        
        # Only use features that exist
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]
        
        print(f"Using {len(self.feature_cols)} features for driver embedding")
        
        # Prepare sequences with driver IDs and auxiliary targets
        self.sequences, self.driver_ids, self.aux_targets = self._prepare_data(df)
        
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
        
        print(f"Dataset: {len(self.sequences)} driver sequences")
        print(f"Unique drivers: {len(set(self.driver_ids))}")
    
    def _prepare_data(self, df):
        """Extract sequences per driver"""
        sequences = []
        driver_ids = []
        aux_targets = []
        
        # Group by vehicle and lap
        grouped = df.groupby(['vehicle_id', 'lap', 'track'])
        
        print(f"  Processing {len(grouped)} laps...")
        
        for (vehicle_id, lap, track), lap_data in grouped:
            if len(lap_data) < self.seq_len:
                continue
            
            # Sort by timestamp
            lap_data = lap_data.sort_values('timestamp')
            
            # Extract sequence
            try:
                seq = lap_data.iloc[:self.seq_len][self.feature_cols]
                seq = seq.astype(float).fillna(0).values
                
                if seq.shape[0] != self.seq_len:
                    continue
                
                # Auxiliary targets (synthetic - in production would use real driver metrics)
                
                # Sector delta proxy (average speed deviation from mean) - clip to reasonable range
                speed_mean = lap_data['speed'].mean() if 'speed' in lap_data and not lap_data['speed'].isna().all() else 100
                sector_delta = np.clip((speed_mean - 100) / 50, -2, 2)
                
                # Throttle discipline (inverse of variance) - clip to 0-1
                throttle_var = lap_data['throttle_variance'].mean() if 'throttle_variance' in lap_data and not lap_data['throttle_variance'].isna().all() else 0.5
                throttle_disc = np.clip(1.0 / (throttle_var + 0.1), 0, 1)
                
                # Brake bias proxy (front/rear ratio) - clip to 0.2-0.8
                brake_f = lap_data['pbrake_f'].mean() if 'pbrake_f' in lap_data and not lap_data['pbrake_f'].isna().all() else 50
                brake_r = lap_data['pbrake_r'].mean() if 'pbrake_r' in lap_data and not lap_data['pbrake_r'].isna().all() else 50
                brake_sum = brake_f + brake_r
                brake_bias = np.clip(brake_f / (brake_sum + 1.0), 0.2, 0.8) if brake_sum > 0 else 0.5
                
                sequences.append(seq)
                driver_ids.append(vehicle_id)
                aux_targets.append({
                    'sector_delta': sector_delta,
                    'throttle_discipline': throttle_disc,
                    'brake_bias': brake_bias
                })
                
            except Exception as e:
                continue
        
        print(f"  Created {len(sequences)} driver sequences")
        
        return sequences, driver_ids, aux_targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.sequences[idx])
        driver_id = self.driver_ids[idx]
        aux = self.aux_targets[idx]
        
        # Convert aux targets to tensor
        aux_tensor = torch.FloatTensor([
            aux['sector_delta'],
            aux['throttle_discipline'],
            aux['brake_bias']
        ])
        
        return X, driver_id, aux_tensor


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


def train_driver_embedding():
    """Train the Driver Embedding model"""
    
    print("="*60)
    print("DRIVER EMBEDDING TRAINING")
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
    print("\n[2/6] Preparing driver sequences...")
    train_dataset = DriverEmbeddingDataset(df_train_sample, seq_len=200)
    
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
    print("\n[3/6] Initializing Driver Embedding Model...")
    input_dim = len(train_dataset.feature_cols)
    model = DriverEmbedding(
        input_dim=input_dim,
        hidden_dim=128,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    model = model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: 128")
    print(f"  Embedding dim: 32")
    print(f"  Num heads: 4")
    print(f"  Num layers: 2")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Multi-task loss: MSE for auxiliary tasks
    criterion_aux = nn.MSELoss()
    
    # Simplified: Just use auxiliary losses for supervision
    # In production, would use proper contrastive learning with driver ID labels
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    print("\n[4/6] Training...")
    num_epochs = 25
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 8
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, (X, driver_ids, aux_targets) in enumerate(train_loader):
            X = X.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings, aux_outputs = model(X)
            
            # Auxiliary task losses (multi-task supervision)
            aux_loss = 0
            aux_loss += criterion_aux(aux_outputs['sector_delta'], aux_targets[:, 0:1])
            aux_loss += criterion_aux(aux_outputs['throttle_discipline'], aux_targets[:, 1:2])
            aux_loss += criterion_aux(aux_outputs['brake_bias'], aux_targets[:, 2:3])
            aux_loss /= 3
            
            # Use auxiliary loss only (embeddings learn implicitly)
            loss = aux_loss
            
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
        
        with torch.no_grad():
            for X, driver_ids, aux_targets in val_loader:
                X = X.to(device)
                aux_targets = aux_targets.to(device)
                
                embeddings, aux_outputs = model(X)
                
                # Auxiliary loss
                aux_loss = 0
                aux_loss += criterion_aux(aux_outputs['sector_delta'], aux_targets[:, 0:1])
                aux_loss += criterion_aux(aux_outputs['throttle_discipline'], aux_targets[:, 1:2])
                aux_loss += criterion_aux(aux_outputs['brake_bias'], aux_targets[:, 2:3])
                aux_loss /= 3
                
                loss = aux_loss
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
                'scaler': train_dataset.scaler,
                'feature_cols': train_dataset.feature_cols,
                'input_dim': input_dim
            }
            
            torch.save(checkpoint, '/tmp/driver_embed_best.pth')
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
    
    blob = bucket.blob('driver_embedding/model.pth')
    blob.upload_from_filename('/tmp/driver_embed_best.pth')
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/driver_embedding/model.pth")
    
    # Save metrics
    metrics = {
        'best_val_loss': float(best_val_loss),
        'num_epochs': epoch + 1,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'input_dim': input_dim,
        'feature_cols': train_dataset.feature_cols,
        'embedding_dim': 32
    }
    
    metrics_path = "/tmp/driver_embed_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('driver_embedding/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    print("✓ Metrics saved")
    
    print("\n[6/6] Evaluation...")
    print(f"\nFinal Metrics:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model Type: Transformer with CLS token")
    print(f"  Output: 32-dim driver embedding + auxiliary predictions")
    print(f"  Use Case: Driver style personalization, performance comparison")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = train_driver_embedding()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

