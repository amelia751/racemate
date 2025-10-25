#!/usr/bin/env python3
"""
Train FCY Hazard Model (Survival Analysis)
Predicts full-course yellow (caution) probability over 6-lap horizon
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
from models.fcy_hazard import FCYHazardModel
import pickle

class FCYDataset(Dataset):
    """Dataset for FCY/caution prediction"""
    
    def __init__(self, df: pd.DataFrame, seq_len=200, scaler=None):
        self.seq_len = seq_len
        
        # Feature columns for sequence
        self.feature_cols = [
            'speed', 'nmot', 'gear', 'pbrake_f', 'pbrake_r',
            'accx_can', 'accy_can', 'Steering_Angle',
            'speed_rolling_mean_5s', 'nmot_rolling_mean_5s',
            'brake_energy', 'lateral_load', 'tire_stress_proxy',
            'steer_rate', 'acc_magnitude', 'steer_jerk'
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
        if len(self.targets) > 0:
            print(f"FCY events: {self.targets.sum()}/{len(self.targets)} ({self.targets.mean()*100:.1f}%)")
    
    def _prepare_data(self, df):
        """Extract sequences and FCY targets (synthetic)"""
        sequences = []
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
                
                # Target: Synthetic FCY probability
                # In production, this would come from actual race control data
                # FCY events are triggered by: aggressive driving, multiple cars close together, anomalies
                
                # Risk factors for FCY:
                high_speed_variance = lap_data['speed'].std() > 20 if 'speed' in lap_data else False
                high_brake_energy = lap_data.get('brake_energy', pd.Series([0])).mean() > 500
                high_steer_jerk = lap_data.get('steer_jerk', pd.Series([0])).abs().mean() > 5
                
                # Base FCY probability (5% per lap is realistic)
                base_prob = 0.05
                
                # Increase probability with risk factors
                risk_multiplier = 1.0
                if high_speed_variance:
                    risk_multiplier += 0.5
                if high_brake_energy:
                    risk_multiplier += 0.3
                if high_steer_jerk:
                    risk_multiplier += 0.2
                
                fcy_prob = min(base_prob * risk_multiplier, 0.3)
                
                # Binary target (did FCY occur?)
                fcy_occurred = np.random.random() < fcy_prob
                
                sequences.append(seq)
                targets.append(float(fcy_occurred))
                
            except Exception as e:
                continue
        
        print(f"  Created {len(sequences)} valid sequences")
        
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


def train_fcy_model():
    """Train the FCY Hazard model"""
    
    print("="*60)
    print("FCY HAZARD MODEL TRAINING")
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
    train_dataset = FCYDataset(df_train_sample, seq_len=200)
    
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
    print("\n[3/6] Initializing FCY Hazard Model...")
    input_dim = len(train_dataset.feature_cols)
    model = FCYHazardModel(
        input_dim=input_dim,
        hidden_channels=128,
        kernel_size=3,
        num_layers=3,
        horizon_laps=6
    )
    
    model = model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Horizon: 6 laps")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Binary cross-entropy for FCY occurrence prediction
    criterion = nn.BCEWithLogitsLoss()
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
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            hazard_rates, cumulative_prob = model(X)
            
            # Loss on cumulative probability (whether FCY occurs in horizon)
            loss = criterion(cumulative_prob, y)
            
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                
                hazard_rates, cumulative_prob = model(X)
                loss = criterion(cumulative_prob, y)
                
                val_loss += loss.item()
                
                # Accuracy
                pred = (torch.sigmoid(cumulative_prob) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {accuracy:.2f}%")
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
                'val_accuracy': accuracy,
                'scaler': train_dataset.dataset.scaler if hasattr(train_dataset, 'dataset') else train_dataset.scaler,
                'feature_cols': train_dataset.dataset.feature_cols if hasattr(train_dataset, 'dataset') else train_dataset.feature_cols,
                'input_dim': input_dim
            }
            
            torch.save(checkpoint, '/tmp/fcy_best.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f}, acc: {accuracy:.2f}%)")
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
    
    blob = bucket.blob('fcy_hazard/model.pth')
    blob.upload_from_filename('/tmp/fcy_best.pth')
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/fcy_hazard/model.pth")
    
    # Save metrics
    metrics = {
        'best_val_loss': float(best_val_loss),
        'best_val_accuracy': float(accuracy),
        'num_epochs': epoch + 1,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'input_dim': input_dim,
        'feature_cols': train_dataset.dataset.feature_cols if hasattr(train_dataset, 'dataset') else train_dataset.feature_cols,
        'horizon_laps': 6
    }
    
    metrics_path = "/tmp/fcy_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('fcy_hazard/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    print("✓ Metrics saved")
    
    print("\n[6/6] Evaluation...")
    print(f"\nFinal Metrics:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Accuracy: {accuracy:.2f}%")
    print(f"  Model Type: Survival analysis (6-lap horizon)")
    print(f"  Output: Hazard rates per lap + cumulative probability")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = train_fcy_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

