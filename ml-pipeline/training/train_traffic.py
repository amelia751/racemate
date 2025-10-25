#!/usr/bin/env python3
"""
Train Traffic GNN Model (Simplified version using attention)
Predicts traffic loss and overtake probability
Uses attention mechanism instead of torch-geometric for reliability
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
from models.traffic_gnn import TrafficGNN
import pickle

class TrafficDataset(Dataset):
    """Dataset for traffic interaction prediction"""
    
    def __init__(self, df: pd.DataFrame, num_cars=5, scaler=None):
        self.num_cars = num_cars  # Number of cars to consider in traffic graph
        
        # Feature columns for each car node
        self.feature_cols = [
            'speed', 'nmot', 'gear', 'pbrake_f', 'pbrake_r',
            'accx_can', 'accy_can', 'Steering_Angle',
            'speed_rolling_mean_5s', 'nmot_rolling_mean_5s',
            'brake_energy', 'lateral_load', 'tire_stress_proxy',
            'steer_rate', 'acc_magnitude', 'throttle_variance'
        ]
        
        # Only use features that exist
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]
        
        print(f"Using {len(self.feature_cols)} features per car node")
        
        # Prepare traffic scenarios
        self.graphs, self.targets = self._prepare_data(df)
        
        # Fit or apply scaler
        if scaler is None:
            self.scaler = StandardScaler()
            # Flatten all graphs to fit scaler
            all_data = np.vstack([g.reshape(-1, len(self.feature_cols)) for g in self.graphs])
            self.scaler.fit(all_data)
        else:
            self.scaler = scaler
        
        # Scale graphs
        self.graphs = [self.scaler.transform(g.reshape(-1, len(self.feature_cols))).reshape(self.num_cars, len(self.feature_cols)) 
                       for g in self.graphs]
        
        print(f"Dataset: {len(self.graphs)} traffic scenarios")
    
    def _prepare_data(self, df):
        """Create traffic graphs from telemetry"""
        graphs = []
        targets = []
        
        # Group by lap and track (create synthetic traffic snapshots)
        # We'll sample random subsets of cars to simulate traffic
        
        print(f"  Creating synthetic traffic scenarios...")
        
        # Get unique laps
        unique_laps = df[['lap', 'track', 'vehicle_id']].drop_duplicates()
        laps_by_track = unique_laps.groupby(['lap', 'track'])['vehicle_id'].apply(list)
        
        np.random.seed(42)
        scenarios_created = 0
        max_scenarios = 3000
        
        for (lap, track), vehicle_ids in laps_by_track.items():
            if len(vehicle_ids) < 2:
                continue
            
            # Create multiple scenarios from this lap (sample different car combinations)
            num_scenarios_from_lap = min(5, max_scenarios - scenarios_created)
            
            for _ in range(num_scenarios_from_lap):
                # Select random cars
                selected_cars = np.random.choice(vehicle_ids, size=min(self.num_cars, len(vehicle_ids)), replace=False)
                
                # Get their data
                car_data_list = []
                for car_id in selected_cars:
                    car_rows = df[(df['lap'] == lap) & (df['track'] == track) & (df['vehicle_id'] == car_id)]
                    if len(car_rows) > 0:
                        # Sample one row
                        car_row = car_rows.sample(n=1, random_state=None).iloc[0]
                        car_features = car_row[self.feature_cols].astype(float).fillna(0).values
                        car_data_list.append(car_features)
                
                if len(car_data_list) < 2:
                    continue
                
                # Pad if needed
                while len(car_data_list) < self.num_cars:
                    car_data_list.append(np.zeros(len(self.feature_cols)))
                
                # Stack into graph
                graph_features = np.stack(car_data_list[:self.num_cars])
                
                # Synthetic targets
                speeds = graph_features[:, self.feature_cols.index('speed')] if 'speed' in self.feature_cols else np.ones(self.num_cars) * 100
                valid_speeds = speeds[speeds > 10]  # Only consider moving cars
                
                if len(valid_speeds) == 0:
                    continue
                
                # Traffic loss depends on speed variance
                speed_variance = np.var(valid_speeds) if len(valid_speeds) > 1 else 0
                traffic_loss = np.clip(speed_variance / 100.0, 0, 5.0)  # 0-5 seconds
                
                # Overtake probability depends on speed differential
                if len(valid_speeds) > 1:
                    speed_diff = np.max(valid_speeds) - np.min(valid_speeds)
                    overtake_prob = np.clip(speed_diff / 100.0, 0.0, 1.0)
                else:
                    overtake_prob = 0.0
                
                # Ensure valid values
                if np.isnan(traffic_loss) or np.isnan(overtake_prob):
                    continue
                
                graphs.append(graph_features)
                targets.append({
                    'traffic_loss': float(traffic_loss),
                    'overtake_prob': float(overtake_prob)
                })
                
                scenarios_created += 1
                
                if scenarios_created >= max_scenarios:
                    break
            
            if scenarios_created >= max_scenarios:
                break
        
        print(f"  Created {len(graphs)} traffic scenarios from {len(laps_by_track)} laps")
        
        return graphs, targets
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.graphs[idx])  # (num_cars, num_features)
        
        target = self.targets[idx]
        traffic_loss = torch.FloatTensor([target['traffic_loss']])
        overtake_prob = torch.FloatTensor([target['overtake_prob']])
        
        return X, traffic_loss, overtake_prob


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


def train_traffic_gnn():
    """Train the Traffic GNN model"""
    
    print("="*60)
    print("TRAFFIC GNN TRAINING")
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
    print("\n[2/6] Preparing traffic scenarios...")
    train_dataset = TrafficDataset(df_train_sample, num_cars=5)
    
    if len(train_dataset) == 0:
        print("✗ No valid scenarios created!")
        return None
    
    # Split for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset_sub, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset_sub, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n[3/6] Initializing Traffic GNN...")
    node_feature_dim = len(train_dataset.feature_cols)
    model = TrafficGNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    )
    
    model = model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  Node feature dim: {node_feature_dim}")
    print(f"  Hidden dim: 64")
    print(f"  Num layers: 2")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion_traffic = nn.MSELoss()
    criterion_overtake = nn.BCELoss()
    
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
        
        for batch_idx, (X, traffic_loss_target, overtake_prob_target) in enumerate(train_loader):
            X = X.to(device)
            traffic_loss_target = traffic_loss_target.to(device)
            overtake_prob_target = overtake_prob_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            traffic_loss_pred, overtake_prob_pred = model(X)
            
            # Combined loss
            loss_traffic = criterion_traffic(traffic_loss_pred, traffic_loss_target)
            loss_overtake = criterion_overtake(overtake_prob_pred, overtake_prob_target)
            loss = loss_traffic + loss_overtake
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X, traffic_loss_target, overtake_prob_target in val_loader:
                X = X.to(device)
                traffic_loss_target = traffic_loss_target.to(device)
                overtake_prob_target = overtake_prob_target.to(device)
                
                traffic_loss_pred, overtake_prob_pred = model(X)
                
                loss_traffic = criterion_traffic(traffic_loss_pred, traffic_loss_target)
                loss_overtake = criterion_overtake(overtake_prob_pred, overtake_prob_target)
                loss = loss_traffic + loss_overtake
                
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
                'node_feature_dim': node_feature_dim
            }
            
            torch.save(checkpoint, '/tmp/traffic_best.pth')
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
    
    blob = bucket.blob('traffic_gnn/model.pth')
    blob.upload_from_filename('/tmp/traffic_best.pth')
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/traffic_gnn/model.pth")
    
    # Save metrics
    metrics = {
        'best_val_loss': float(best_val_loss),
        'num_epochs': epoch + 1,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'node_feature_dim': node_feature_dim,
        'feature_cols': train_dataset.feature_cols
    }
    
    metrics_path = "/tmp/traffic_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('traffic_gnn/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    print("✓ Metrics saved")
    
    print("\n[6/6] Evaluation...")
    print(f"\nFinal Metrics:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model Type: Attention-based GNN (simplified)")
    print(f"  Output: Traffic loss (seconds) + overtake probability")
    print(f"  Use Case: Traffic analysis, overtake prediction")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = train_traffic_gnn()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

