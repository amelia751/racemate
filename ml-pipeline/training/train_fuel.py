#!/usr/bin/env python3
"""
Train Fuel Consumption Model (XGBoost)
Simple tabular model - good for testing the training pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.fuel_consumption import FuelConsumptionModel
from config.settings import settings
from google.cloud import storage
from google.oauth2 import service_account
import pickle

def load_data_from_gcs(bucket_name: str, blob_path: str) -> pd.DataFrame:
    """Load parquet from GCS"""
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Download to temp file
    local_path = f"/tmp/{blob_path.split('/')[-1]}"
    blob.download_to_filename(local_path)
    
    df = pd.read_parquet(local_path)
    print(f"Loaded {len(df)} rows from gs://{bucket_name}/{blob_path}")
    
    return df

def prepare_fuel_data(df: pd.DataFrame):
    """Prepare features and target for fuel model - aggregated by lap"""
    
    print(f"Input data shape: {df.shape}")
    print(f"Columns available: {len(df.columns)}")
    
    # Aggregate by vehicle and lap
    # This handles sparse telemetry data better than row-by-row
    agg_funcs = {
        'nmot': 'mean',
        'aps': 'mean', 
        'gear': 'mean',
        'speed': 'mean',
        'throttle_variance': 'mean',
        'on_full_throttle': 'sum',  # Total time at full throttle
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    print(f"Aggregating by lap using: {list(agg_dict.keys())}")
    
    df_agg = df.groupby(['vehicle_id', 'lap', 'track', 'race'], dropna=False).agg(agg_dict).reset_index()
    
    print(f"After aggregation: {len(df_agg)} laps")
    
    # Check null counts per column and remove columns that are entirely null
    print("\nNull counts in aggregated data:")
    valid_cols = []
    for col in agg_dict.keys():
        null_count = df_agg[col].isnull().sum()
        null_pct = null_count/len(df_agg)*100
        print(f"  {col}: {null_count} ({null_pct:.1f}%)")
        
        if null_pct < 100:  # Keep columns that have at least SOME data
            valid_cols.append(col)
        else:
            print(f"    ⚠️  Dropping {col} - entirely null")
    
    # Use forward fill and backward fill for remaining nulls
    for col in valid_cols:
        df_agg[col] = df_agg.groupby('vehicle_id')[col].ffill().bfill()
        
        # If still nulls, use global median
        if df_agg[col].isnull().any():
            median_val = df_agg[col].median()
            if not np.isnan(median_val):
                df_agg[col] = df_agg[col].fillna(median_val)
                print(f"  Filled remaining nulls in {col} with median: {median_val:.2f}")
            else:
                # If median is still NaN, drop this column too
                print(f"    ⚠️  Dropping {col} - cannot compute median")
                valid_cols.remove(col)
    
    print(f"\nUsing {len(valid_cols)} features: {valid_cols}")
    print(f"After processing: {len(df_agg)} laps with complete data")
    
    # Feature columns
    feature_cols = valid_cols + ['lap']
    
    # Create synthetic fuel consumption target
    # Fuel burn depends on available features
    fuel_components = []
    
    if 'nmot' in valid_cols:
        fuel_components.append(0.0001 * df_agg['nmot'])
    if 'aps' in valid_cols:
        fuel_components.append(0.003 * df_agg['aps'])
    if 'speed' in valid_cols:
        fuel_components.append(0.001 * df_agg['speed'])
    if 'on_full_throttle' in valid_cols:
        fuel_components.append(0.0001 * df_agg['on_full_throttle'])
    
    # Add lap effect
    fuel_components.append(0.01 * df_agg['lap'])
    
    # Add noise
    fuel_components.append(np.random.normal(0, 0.05, len(df_agg)))
    
    df_agg['fuel_burn_rate'] = sum(fuel_components).clip(0.5, 2.5)
    
    X = df_agg[feature_cols]
    y = df_agg['fuel_burn_rate']
    
    print(f"\nFinal dataset:")
    print(f"  Features: {feature_cols}")
    print(f"  Samples: {len(X)}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, feature_cols

def train_fuel_model():
    """Train fuel consumption model"""
    
    print("="*60)
    print("FUEL CONSUMPTION MODEL TRAINING")
    print("="*60)
    
    # Load training data
    print("\n[1/5] Loading data from GCS...")
    df_train = load_data_from_gcs(
        settings.gcs_bucket_processed,
        'splits/train.parquet'
    )
    
    # Prepare data
    print("\n[2/5] Preparing features...")
    X_train, y_train, feature_names = prepare_fuel_data(df_train)
    
    # Split for validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train samples: {len(X_train_sub):,}")
    print(f"Val samples: {len(X_val):,}")
    
    # Train model
    print("\n[3/5] Training XGBoost model...")
    model = FuelConsumptionModel()
    model.feature_names = feature_names
    model.train(X_train_sub, y_train_sub, X_val, y_val)
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    train_pred = model.predict(X_train_sub)
    val_pred = model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train_sub, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train_sub, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    train_r2 = r2_score(y_train_sub, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print("\nMetrics:")
    print(f"  Train MAE: {train_mae:.4f} L/lap")
    print(f"  Val MAE:   {val_mae:.4f} L/lap")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Val RMSE:   {val_rmse:.4f}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Val R²:   {val_r2:.4f}")
    
    print("\nFeature Importances:")
    importances = model.get_feature_importance()
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:20s}: {imp:.4f}")
    
    # Save model
    print("\n[5/5] Saving model to GCS...")
    local_model_path = "/tmp/fuel_model.pkl"
    model.save(local_model_path)
    
    # Upload to GCS
    credentials = service_account.Credentials.from_service_account_file(
        settings.get_absolute_credential_path()
    )
    client = storage.Client(project=settings.gcp_project_id, credentials=credentials)
    bucket = client.bucket(settings.gcs_bucket_models)
    blob = bucket.blob('fuel_consumption/model.pkl')
    blob.upload_from_filename(local_model_path)
    
    print(f"✓ Model saved to gs://{settings.gcs_bucket_models}/fuel_consumption/model.pkl")
    
    # Save metrics
    metrics = {
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'train_r2': float(train_r2),
        'val_r2': float(val_r2),
        'feature_importances': importances
    }
    
    metrics_path = "/tmp/fuel_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    blob = bucket.blob('fuel_consumption/metrics.pkl')
    blob.upload_from_filename(metrics_path)
    
    print("✓ Metrics saved")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, metrics

if __name__ == "__main__":
    try:
        model, metrics = train_fuel_model()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

