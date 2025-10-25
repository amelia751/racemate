#!/usr/bin/env python3
"""
Complete data processing pipeline
Parse CSVs → Engineer Features → Upload to GCS
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from csv_parser import TelemetryParser
from feature_engineering import FeatureEngineer
from upload_to_gcs import GCSUploader
from config.settings import settings
import pandas as pd

def run_complete_pipeline():
    """Run the complete data processing pipeline"""
    
    print("="*60)
    print("COGNIRACE DATA PROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Parse CSVs
    print("\n[1/4] Parsing Telemetry CSVs...")
    parser = TelemetryParser()
    df_raw = parser.process_all_tracks()
    
    if df_raw.empty:
        print("✗ No data loaded. Exiting.")
        return False
    
    print(f"\n✓ Loaded {len(df_raw)} total telemetry points")
    
    # Step 2: Engineer Features
    print("\n[2/4] Engineering Features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_all_features(df_raw)
    
    # Step 3: Save locally
    print("\n[3/4] Saving Processed Data...")
    local_path = os.path.join(settings.processed_data_path, 'all_features.parquet')
    os.makedirs(settings.processed_data_path, exist_ok=True)
    df_features.to_parquet(local_path, index=False, compression='snappy')
    print(f"✓ Saved locally: {local_path}")
    
    # Step 4: Upload to GCS
    print("\n[4/4] Uploading to Google Cloud Storage...")
    uploader = GCSUploader()
    
    # Upload full dataset
    full_path = uploader.upload_dataframe(
        df_features,
        settings.gcs_bucket_processed,
        'all_tracks/features_complete.parquet'
    )
    
    # Create and upload train/test splits
    train_path, test_path = uploader.create_train_test_split(df_features)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nData Summary:")
    print(f"  Total rows: {len(df_features):,}")
    print(f"  Total features: {len(df_features.columns)}")
    print(f"  Vehicles: {df_features['vehicle_id'].nunique()}")
    print(f"  Sessions: {df_features['session_id'].nunique()}")
    print(f"\nData uploaded to:")
    print(f"  Full: {full_path}")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = run_complete_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

