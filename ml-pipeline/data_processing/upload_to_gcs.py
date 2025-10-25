from google.cloud import storage
from google.oauth2 import service_account
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import settings

class GCSUploader:
    """Upload processed data to Cloud Storage"""
    
    def __init__(self):
        creds_path = settings.get_absolute_credential_path()
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        self.client = storage.Client(
            project=settings.gcp_project_id,
            credentials=credentials
        )
    
    def upload_dataframe(self, df: pd.DataFrame, bucket_name: str, 
                        blob_path: str, format='parquet'):
        """Upload DataFrame to GCS"""
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Save locally first
        local_path = f"/tmp/{Path(blob_path).name}"
        
        try:
            if format == 'parquet':
                df.to_parquet(local_path, index=False, compression='snappy')
            elif format == 'csv':
                df.to_csv(local_path, index=False)
            
            # Upload
            blob.upload_from_filename(local_path)
            
            gcs_path = f"gs://{bucket_name}/{blob_path}"
            print(f"✓ Uploaded to {gcs_path}")
            print(f"  Size: {len(df)} rows, {len(df.columns)} columns")
            
            # Clean up
            if os.path.exists(local_path):
                os.remove(local_path)
            
            return gcs_path
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return None
    
    def upload_processed_data(self, df: pd.DataFrame, session_id: str):
        """Upload engineered features"""
        
        blob_path = f"processed/{session_id}/features.parquet"
        
        return self.upload_dataframe(
            df,
            settings.gcs_bucket_processed,
            blob_path,
            format='parquet'
        )
    
    def create_train_test_split(self, df: pd.DataFrame):
        """Split by time and upload"""
        
        print("\nCreating train/test split...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # 80/20 split
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Upload
        train_path = self.upload_dataframe(
            train_df,
            settings.gcs_bucket_processed,
            'splits/train.parquet'
        )
        
        test_path = self.upload_dataframe(
            test_df,
            settings.gcs_bucket_processed,
            'splits/test.parquet'
        )
        
        print(f"\n✓ Train: {len(train_df)} rows")
        print(f"✓ Test: {len(test_df)} rows")
        
        return train_path, test_path

if __name__ == "__main__":
    print("Testing GCS Uploader...")
    
    # Create small test dataframe
    import numpy as np
    test_df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='1s'),
        'vehicle_id': ['test-vehicle'] * 100,
        'speed': np.random.uniform(50, 150, 100),
        'rpm': np.random.uniform(2000, 7000, 100)
    })
    
    uploader = GCSUploader()
    result = uploader.upload_dataframe(
        test_df,
        settings.gcs_bucket_processed,
        'test/sample_data.parquet'
    )
    
    if result:
        print(f"\n✓ Upload test successful!")
    else:
        print(f"\n✗ Upload test failed")

