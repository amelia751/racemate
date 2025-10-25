from google.cloud import storage
from google.oauth2 import service_account
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import settings

def get_storage_client():
    """Initialize GCS client"""
    creds_path = settings.get_absolute_credential_path()
    credentials = service_account.Credentials.from_service_account_file(creds_path)
    return storage.Client(
        project=settings.gcp_project_id,
        credentials=credentials
    )

def create_bucket_if_not_exists(client, bucket_name, location=None):
    """Create bucket with error handling"""
    try:
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            bucket.location = location or settings.gcp_region
            bucket.storage_class = "STANDARD"
            bucket = client.create_bucket(bucket, location=location)
            print(f"✓ Created bucket: {bucket_name}")
        else:
            print(f"✓ Bucket exists: {bucket_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to create {bucket_name}: {e}")
        return False

def setup_all_buckets():
    """Create all required GCS buckets"""
    print(f"Creating buckets in project: {settings.gcp_project_id}")
    print(f"Region: {settings.gcp_region}\n")
    
    try:
        client = get_storage_client()
    except Exception as e:
        print(f"✗ Failed to initialize storage client: {e}")
        return False
    
    buckets = [
        settings.gcs_bucket_raw,
        settings.gcs_bucket_processed,
        settings.gcs_bucket_models,
        settings.gcs_bucket_results,
        "cognirace-vertex-staging",  # For Vertex AI
    ]
    
    results = []
    for bucket_name in buckets:
        success = create_bucket_if_not_exists(client, bucket_name)
        results.append(success)
    
    if all(results):
        print(f"\n✓ All {len(buckets)} buckets ready!")
        return True
    else:
        print(f"\n✗ Some buckets failed. Check permissions.")
        return False

if __name__ == "__main__":
    success = setup_all_buckets()
    sys.exit(0 if success else 1)

