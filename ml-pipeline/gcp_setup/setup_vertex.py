from google.cloud import aiplatform
from google.oauth2 import service_account
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import settings

def initialize_vertex_ai():
    """Initialize Vertex AI with service account"""
    try:
        creds_path = settings.get_absolute_credential_path()
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        
        aiplatform.init(
            project=settings.gcp_project_id,
            location=settings.vertex_ai_location,
            credentials=credentials,
            staging_bucket=settings.vertex_ai_staging_bucket
        )
        
        print(f"✓ Vertex AI initialized")
        print(f"  Project: {settings.gcp_project_id}")
        print(f"  Location: {settings.vertex_ai_location}")
        print(f"  Staging: {settings.vertex_ai_staging_bucket}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize Vertex AI: {e}")
        return False

if __name__ == "__main__":
    success = initialize_vertex_ai()
    sys.exit(0 if success else 1)

