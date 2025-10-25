from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # GCP
    gcp_project_id: str
    gcp_service_account_path: str
    gcp_region: str = "us-central1"
    
    # Cloud Storage
    gcs_bucket_models: str
    
    # API Configuration
    api_port: int = 8005
    api_host: str = "0.0.0.0"
    api_workers: int = 4
    api_reload: bool = False
    api_log_level: str = "info"
    
    # Model Cache
    model_cache_dir: str = "/tmp/cognirace_models"
    model_cache_ttl: int = 3600
    
    # Prediction Settings
    prediction_timeout: int = 30
    batch_size_limit: int = 32
    
    class Config:
        env_file = ".env.local"
        case_sensitive = False
    
    def get_absolute_credential_path(self) -> str:
        """Get absolute path for service account credentials"""
        if self.gcp_service_account_path.startswith('../'):
            # Relative to backend-api directory
            base_dir = Path(__file__).parent.parent
            return str(base_dir / self.gcp_service_account_path)
        return self.gcp_service_account_path

# Global settings instance
settings = Settings()

