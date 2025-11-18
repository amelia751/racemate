from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # GCP
    gcp_project_id: str
    gcp_service_account_path: str = ""  # Optional - Cloud Run uses service account automatically
    gcp_region: str = "us-central1"
    
    # Cloud Storage
    gcs_bucket_models: str
    
    # API Configuration
    api_port: int = int(os.getenv("PORT", "8005"))  # Use Cloud Run's PORT or default to 8005
    api_host: str = "0.0.0.0"
    api_workers: int = 4
    api_reload: bool = False
    api_log_level: str = "info"
    
    # Model Cache
    model_cache_dir: str = "/tmp/racemate_models"
    model_cache_ttl: int = 3600
    
    # Prediction Settings
    prediction_timeout: int = 30
    batch_size_limit: int = 32
    
    # Google/Gemini Configuration
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-exp"
    
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

