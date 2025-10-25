from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # GCP
    gcp_project_id: str
    gcp_service_account_path: str
    gcp_region: str = "us-central1"
    gcp_zone: str = "us-central1-a"
    
    # Cloud Storage
    gcs_bucket_raw: str
    gcs_bucket_processed: str
    gcs_bucket_models: str
    gcs_bucket_results: str
    
    # Vertex AI
    vertex_ai_location: str = "us-central1"
    vertex_ai_staging_bucket: str
    
    # Data Paths
    local_data_path: str
    processed_data_path: str
    
    # Training
    training_machine_type: str = "n1-standard-8"
    training_accelerator_type: str = "NVIDIA_TESLA_T4"
    training_accelerator_count: int = 1
    
    # Deployment
    endpoint_machine_type: str = "n1-standard-4"
    endpoint_min_replicas: int = 0
    endpoint_max_replicas: int = 2
    
    # API Configuration
    api_port: int = 8005
    api_host: str = "0.0.0.0"
    api_workers: int = 4
    api_timeout: int = 300
    
    # Model Endpoints (optional, will be set after deployment)
    endpoint_fuel: str = ""
    endpoint_laptime: str = ""
    endpoint_tire: str = ""
    endpoint_fcy: str = ""
    endpoint_pitloss: str = ""
    endpoint_anomaly: str = ""
    endpoint_driver: str = ""
    endpoint_traffic: str = ""
    
    class Config:
        env_file = ".env.local"
        case_sensitive = False
        
    def get_absolute_credential_path(self) -> str:
        """Get absolute path for service account credentials"""
        if self.gcp_service_account_path.startswith('./'):
            # Relative to ml-pipeline directory
            base_dir = Path(__file__).parent.parent
            return str(base_dir / self.gcp_service_account_path.lstrip('./'))
        return self.gcp_service_account_path

# Global settings instance
settings = Settings()

