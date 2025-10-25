"""
Model loading service for Cognirace API
Downloads models from GCS and caches them locally
"""

import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account
from config.settings import settings
import time
import os

class ModelLoader:
    """Load and cache ML models from GCS"""
    
    def __init__(self):
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(
            settings.get_absolute_credential_path()
        )
        self.storage_client = storage.Client(
            project=settings.gcp_project_id,
            credentials=credentials
        )
        
        # Create cache directory
        self.cache_dir = Path(settings.model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache
        self.loaded_models = {}
        self.model_metadata = {}
        
        print(f"✓ Model loader initialized")
        print(f"  Cache dir: {self.cache_dir}")
    
    def download_model(self, model_name: str) -> Path:
        """Download model from GCS to local cache"""
        
        local_path = self.cache_dir / model_name
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Download model.pth
        model_file = local_path / "model.pth"
        
        if model_file.exists():
            # Check if cached file is still fresh
            age = time.time() - model_file.stat().st_mtime
            if age < settings.model_cache_ttl:
                print(f"  Using cached model: {model_name}")
                return local_path
        
        print(f"  Downloading {model_name} from GCS...")
        
        try:
            bucket = self.storage_client.bucket(settings.gcs_bucket_models)
            
            # Download model checkpoint
            blob = bucket.blob(f"{model_name}/model.pth")
            blob.download_to_filename(str(model_file))
            
            # Download metrics if available
            metrics_blob = bucket.blob(f"{model_name}/metrics.pkl")
            if metrics_blob.exists():
                metrics_blob.download_to_filename(str(local_path / "metrics.pkl"))
            
            print(f"  ✓ Downloaded {model_name}")
            
        except Exception as e:
            print(f"  ✗ Failed to download {model_name}: {e}")
            raise
        
        return local_path
    
    def load_model(self, model_name: str, model_class: Any) -> Dict[str, Any]:
        """Load model and return model + metadata"""
        
        # Check cache
        if model_name in self.loaded_models:
            print(f"  Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        print(f"Loading model: {model_name}")
        
        # Download if needed
        model_path = self.download_model(model_name)
        
        # Load checkpoint
        checkpoint_path = model_path / "model.pth"
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False
        )
        
        # Load metrics
        metrics = {}
        metrics_path = model_path / "metrics.pkl"
        if metrics_path.exists():
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
        
        # Extract model state and metadata
        model_state = checkpoint.get('model_state_dict', checkpoint)
        scaler = checkpoint.get('scaler')
        feature_cols = checkpoint.get('feature_cols', [])
        
        # Initialize model (will be done by specific prediction services)
        model_data = {
            'model_state': model_state,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'metrics': metrics,
            'checkpoint': checkpoint
        }
        
        # Cache
        self.loaded_models[model_name] = model_data
        self.model_metadata[model_name] = {
            'loaded_at': time.time(),
            'metrics': metrics
        }
        
        print(f"  ✓ Loaded {model_name}")
        if feature_cols:
            print(f"    Features: {len(feature_cols)}")
        if metrics:
            print(f"    Metrics: {list(metrics.keys())[:3]}")
        
        return model_data
    
    def clear_cache(self):
        """Clear model cache"""
        self.loaded_models.clear()
        self.model_metadata.clear()
        print("✓ Model cache cleared")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a loaded model"""
        return self.model_metadata.get(model_name)


# Global model loader instance
model_loader = ModelLoader()

