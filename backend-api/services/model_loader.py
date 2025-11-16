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
import sys

# Add ml-pipeline to path so we can unpickle sklearn models with custom classes
ml_pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml-pipeline'))
if ml_pipeline_path not in sys.path:
    sys.path.insert(0, ml_pipeline_path)

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
        
        # Check if any model file exists in cache
        model_file_pth = local_path / "model.pth"
        model_file_pkl = local_path / "model.pkl"
        
        if model_file_pth.exists() or model_file_pkl.exists():
            # Check if cached file is still fresh
            existing_file = model_file_pth if model_file_pth.exists() else model_file_pkl
            age = time.time() - existing_file.stat().st_mtime
            if age < settings.model_cache_ttl:
                print(f"  Using cached model: {model_name}")
                return local_path
        
        print(f"  Downloading {model_name} from GCS...")
        
        try:
            bucket = self.storage_client.bucket(settings.gcs_bucket_models)
            
            # Try downloading .pth first (PyTorch), then .pkl (sklearn)
            try:
                blob_pth = bucket.blob(f"{model_name}/model.pth")
                if blob_pth.exists():
                    blob_pth.download_to_filename(str(model_file_pth))
                else:
                    # Try .pkl
                    blob_pkl = bucket.blob(f"{model_name}/model.pkl")
                    blob_pkl.download_to_filename(str(model_file_pkl))
            except:
                # Fallback to .pkl
                blob_pkl = bucket.blob(f"{model_name}/model.pkl")
                blob_pkl.download_to_filename(str(model_file_pkl))
            
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
        
        # Try both .pth and .pkl extensions
        checkpoint_path_pth = model_path / "model.pth"
        checkpoint_path_pkl = model_path / "model.pkl"
        
        model_data = {}
        metrics = {}
        
        # Load metrics
        metrics_path = model_path / "metrics.pkl"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
            except:
                print(f"  ⚠️  Could not load metrics")
        
        # Try PyTorch model first
        if checkpoint_path_pth.exists():
            try:
                checkpoint = torch.load(
                    checkpoint_path_pth,
                    map_location='cpu',
                    weights_only=False
                )
                
                model_data = {
                    'model_state': checkpoint.get('model_state_dict', checkpoint),
                    'scaler': checkpoint.get('scaler'),
                    'feature_cols': checkpoint.get('feature_cols', []),
                    'metrics': metrics,
                    'checkpoint': checkpoint,
                    'type': 'pytorch'
                }
                print(f"  ✓ Loaded {model_name} (PyTorch)")
                
            except Exception as e:
                print(f"  ✗ Error loading PyTorch model: {e}")
                return None
                
        # Try sklearn model
        elif checkpoint_path_pkl.exists():
            try:
                # Try with joblib first (more robust for sklearn)
                try:
                    import joblib
                    model = joblib.load(checkpoint_path_pkl)
                except:
                    # Fall back to pickle
                    with open(checkpoint_path_pkl, 'rb') as f:
                        model = pickle.load(f)
                
                model_data = {
                    'model': model,
                    'metrics': metrics,
                    'type': 'sklearn'
                }
                print(f"  ✓ Loaded {model_name} (sklearn)")
                
            except Exception as e:
                print(f"  ✗ Error loading sklearn model: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"  ✗ No model file found in {model_path}")
            return None
        
        # Cache
        self.loaded_models[model_name] = model_data
        self.model_metadata[model_name] = {
            'loaded_at': time.time(),
            'metrics': metrics
        }
        
        if model_data.get('feature_cols'):
            print(f"    Features: {len(model_data['feature_cols'])}")
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

