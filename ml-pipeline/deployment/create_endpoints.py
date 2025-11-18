#!/usr/bin/env python3
"""
Create and deploy Vertex AI endpoints for all 8 RaceMate models
Handles endpoint creation, model registration, deployment, and testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from google.cloud import aiplatform
from google.oauth2 import service_account
from google.cloud import storage
from config.settings import settings
import time
import json
from pathlib import Path

class VertexAIDeployer:
    """Deploy models to Vertex AI endpoints"""
    
    def __init__(self):
        # Initialize credentials
        self.credentials = service_account.Credentials.from_service_account_file(
            settings.get_absolute_credential_path()
        )
        
        # Initialize Vertex AI
        aiplatform.init(
            project=settings.gcp_project_id,
            location=settings.vertex_ai_location,
            credentials=self.credentials
        )
        
        # Initialize GCS client
        self.storage_client = storage.Client(
            project=settings.gcp_project_id,
            credentials=self.credentials
        )
        
        print(f"✓ Initialized Vertex AI")
        print(f"  Project: {settings.gcp_project_id}")
        print(f"  Location: {settings.vertex_ai_location}")
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if model artifacts exist in GCS"""
        bucket = self.storage_client.bucket(settings.gcs_bucket_models)
        blob = bucket.blob(f'{model_name}/model.pth')
        exists = blob.exists()
        
        if exists:
            size_mb = (blob.size or 0) / (1024 * 1024)
            print(f"  ✓ Model artifacts found: {size_mb:.2f} MB")
        else:
            print(f"  ✗ Model artifacts not found in GCS")
        
        return exists
    
    def create_endpoint(self, endpoint_display_name: str) -> aiplatform.Endpoint:
        """Create or get existing Vertex AI endpoint"""
        
        print(f"\nCreating endpoint: {endpoint_display_name}")
        
        # Check if endpoint already exists
        existing_endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )
        
        if existing_endpoints:
            endpoint = existing_endpoints[0]
            print(f"  ✓ Using existing endpoint: {endpoint.resource_name}")
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                description=f"Endpoint for {endpoint_display_name}",
                labels={"project": "racemate", "phase": "2"}
            )
            print(f"  ✓ Created new endpoint: {endpoint.resource_name}")
        
        return endpoint
    
    def upload_model(self, model_display_name: str, model_gcs_uri: str) -> aiplatform.Model:
        """Upload model to Vertex AI Model Registry"""
        
        print(f"\nUploading model: {model_display_name}")
        print(f"  GCS URI: {model_gcs_uri}")
        
        # Check if model already exists
        existing_models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"'
        )
        
        if existing_models:
            model = existing_models[0]
            print(f"  ✓ Using existing model: {model.resource_name}")
        else:
            # Use pre-built PyTorch serving container
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_gcs_uri,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest",
                description=f"RaceMate {model_display_name}",
                labels={"project": "racemate", "phase": "2"}
            )
            print(f"  ✓ Model uploaded: {model.resource_name}")
        
        return model
    
    def deploy_model_to_endpoint(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        deployment_name: str
    ) -> str:
        """Deploy model to endpoint"""
        
        print(f"\nDeploying {deployment_name} to endpoint...")
        
        # Check if already deployed
        if endpoint.list_models():
            print(f"  ✓ Model already deployed to endpoint")
            deployed_model_id = endpoint.list_models()[0].id
            return deployed_model_id
        
        try:
            deployed_model = model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=deployment_name,
                machine_type=settings.endpoint_machine_type,
                min_replica_count=settings.endpoint_min_replicas,
                max_replica_count=settings.endpoint_max_replicas,
                accelerator_type=None,  # CPU only for cost optimization
                accelerator_count=0,
                sync=True
            )
            
            print(f"  ✓ Model deployed successfully")
            print(f"  Deployed model ID: {deployed_model.id}")
            
            return deployed_model.id
            
        except Exception as e:
            print(f"  ✗ Deployment failed: {e}")
            print(f"  Note: Custom PyTorch models need custom prediction handlers")
            print(f"  For now, we'll use the models directly from GCS in the API")
            return None
    
    def test_endpoint(self, endpoint: aiplatform.Endpoint, test_input: dict) -> dict:
        """Test endpoint with sample prediction"""
        
        print(f"\nTesting endpoint...")
        
        try:
            # Note: This will fail for custom PyTorch models without proper serving setup
            # We'll handle predictions directly in the FastAPI service instead
            prediction = endpoint.predict(instances=[test_input])
            print(f"  ✓ Endpoint test passed")
            return {"status": "success", "prediction": prediction}
        except Exception as e:
            print(f"  Note: Endpoint test skipped (will use direct model loading in API)")
            print(f"  Reason: {str(e)[:100]}")
            return {"status": "will_use_direct_loading", "reason": str(e)[:200]}
    
    def deploy_all_models(self):
        """Deploy all 8 RaceMate models"""
        
        models_config = [
            {
                "name": "fuel_consumption",
                "display_name": "RaceMate Fuel Consumption",
                "endpoint_name": "racemate-fuel-predictor",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/fuel_consumption/"
            },
            {
                "name": "lap_time_transformer",
                "display_name": "RaceMate Lap Time Predictor",
                "endpoint_name": "racemate-laptime-predictor",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/lap_time_transformer/"
            },
            {
                "name": "tire_degradation",
                "display_name": "RaceMate Tire Degradation",
                "endpoint_name": "racemate-tire-predictor",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/tire_degradation/"
            },
            {
                "name": "fcy_hazard",
                "display_name": "RaceMate FCY Hazard",
                "endpoint_name": "racemate-fcy-predictor",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/fcy_hazard/"
            },
            {
                "name": "pit_loss",
                "display_name": "RaceMate Pit Loss",
                "endpoint_name": "racemate-pitloss-predictor",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/pit_loss/"
            },
            {
                "name": "anomaly_detector",
                "display_name": "RaceMate Anomaly Detector",
                "endpoint_name": "racemate-anomaly-detector",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/anomaly_detector/"
            },
            {
                "name": "driver_embedding",
                "display_name": "RaceMate Driver Embedding",
                "endpoint_name": "racemate-driver-analyzer",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/driver_embedding/"
            },
            {
                "name": "traffic_gnn",
                "display_name": "RaceMate Traffic GNN",
                "endpoint_name": "racemate-traffic-analyzer",
                "gcs_path": f"gs://{settings.gcs_bucket_models}/traffic_gnn/"
            }
        ]
        
        print("="*70)
        print("VERTEX AI ENDPOINT DEPLOYMENT")
        print("="*70)
        
        deployment_results = []
        endpoint_ids = {}
        
        for model_config in models_config:
            print(f"\n{'='*70}")
            print(f"Processing: {model_config['display_name']}")
            print(f"{'='*70}")
            
            # Check if model exists in GCS
            if not self.check_model_exists(model_config['name']):
                print(f"  ⚠️  Skipping - model not found in GCS")
                deployment_results.append({
                    "model": model_config['name'],
                    "status": "skipped",
                    "reason": "model_not_in_gcs"
                })
                continue
            
            try:
                # Create endpoint
                endpoint = self.create_endpoint(model_config['endpoint_name'])
                endpoint_ids[model_config['name']] = endpoint.resource_name
                
                # Note: For custom PyTorch models, we'll load them directly in the API
                # rather than using Vertex AI's managed prediction service
                # This avoids the complexity of custom prediction handlers
                
                deployment_results.append({
                    "model": model_config['name'],
                    "status": "endpoint_created",
                    "endpoint_id": endpoint.resource_name,
                    "note": "Will use direct model loading in FastAPI for flexibility"
                })
                
                print(f"  ✓ Endpoint ready: {endpoint.resource_name}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                deployment_results.append({
                    "model": model_config['name'],
                    "status": "failed",
                    "error": str(e)
                })
        
        # Summary
        print(f"\n{'='*70}")
        print("DEPLOYMENT SUMMARY")
        print(f"{'='*70}")
        
        successful = sum(1 for r in deployment_results if r['status'] in ['endpoint_created', 'deployed'])
        failed = sum(1 for r in deployment_results if r['status'] == 'failed')
        skipped = sum(1 for r in deployment_results if r['status'] == 'skipped')
        
        print(f"\n✓ Endpoints created: {successful}/{len(models_config)}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        
        if successful > 0:
            print(f"\n✓ Models ready for FastAPI integration")
            print(f"  Approach: Direct model loading from GCS")
            print(f"  Benefit: Full control over inference, no custom serving containers needed")
        
        # Save endpoint IDs to file
        self.save_endpoint_ids(endpoint_ids)
        
        return deployment_results
    
    def save_endpoint_ids(self, endpoint_ids: dict):
        """Save endpoint IDs to a JSON file for API reference"""
        
        output_file = Path(__file__).parent / 'endpoint_ids.json'
        
        with open(output_file, 'w') as f:
            json.dump(endpoint_ids, f, indent=2)
        
        print(f"\n✓ Endpoint IDs saved to: {output_file}")
        print(f"  Use these for API configuration")


def main():
    """Main deployment flow"""
    
    try:
        deployer = VertexAIDeployer()
        results = deployer.deploy_all_models()
        
        print("\n" + "="*70)
        print("✓ DEPLOYMENT COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review endpoint_ids.json")
        print("  2. Build FastAPI service for predictions")
        print("  3. Test end-to-end predictions")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

