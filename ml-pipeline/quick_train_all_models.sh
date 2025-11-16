#!/bin/bash
# Quick train all 8 models

export GOOGLE_APPLICATION_CREDENTIALS="/Users/anhlam/hack-the-track/ml-pipeline/config/gcp_credentials.json"

cd /Users/anhlam/hack-the-track/ml-pipeline
source venv/bin/activate

echo "🏋️  Training all 8 models..."
echo ""

# Train each model
python training/train_fuel.py && echo "✅ Fuel model trained"
python training/train_lap_time.py && echo "✅ Lap-time model trained"  
python training/train_tire.py && echo "✅ Tire model trained"
python training/train_fcy.py && echo "✅ FCY model trained"
python training/train_pit_loss.py && echo "✅ Pit loss model trained"
python training/train_anomaly.py && echo "✅ Anomaly model trained"
python training/train_driver_embed.py && echo "✅ Driver embedding trained"
python training/train_traffic.py && echo "✅ Traffic model trained"

echo ""
echo "🎉 All models trained!"
