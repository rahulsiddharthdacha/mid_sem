#!/bin/bash

echo "🚀 Starting MLflow Server..."
echo "================================"

# Set MLflow tracking directory
export MLFLOW_TRACKING_URI=./mlruns

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns

echo ""
echo "✅ MLflow UI started!"
echo "Access at: http://localhost:5000"
echo ""
echo "To stop MLflow, press Ctrl+C"
