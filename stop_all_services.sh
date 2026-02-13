#!/bin/bash

echo "🛑 Stopping All Services..."
echo "===================================="

# Function to stop process by pattern
stop_process() {
    local pattern=$1
    local name=$2
    pids=$(ps aux | grep "$pattern" | grep -v grep | awk '{print $2}')
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill $pid 2>/dev/null
        done
        echo "  ✅ $name stopped"
    else
        echo "  ℹ️  $name not running"
    fi
}

# Stop MLflow
echo "Stopping MLflow..."
stop_process "mlflow ui" "MLflow"

# Stop Airflow
echo "Stopping Airflow..."
stop_process "airflow webserver" "Airflow webserver"
stop_process "airflow scheduler" "Airflow scheduler"

# Stop Streamlit
echo "Stopping Streamlit..."
stop_process "streamlit run" "Streamlit"

echo ""
echo "✅ All services stopped!"
