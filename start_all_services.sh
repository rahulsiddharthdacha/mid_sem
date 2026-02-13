#!/bin/bash

echo "🚀 Starting All Services..."
echo "===================================="
echo ""

# Function to check if a port is in use
check_port() {
    lsof -i:$1 > /dev/null 2>&1
    return $?
}

# Start MLflow
echo "1️⃣ Starting MLflow..."
if check_port 5000; then
    echo "   ⚠️  Port 5000 already in use, skipping MLflow"
else
    nohup mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns > mlflow.log 2>&1 &
    echo "   ✅ MLflow started on port 5000"
fi

sleep 2

# Start Airflow
echo ""
echo "2️⃣ Starting Airflow..."
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False

mkdir -p $AIRFLOW_HOME

if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "   Initializing Airflow database..."
    airflow db init > /dev/null 2>&1
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin > /dev/null 2>&1
fi

if check_port 8080; then
    echo "   ⚠️  Port 8080 already in use, skipping Airflow"
else
    nohup airflow webserver --port 8080 > airflow_webserver.log 2>&1 &
    nohup airflow scheduler > airflow_scheduler.log 2>&1 &
    echo "   ✅ Airflow started on port 8080"
fi

sleep 2

# Start Streamlit UI
echo ""
echo "3️⃣ Starting Streamlit UI..."
if check_port 8501; then
    echo "   ⚠️  Port 8501 already in use, skipping Streamlit"
else
    nohup streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
    echo "   ✅ Streamlit UI started on port 8501"
fi

echo ""
echo "===================================="
echo "✅ All services started!"
echo "===================================="
echo ""
echo "📊 Access the services at:"
echo "  • MLflow UI:     http://localhost:5000"
echo "  • Airflow UI:    http://localhost:8080 (admin/admin)"
echo "  • Streamlit UI:  http://localhost:8501"
echo ""
echo "📝 Logs:"
echo "  • MLflow:        mlflow.log"
echo "  • Airflow Web:   airflow_webserver.log"
echo "  • Airflow Sched: airflow_scheduler.log"
echo "  • Streamlit:     streamlit.log"
echo ""
echo "🛑 To stop all services:"
echo "   ./stop_all_services.sh"
echo ""
