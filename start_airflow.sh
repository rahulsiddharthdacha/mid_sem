#!/bin/bash

echo "🚀 Starting Apache Airflow..."
echo "================================"

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# Create airflow_home directory
mkdir -p $AIRFLOW_HOME

# Check if Airflow DB is initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Initializing Airflow database..."
    airflow db init
    
    echo ""
    echo "Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
fi

echo ""
echo "Starting Airflow webserver on port 8080..."
airflow webserver --port 8080 -D

echo "Starting Airflow scheduler..."
airflow scheduler -D

echo ""
echo "✅ Airflow started!"
echo "================================"
echo "Web UI: http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "To stop Airflow:"
echo "  pkill -f 'airflow webserver'"
echo "  pkill -f 'airflow scheduler'"
