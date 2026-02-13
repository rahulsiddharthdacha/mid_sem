from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from features.feature_extractor import extract_features

def parse_excel():
    df = extract_features("data/sample.xlsx")
    df.to_csv("features.csv", index=False)

with DAG(
    dag_id="excel_table_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    extract_task = PythonOperator(
        task_id="extract_excel_features",
        python_callable=parse_excel
    )

    extract_task
