from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "member2",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="agri_pipeline",
    default_args=default_args,
    description="Smart Agriculture Data Pipeline",
    schedule="@daily",
    catchup=False,
    tags=["spark", "kafka", "agriculture"],
) as dag:

    kafka_producer = BashOperator(
        task_id="kafka_producer",
        bash_command="""
        cd /opt/airflow/project && python3 src/ingestion/kafka_producer.py
        """,
    )

    streaming_processing = BashOperator(
        task_id="streaming_processing",
        bash_command="""
        cd /opt/airflow/project && python3 src/spark/streaming_processing.py
        """,
    )

    analytics_sql = BashOperator(
        task_id="analytics_sql",
        bash_command="""
        cd /opt/airflow/project && python3 src/spark/analytics_sql.py
        """,
    )

    model_training = BashOperator(
        task_id="model_training",
        bash_command="""
        cd /opt/airflow/project && python3 src/ml/train_mllib.py
        """,
    )

    kafka_producer >> streaming_processing >> analytics_sql >> model_training
