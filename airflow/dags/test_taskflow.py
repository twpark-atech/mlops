from airflow import DAG
from airflow.decorators import task
from datetime import datetime

with DAG(
    dag_id="test_taskflow",
    start_date=datetime(2025, 1, 1),
    schedule=None,     # Airflow 3.x에서는 schedule 사용
    catchup=False,
    tags=["test"],
):

    @task
    def start_task():
        print("🔵 Start task 실행됨")
        return "hello airflow"

    @task
    def process_task(message: str):
        print(f"🟡 process_task 실행됨, message = {message}")
        return message.upper()

    @task
    def end_task(result: str):
        print(f"🟢 최종 결과: {result}")

    msg = start_task()
    processed = process_task(msg)
    end_task(processed)
