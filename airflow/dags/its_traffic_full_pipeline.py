"""
Airflow DAG orchestrating ITS traffic ingestion → data lake curation → ML training.

Steps:
1. Download ITS ZIP → stream to MinIO raw zone + notify Kafka.
2. Spark jobs for RAW→BRONZE→SILVER conversions.
3. Publish SILVER aggregates to GOLD/PostgreSQL.
4. Pull GOLD data to train the ConvLSTM model.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ingestion.file_ingestor.main import ingest_from_url
from src.pipelines.raw_to_bronze import run_its_traffic_raw_to_bronze
from src.pipelines.bronze_to_silver import run_its_traffic_bronze_to_silver
from src.pipelines.silver_to_gold import run_its_traffic_silver_to_gold
from src.training.train_its_traffic_convlstm import TrainConfig, run_training

logger = logging.getLogger(__name__)

DATE_FMT = "%Y%m%d"
DEFAULT_URL_TEMPLATE = (
    "https://www.its.go.kr/opendata/fileDownload/traffic/{year}/{date}_5Min.zip"
)


def _resolve_pipeline_date(context) -> tuple[datetime, str]:
    """Return logical datetime and formatted string for the processing date."""
    params = context["params"]
    override = params.get("processing_date")
    if override:
        logical = datetime.strptime(override, DATE_FMT)
    else:
        logical = context["logical_date"]
    return logical, logical.strftime(DATE_FMT)


def _build_ingestion_url(
    params: Dict[str, Any],
    logical_date,
    date_str: str,
) -> str:
    template = params.get("ingestion_url_template", DEFAULT_URL_TEMPLATE)
    replacements = {
        "date": date_str,
        "year": logical_date.strftime("%Y"),
        "month": logical_date.strftime("%m"),
        "day": logical_date.strftime("%d"),
    }
    return template.format(**replacements)


def ingest_raw_to_datalake(**context):
    params = context["params"]
    logical_date, date_str = _resolve_pipeline_date(context)
    job_prefix = params.get("ingestion_job_name_prefix", "its_traffic_5min")
    url = _build_ingestion_url(params, logical_date, date_str)
    logger.info("Triggering ingestion for %s via %s", date_str, url)
    ingest_from_url(job_name=f"{job_prefix}_{date_str}", url=url)


def run_raw_to_bronze(**context):
    _, date_str = _resolve_pipeline_date(context)
    logger.info("Launching RAW→BRONZE for %s", date_str)
    run_its_traffic_raw_to_bronze(start_date=date_str, end_date=date_str)


def run_bronze_to_silver(**context):
    _, date_str = _resolve_pipeline_date(context)
    logger.info("Launching BRONZE→SILVER for %s", date_str)
    run_its_traffic_bronze_to_silver(start_date=date_str, end_date=date_str)


def run_silver_to_gold(**context):
    _, date_str = _resolve_pipeline_date(context)
    logger.info("Launching SILVER→GOLD for %s", date_str)
    run_its_traffic_silver_to_gold(start_date=date_str, end_date=date_str)


def train_convlstm(**context):
    params = context["params"]
    logical_date, end_date = _resolve_pipeline_date(context)
    window_days = int(params.get("training_window_days", 30))
    window_days = max(window_days, 1)
    start_dt = (logical_date - timedelta(days=window_days - 1)).strftime(DATE_FMT)
    job_name = params.get("training_job_name", "its_traffic_5min_convlstm")

    cfg = TrainConfig(job_name=job_name, start_date=start_dt, end_date=end_date)
    overrides: Dict[str, Any] = params.get("training_overrides", {})
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    logger.info(
        "Training %s with window %s→%s (overrides=%s)",
        job_name,
        start_dt,
        end_date,
        overrides,
    )
    run_training(cfg)


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="its_traffic_full_pipeline",
    description="E2E pipeline: Kafka ingestion → MinIO lake → Spark curation → Postgres + ML",
    schedule="@daily",
    start_date=datetime(2025, 11, 13),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["its", "traffic", "mlops"],
    params={
        "ingestion_url_template": DEFAULT_URL_TEMPLATE,
        "ingestion_job_name_prefix": "its_traffic_5min",
        "training_window_days": 30,
        "training_job_name": "its_traffic_5min_convlstm",
        "training_overrides": {},
        "processing_date": datetime(2025, 11, 13).strftime(DATE_FMT),
    },
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_raw_zip",
        python_callable=ingest_raw_to_datalake,
    )

    raw_to_bronze_task = PythonOperator(
        task_id="raw_to_bronze",
        python_callable=run_raw_to_bronze,
    )

    bronze_to_silver_task = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=run_bronze_to_silver,
    )

    silver_to_gold_task = PythonOperator(
        task_id="silver_to_gold",
        python_callable=run_silver_to_gold,
    )

    train_task = PythonOperator(
        task_id="train_convlstm",
        python_callable=train_convlstm,
    )

    ingest_task >> raw_to_bronze_task >> bronze_to_silver_task >> silver_to_gold_task >> train_task
