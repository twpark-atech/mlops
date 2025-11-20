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

# Airflow 3.1.3 compatible import
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

from minio import Minio
from minio.error import S3Error

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Project imports
from src.common.config_loader import load_base_config
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


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def resolve_processing_date(context) -> tuple[datetime, str]:
    params = context["params"]

    override = params.get("processing_date")
    if override:
        dt = datetime.strptime(override, DATE_FMT)
    else:
        dt = context["logical_date"]

    return dt, dt.strftime(DATE_FMT)


def build_ingestion_url(params: Dict[str, Any], logical_date, date_str: str) -> str:
    template = params.get("ingestion_url_template", DEFAULT_URL_TEMPLATE)
    return template.format(
        date=date_str,
        year=logical_date.strftime("%Y"),
        month=logical_date.strftime("%m"),
        day=logical_date.strftime("%d"),
    )


def minio_partition_exists(minio_conf: Dict[str, Any], prefix: str, date_str: str) -> bool:
    client = Minio(
        minio_conf["endpoint"],
        access_key=minio_conf["access_key"],
        secret_key=minio_conf["secret_key"],
        secure=minio_conf.get("secure", False),
    )

    bucket = minio_conf["bucket"]
    pfx = f"{prefix.rstrip('/')}/date={date_str}/"

    try:
        obj = next(client.list_objects(bucket, prefix=pfx, recursive=True), None)
        return obj is not None
    except S3Error as exc:
        logger.warning("MinIO lookup failed (%s): %s", pfx, exc)
        return False


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

@dag(
    dag_id="its_traffic_full_pipeline_tf",
    description="ITS Traffic → MinIO → Spark → Postgres → ML Pipeline (Airflow 3.1.3 Compatible)",
    schedule="@daily",
    start_date=datetime(2025, 11, 13),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    params={
        "ingestion_url_template": DEFAULT_URL_TEMPLATE,
        "ingestion_job_name_prefix": "its_traffic_5min",
        "ingestion_lake_prefix": "traffic/raw",
        "training_window_days": 30,
        "training_job_name": "its_traffic_5min_convlstm",
        "training_overrides": {},
        "processing_date": datetime(2025, 11, 13).strftime(DATE_FMT),
    },
    tags=["its", "traffic", "mlops"],
)
def its_traffic_pipeline_tf():
    base_conf = load_base_config()

    # -------------------------------
    # 1) RAW ingestion
    # -------------------------------
    @task
    def ingest_raw(**context):
        params = context["params"]
        minio_conf = base_conf["minio"]
        lake_prefix = params["ingestion_lake_prefix"]

        logical_date, date_str = resolve_processing_date(context)
        url = build_ingestion_url(params, logical_date, date_str)

        # partition 존재 시 skip
        if minio_partition_exists(minio_conf, lake_prefix, date_str):
            logger.info("[SKIP] RAW partition already exists for %s", date_str)
            return date_str  # ✔ 무조건 date만 반환한다

        job_name = f"{params['ingestion_job_name_prefix']}_{date_str}"
        logger.info("[INGEST] %s → %s", url, job_name)

        ingest_from_url(job_name=job_name, url=url)

        return date_str  # ✔ dict 대신 문자열 반환

    # -------------------------------
    # 2) RAW → BRONZE
    # -------------------------------
    @task
    def raw_to_bronze(date: str):
        logger.info("[RAW→BRONZE] %s", date)
        run_its_traffic_raw_to_bronze(start_date=date, end_date=date)
        return date

    # -------------------------------
    # 3) BRONZE → SILVER
    # -------------------------------
    @task
    def bronze_to_silver(date: str):
        logger.info("[BRONZE→SILVER] %s", date)
        run_its_traffic_bronze_to_silver(start_date=date, end_date=date)
        return date

    # -------------------------------
    # 4) SILVER → GOLD
    # -------------------------------
    @task
    def silver_to_gold(date: str):
        logger.info("[SILVER→GOLD] %s", date)
        run_its_traffic_silver_to_gold(start_date=date, end_date=date)
        return date

    # -------------------------------
    # 5) ML Training
    # -------------------------------
    @task
    def train_model(date: str, **context):
        params = context["params"]
        logical_date, _ = resolve_processing_date(context)

        # window 계산
        window = max(int(params.get("training_window_days", 30)), 1)
        start_dt = (logical_date - timedelta(days=window - 1)).strftime(DATE_FMT)
        end_dt = date
        job_name = params["training_job_name"]

        cfg = TrainConfig(job_name=job_name, start_date=start_dt, end_date=end_dt)

        # override 적용
        for key, value in params.get("training_overrides", {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        logger.info("[TRAIN] %s window %s → %s", job_name, start_dt, end_dt)
        run_training(cfg)

    # -------------------------------
    # DAG FLOW
    # -------------------------------
    start = EmptyOperator(task_id="start")

    ingestion_date = ingest_raw()
    bronze = raw_to_bronze(ingestion_date)
    silver = bronze_to_silver(bronze)
    gold = silver_to_gold(silver)
    train_model(gold)

    start >> ingestion_date


its_traffic_pipeline_tf()
