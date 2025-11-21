# airflow/dags/its_pipeline.py
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

from minio import Minio
from minio.error import S3Error

# ----------------------------------------------------------------------
# PYTHONPATH 설정
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.config_loader import load_base_config
from src.ingestion.file_ingestor.main import ingest_from_url
from src.pipelines.its import its_raw_to_bronze, its_bronze_to_silver, its_silver_to_gold
from src.training.its import TrainConfig, run_training

logger = logging.getLogger(__name__)

DATE_FMT = "%Y%m%d"

# ----------------------------------------------------------------------
# 기본 설정 로드
# ----------------------------------------------------------------------
_DEFAULT_CONF = load_base_config()
_INGESTION_DEFAULTS = _DEFAULT_CONF.get("ingestion", {})
_TRAINING_DEFAULTS = _DEFAULT_CONF.get("training", {})
_DATALAKE_DEFAULTS = _DEFAULT_CONF.get("datalake", {})

DEFAULT_URL_TEMPLATE = _INGESTION_DEFAULTS.get(
    "url_template",
    "https://www.its.go.kr/opendata/fileDownload/traffic/{year}/{date}_5Min.zip",
)
DEFAULT_JOB_NAME_PREFIX = _INGESTION_DEFAULTS.get("job_name_prefix", "its_traffic_5min")
DEFAULT_LAKE_PREFIX = _INGESTION_DEFAULTS.get(
    "lake_prefix",
    _DATALAKE_DEFAULTS.get("prefix", "traffic/raw"),
)
DEFAULT_TRAINING_JOB_NAME = _TRAINING_DEFAULTS.get(
    "job_name",
    "its_traffic_5min_convlstm",
)

# 학습 valid 구간(마지막 N일) 기본값: env → config → 1일
DEFAULT_VALID_DAYS = int(
    os.getenv(
        "TRAINING_VALID_DAYS",
        str(_TRAINING_DEFAULTS.get("valid_days", 1)),
    )
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _parse_start_date(raw: str | None, fallback: datetime) -> datetime:
    if not raw:
        return fallback
    return datetime.strptime(raw, "%Y-%m-%d")


# ----------------------------------------------------------------------
# Airflow DAG 기본 설정
# ----------------------------------------------------------------------
SCHEDULE_INTERVAL = os.getenv("ITS_DAG_SCHEDULE", "@daily")

# 파이프라인 기준 시작일 (env 없으면 2025-11-01)
START_DATE = _parse_start_date(
    os.getenv("AIRFLOW_ITS_START_DATE"),
    datetime(2025, 11, 1),
)

CATCHUP_ENABLED = _env_bool("CATCHUP_ENABLED", True)
MAX_ACTIVE_RUNS = int(os.getenv("ITS_DAG_MAX_ACTIVE_RUNS", "1"))

# 파이프라인 범위 기반 시작일 (날짜 리스트 생성 시 사용)
PIPELINE_BASE_DATE = START_DATE.date()


def build_ingestion_url(params: Dict[str, Any], logical_date: datetime, date_str: str) -> str:
    """
    주어진 date_str(YYYYMMDD)에 대해 ingestion URL을 생성한다.
    logical_date는 URL template에서 {year}/{month}/{day} 치환에 사용.
    """
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


# ----------------------------------------------------------------------
# DAG 정의
# ----------------------------------------------------------------------
@dag(
    dag_id="its_traffic_full_pipeline",
    description="ITS Traffic → MinIO → Spark → Postgres → ML Pipeline (range-based)",
    schedule=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    catchup=CATCHUP_ENABLED,
    max_active_runs=MAX_ACTIVE_RUNS,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    params={
        "ingestion_url_template": DEFAULT_URL_TEMPLATE,
        "ingestion_job_name_prefix": DEFAULT_JOB_NAME_PREFIX,
        "ingestion_lake_prefix": DEFAULT_LAKE_PREFIX,
        "training_job_name": DEFAULT_TRAINING_JOB_NAME,
        "training_overrides": {},
        "valid_days": DEFAULT_VALID_DAYS,
        # 필요하면 processing_date 등 추가 파라미터를 여기에 붙인다.
    },
    tags=["its", "traffic", "mlops"],
)
def its_traffic_full_pipeline():
    """
    한 번 실행 시:
      1) PIPELINE_BASE_DATE ~ (logical_date - 1일)까지 RAW 수집 (이미 있으면 skip)
      2) 같은 범위를 BRONZE → SILVER → GOLD로 전처리
      3) 전체 구간 중 마지막 N일을 valid로 떼고 나머지로 학습
    """
    base_conf = load_base_config()

    # -------------------------------
    # 0) 날짜 리스트 생성
    # -------------------------------
    @task
    def build_date_list(**context) -> List[str]:
        """
        PIPELINE_BASE_DATE ~ (logical_date - 1일)까지의 날짜 리스트(YYYYMMDD)를 생성.
        예: PIPELINE_BASE_DATE=2025-11-01, logical_date=2025-11-22 → 20251101~20251121
        """
        logical_date: datetime = context["logical_date"]
        end_date = logical_date.date() - timedelta(days=1)
        start_date = PIPELINE_BASE_DATE

        if start_date > end_date:
            logger.warning(
                "[DATE_RANGE] start_date(%s) > end_date(%s). No dates to process.",
                start_date,
                end_date,
            )
            return []

        dates: List[str] = []
        cur = start_date
        while cur <= end_date:
            dates.append(cur.strftime(DATE_FMT))
            cur += timedelta(days=1)

        logger.info(
            "[DATE_RANGE] %s → %s (%d days)",
            dates[0],
            dates[-1],
            len(dates),
        )
        return dates

    # -------------------------------
    # 1) RAW ingestion (범위 기반)
    # -------------------------------
    @task
    def ingest_all_raw(dates: List[str], **context) -> List[str]:
        """
        날짜 리스트 전체에 대해:
          - MinIO에 RAW 파티션이 없으면 다운로드 & 업로드
          - 있으면 skip
        모든 날짜에 대해 RAW 존재를 보장한 뒤, 원본 dates를 그대로 반환.
        """
        if not dates:
            logger.warning("[INGEST] No dates to ingest. Skipping RAW ingestion.")
            return []

        params = context["params"]
        minio_conf = base_conf["minio"]
        lake_prefix = params["ingestion_lake_prefix"]

        ingested_count = 0
        skipped_count = 0

        for date_str in dates:
            if minio_partition_exists(minio_conf, lake_prefix, date_str):
                logger.info("[INGEST][SKIP] RAW already exists for %s", date_str)
                skipped_count += 1
                continue

            # URL 생성용 logical_date
            dt = datetime.strptime(date_str, DATE_FMT)
            url = build_ingestion_url(params, dt, date_str)

            job_name = f"{params['ingestion_job_name_prefix']}_{date_str}"
            logger.info("[INGEST] %s → %s", url, job_name)
            ingest_from_url(job_name=job_name, url=url)
            ingested_count += 1

        logger.info(
            "[INGEST] Completed. Total=%d, ingested=%d, skipped(existing)=%d",
            len(dates),
            ingested_count,
            skipped_count,
        )
        # 전처리는 모든 날짜에 대해 수행해야 하므로 원본 dates 반환
        return dates

    # -------------------------------
    # 2) RAW → BRONZE (범위 기반)
    # -------------------------------
    @task
    def raw_to_bronze_all(dates: List[str]) -> List[str]:
        if not dates:
            logger.warning("[BRONZE] No dates to process. Skipping RAW→BRONZE.")
            return []

        for date_str in dates:
            logger.info("[RAW→BRONZE] %s", date_str)
            its_raw_to_bronze(start_date=date_str, end_date=date_str)
        return dates

    # -------------------------------
    # 3) BRONZE → SILVER (범위 기반)
    # -------------------------------
    @task
    def bronze_to_silver_all(dates: List[str]) -> List[str]:
        if not dates:
            logger.warning("[SILVER] No dates to process. Skipping BRONZE→SILVER.")
            return []

        for date_str in dates:
            logger.info("[BRONZE→SILVER] %s", date_str)
            its_bronze_to_silver(start_date=date_str, end_date=date_str)
        return dates

    # -------------------------------
    # 4) SILVER → GOLD (범위 기반)
    # -------------------------------
    @task
    def silver_to_gold_all(dates: List[str]) -> List[str]:
        if not dates:
            logger.warning("[GOLD] No dates to process. Skipping SILVER→GOLD.")
            return []

        for date_str in dates:
            logger.info("[SILVER→GOLD] %s", date_str)
            its_silver_to_gold(start_date=date_str, end_date=date_str)
        return dates

    # -------------------------------
    # 5) ML Training (범위 기반: 마지막 N일 valid)
    # -------------------------------
    @task
    def train_model_full(dates: List[str], **context):
        if not dates:
            logger.warning("[TRAIN] No dates to train on. Skipping training.")
            return

        params = context["params"]
        valid_days = int(params.get("valid_days", DEFAULT_VALID_DAYS))
        valid_days = max(valid_days, 1)

        if len(dates) <= valid_days:
            logger.warning(
                "[TRAIN] Not enough dates (%d) for train/valid split (valid_days=%d). Skipping.",
                len(dates),
                valid_days,
            )
            return

        # 정렬되어 있다고 가정: [d0, d1, ..., dN]
        train_dates = dates[:-valid_days]
        valid_dates = dates[-valid_days:]

        train_start = train_dates[0]
        train_end = train_dates[-1]
        valid_start = valid_dates[0]
        valid_end = valid_dates[-1]

        job_name = params["training_job_name"]

        # TrainConfig는 start_date/end_date만 필수로 사용하고,
        # valid 구간 정보는 있으면 필드로 세팅 (없으면 무시)하는 방식으로 처리
        cfg = TrainConfig(
            job_name=job_name,
            start_date=train_start,
            end_date=train_end,
        )

        # training_overrides + train/valid 구간 정보 합쳐서 덮어쓰기
        overrides: Dict[str, Any] = dict(params.get("training_overrides", {}))
        overrides.setdefault("valid_start_date", valid_start)
        overrides.setdefault("valid_end_date", valid_end)
        overrides.setdefault("valid_days", valid_days)

        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        logger.info(
            "[TRAIN] %s train %s→%s, valid %s→%s (valid_days=%d)",
            job_name,
            train_start,
            train_end,
            valid_start,
            valid_end,
            valid_days,
        )
        run_training(cfg)

    # -------------------------------
    # DAG FLOW
    # -------------------------------
    start = EmptyOperator(task_id="start")

    date_list = build_date_list()
    ingested = ingest_all_raw(date_list)
    bronze = raw_to_bronze_all(ingested)
    silver = bronze_to_silver_all(bronze)
    gold = silver_to_gold_all(silver)
    train_model_full(gold)

    start >> date_list


its_traffic_full_pipeline()