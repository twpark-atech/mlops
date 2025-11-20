# airflow/src/ingestion/file_ingestor/main.py
import io
import os
import logging
import zipfile
import requests
from typing import Any, Dict, List, Optional

from minio import Minio

from src.common.config_loader import load_base_config
from src.common.kafka_utils import get_kafka_producer, send_event
from src.common.logging_utils import setup_logging


logger = logging.getLogger(__name__)


def infer_date_from_url(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    if "_" in name:
        prefix = name.split("_")[0]
    else:
        prefix = name.split(".")[0]
    if len(prefix) == 8 and prefix.isdigit():
        return prefix
    return "unknown_date"


def download_zip_in_memory(url: str, timeout: int = 60) -> bytes:
    logger.info(f"Downloading Zip from: {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    size = len(resp.content)
    logger.info(f"Donwloaded {size:,} bytes from {url}")
    if size == 0:
        raise ValueError(f"Empty response returned from {url}. Cannot process ZIP.")
    return resp.content


def get_minio_client(conf: Dict[str, Any]) -> Minio:
    return Minio(
        conf["endpoint"],
        access_key=conf["access_key"],
        secret_key=conf["secret_key"],
        secure=conf.get("secure", False)
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        logger.info(f"Bucket '{bucket}' not found. Creating...")
        client.make_bucket(bucket)
    else:
        logger.info(f"Bucket '{bucket}' already exists.")
        

def upload_zip_to_datalake(
    zip_bytes: bytes,
    date_partition: str,
    lake_prefix: str,
    minio_conf: Dict[str, Any],
    kafka_conf: Dict[str, Any],
    kafka_topic: str
) -> None:
    minio_client = get_minio_client(minio_conf)
    bucket = minio_conf["bucket"]
    ensure_bucket(minio_client, bucket)

    producer = get_kafka_producer(kafka_conf["bootstrap_servers"])

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            logger.info(f"{len(members)} files found in ZIP.")

            for member in members:
                file_bytes = zf.read(member)
                object_name = f"{lake_prefix}/date={date_partition}/{member.filename}"

                logger.info(
                    f"Uploading to MinIO bucket={bucket}, key={object_name}, "
                    f"size={len(file_bytes):,} bytes"
                )
                
                data_stream = io.BytesIO(file_bytes)
                data_stream.seek(0)

                minio_client.put_object(
                    bucket_name=bucket,
                    object_name=object_name,
                    data=data_stream,
                    length=len(file_bytes),
                    content_type="application/octet-stream"
                )

                event = {
                    "date": date_partition,
                    "bucket": bucket,
                    "key": object_name,
                    "size": len(file_bytes)
                }
                send_event(producer, topic=kafka_topic, payload=event)

            producer.flush()
            logger.info("All files uploaded and events sent.")
    except zipfile.BadZipFile:
        logger.error(
            "Received invalid ZIP (date=%s, lake_prefix=%s).",
            date_partition,
            lake_prefix,
        )
        raise


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def run_job(job: Dict[str, Any], base_conf: Dict[str, Any]) -> None:
    url = job["url"]

    kafka_conf = base_conf.get("kafka", {})
    kafka_topic = job.get(
        "kafka_topic",
        os.getenv(
            "KAFKA_TOPIC_ITS_TRAFFIC_RAW",
            kafka_conf.get("topics", {}).get("its_traffic_raw", "its_traffic_raw")
        )
    )

    datalake_conf = base_conf.get("datalake", {})
    ingestion_conf = base_conf.get("ingestion", {})
    lake_prefix = job.get(
        "lake_prefix",
        os.getenv(
            "INGESTION_LAKE_PREFIX",
            ingestion_conf.get("lake_prefix", datalake_conf.get("prefix", "traffic/raw"))
        )
    )

    date_part = infer_date_from_url(url)
    logger.info(f"[{job['name']}] Inferred date partition: {date_part}")

    minio_conf = base_conf.get("minio", {})
    minio_settings = {
        "endpoint": os.getenv("MINIO_ENDPOINT", minio_conf.get("endpoint", "minio:9000")),
        "access_key": os.getenv("MINIO_ACCESS_KEY", minio_conf.get("access_key", "")),
        "secret_key": os.getenv("MINIO_SECRET_KEY", minio_conf.get("secret_key", "")),
        "secure": _env_bool("MINIO_SECURE", bool(minio_conf.get("secure", False))),
        "bucket": os.getenv("MINIO_BUCKET", minio_conf.get("bucket", "its")),        
    }

    kafka_bootstrap = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        kafka_conf.get("bootstrap_servers", "broker:29092")
    )

    zip_bytes = download_zip_in_memory(url)
    upload_zip_to_datalake(
        zip_bytes=zip_bytes,
        date_partition=date_part,
        lake_prefix=lake_prefix,
        minio_conf=minio_settings,
        kafka_conf={"bootstrap_servers": kafka_bootstrap},
        kafka_topic=kafka_topic
    )
    logger.info(f"[{job['name']}] Ingestion completed.")


def ingest_from_url(job_name: str, url: str) -> None:
    setup_logging()
    logger.info(f"Starting file ingestion from {url}...")

    base_conf = load_base_config()
    job = {"name": job_name, "url": url}
    run_job(job, base_conf)

    logger.info("All file ingestion jobs completed.")
