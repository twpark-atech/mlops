"""
Generic API ingestor that fetches data from an HTTP endpoint and stores the raw
response into MinIO. All key knobs are passed in from the DAG (method, headers,
params, bucket, object naming, Kafka topic, etc.) and fall back to base config
or environment variables for compatibility.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
from typing import Any, Dict, Optional

import requests
from minio import Minio

from src.common.config_loader import load_base_config
from src.common.kafka_utils import get_kafka_producer, send_event
from src.common.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _filename_from_url(url: str, fallback: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return name or fallback


def _build_object_name(
    template: Optional[str],
    lake_prefix: str,
    date_partition: str,
    filename: str,
) -> str:
    lake_prefix = lake_prefix.rstrip("/")
    tpl = template or "{lake_prefix}/date={date}/{filename}"
    return tpl.format(
        lake_prefix=lake_prefix,
        date=date_partition,
        filename=filename,
    )


def _get_minio_client(conf: Dict[str, Any]) -> Minio:
    return Minio(
        conf["endpoint"],
        access_key=conf["access_key"],
        secret_key=conf["secret_key"],
        secure=conf.get("secure", False),
    )


def _ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        logger.info("Bucket '%s' not found. Creating...", bucket)
        client.make_bucket(bucket)
    else:
        logger.info("Bucket '%s' already exists.", bucket)


def _upload_bytes_to_minio(
    minio_conf: Dict[str, Any],
    object_name: str,
    file_bytes: bytes,
) -> None:
    client = _get_minio_client(minio_conf)
    bucket = minio_conf["bucket"]
    _ensure_bucket(client, bucket)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type="application/octet-stream",
    )
    logger.info(
        "Uploaded to MinIO bucket=%s, key=%s, size=%s bytes",
        bucket,
        object_name,
        f"{len(file_bytes):,}",
    )


def _maybe_send_kafka(
    kafka_conf: Dict[str, Any],
    topic: Optional[str],
    payload: Dict[str, Any],
) -> None:
    if not topic:
        return
    producer = get_kafka_producer(kafka_conf["bootstrap_servers"])
    send_event(producer, topic=topic, payload=payload, flush=True)


def _infer_date(job_date: Optional[str]) -> str:
    if job_date:
        return job_date
    return dt.datetime.utcnow().strftime("%Y%m%d")


def _infer_filename(
    url: str,
    job_name: str,
    filename: Optional[str],
    content_type: str,
) -> str:
    if filename:
        return filename
    ext = ""
    if "json" in content_type.lower():
        ext = ".json"
    elif "csv" in content_type.lower():
        ext = ".csv"
    elif "xml" in content_type.lower():
        ext = ".xml"
    base = _filename_from_url(url, job_name)
    if ext and not base.lower().endswith(ext):
        return f"{base}{ext}"
    return base


def run_job(job: Dict[str, Any], base_conf: Dict[str, Any]) -> None:
    url = job["url"]
    job_name = job["name"]

    kafka_conf = base_conf.get("kafka", {})
    kafka_topic = job.get(
        "kafka_topic",
        os.getenv(
            "KAFKA_TOPIC_API_RAW",
            os.getenv(
                "KAFKA_TOPIC_RAW",
                kafka_conf.get("topics", {}).get("api_raw", "api_raw"),
            ),
        ),
    )

    datalake_conf = base_conf.get("datalake", {})
    ingestion_conf = base_conf.get("ingestion", {})

    lake_prefix = job.get(
        "lake_prefix",
        os.getenv(
            "INGESTION_LAKE_PREFIX",
            ingestion_conf.get("lake_prefix", datalake_conf.get("prefix", "traffic/raw")),
        ),
    )

    file_type = job.get(
        "file_type",
        os.getenv("INGESTION_FILE_TYPE", ingestion_conf.get("file_type", "json")),
    )

    minio_conf = base_conf.get("minio", {})
    minio_settings = {
        "endpoint": os.getenv("MINIO_ENDPOINT", minio_conf.get("endpoint", "minio:9000")),
        "access_key": os.getenv("MINIO_ACCESS_KEY", minio_conf.get("access_key", "")),
        "secret_key": os.getenv("MINIO_SECRET_KEY", minio_conf.get("secret_key", "")),
        "secure": _env_bool("MINIO_SECURE", bool(minio_conf.get("secure", False))),
        "bucket": job.get(
            "bucket",
            os.getenv("MINIO_BUCKET", minio_conf.get("bucket", "its")),
        ),
    }

    method = job.get("method", "GET").upper()
    headers = job.get("headers")
    params = job.get("params")
    body = job.get("body")
    timeout = int(job.get("timeout", os.getenv("INGESTION_TIMEOUT_SEC", 60)))
    date_str = _infer_date(job.get("date"))
    object_key_template = job.get("object_key_template") or ingestion_conf.get("object_key_template")

    logger.info("[%s] Requesting %s %s ...", job_name, method, url)
    resp = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        data=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    file_bytes = resp.content
    if len(file_bytes) == 0:
        raise ValueError(f"Empty response from {url}.")
    content_type = resp.headers.get("Content-Type", "application/octet-stream")
    filename = _infer_filename(url, job_name, job.get("filename"), content_type)
    object_name = _build_object_name(object_key_template, lake_prefix, date_str, filename)

    _upload_bytes_to_minio(minio_settings, object_name, file_bytes)

    meta = {
        "job_name": job_name,
        "url": url,
        "date": date_str,
        "bucket": minio_settings["bucket"],
        "key": object_name,
        "content_type": content_type,
        "status_code": resp.status_code,
        "file_type": file_type,
    }
    _maybe_send_kafka({"bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", kafka_conf.get("bootstrap_servers", "broker:29092"))}, kafka_topic, meta)
    logger.info("[%s] Ingestion completed.", job_name)


def ingest_from_api(
    job_name: str,
    url: str,
    *,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    body: Optional[Any] = None,
    file_type: Optional[str] = None,
    lake_prefix: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    bucket: Optional[str] = None,
    object_key_template: Optional[str] = None,
    date: Optional[str] = None,
    filename: Optional[str] = None,
    timeout: Optional[int] = None,
) -> None:
    """
    Entry point used from DAGs. All parameters are optional and override config/env.
    """
    setup_logging()
    logger.info("Starting API ingestion: %s", url)

    base_conf = load_base_config()
    job: Dict[str, Any] = {
        "name": job_name,
        "url": url,
        "method": method,
        "params": params,
        "headers": headers,
        "body": body,
    }
    if file_type:
        job["file_type"] = file_type
    if lake_prefix:
        job["lake_prefix"] = lake_prefix
    if kafka_topic:
        job["kafka_topic"] = kafka_topic
    if bucket:
        job["bucket"] = bucket
    if object_key_template:
        job["object_key_template"] = object_key_template
    if date:
        job["date"] = date
    if filename:
        job["filename"] = filename
    if timeout is not None:
        job["timeout"] = timeout

    run_job(job, base_conf)


def main() -> None:
    raise SystemExit("Use ingest_from_api from DAG code.")


if __name__ == "__main__":
    main()
