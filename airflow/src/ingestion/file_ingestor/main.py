# airflow/src/ingestion/file_ingestor/main.py
import io
import os
import logging
import zipfile
import requests
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1] or "payload"


def download_bytes(url: str, timeout: int = 60) -> bytes:
    logger.info(f"Downloading from: {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    size = len(resp.content)
    logger.info(f"Downloaded {size:,} bytes from {url}")
    if size == 0:
        raise ValueError(f"Empty response returned from {url}. Cannot process.")
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


def _build_object_name(
    template: Optional[str],
    lake_prefix: str,
    date_partition: Optional[str],
    filename: str
) -> str:
    lake_prefix = lake_prefix.rstrip("/")
    default_tpl = (
        "{lake_prefix}/date={date}/{filename}"
        if date_partition and date_partition != "unknown_date"
        else "{lake_prefix}/{filename}"
    )
    tpl = template or default_tpl
    return tpl.format(
        lake_prefix=lake_prefix,
        date=date_partition or "unknown_date",
        filename=filename
    )


def _upload_objects(
    minio_conf: Dict[str, Any],
    kafka_conf: Dict[str, Any],
    kafka_topic: str,
    objects: Iterable[Tuple[str, bytes]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    minio_client = get_minio_client(minio_conf)
    bucket = minio_conf["bucket"]
    ensure_bucket(minio_client, bucket)

    producer = get_kafka_producer(kafka_conf["bootstrap_servers"])
    for object_name, file_bytes in objects:
        logger.info(
            "Uploading to MinIO bucket=%s, key=%s, size=%s bytes",
            bucket,
            object_name,
            f"{len(file_bytes):,}",
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
            "bucket": bucket,
            "key": object_name,
            "size": len(file_bytes),
        }
        if metadata:
            event.update(metadata)
        send_event(producer, topic=kafka_topic, payload=event)

    producer.flush()
    logger.info("All files uploaded and events sent.")


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _expand_zip(
    zip_bytes: bytes,
    lake_prefix: str,
    date_partition: str,
    object_key_template: Optional[str]
) -> List[Tuple[str, bytes]]:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            logger.info("%s files found in ZIP.", len(members))
            outputs = []
            for member in members:
                file_bytes = zf.read(member)
                object_name = _build_object_name(
                    object_key_template,
                    lake_prefix,
                    date_partition,
                    member.filename
                )
                outputs.append((object_name, file_bytes))
            return outputs
    except zipfile.BadZipFile:
        logger.error("Received invalid ZIP (date=%s, lake_prefix=%s).", date_partition, lake_prefix)
        raise


def _wrap_single_file(
    file_bytes: bytes,
    lake_prefix: str,
    date_partition: str,
    filename: str,
    object_key_template: Optional[str]
) -> List[Tuple[str, bytes]]:
    object_name = _build_object_name(object_key_template, lake_prefix, date_partition, filename)
    return [(object_name, file_bytes)]


def run_job(job: Dict[str, Any], base_conf: Dict[str, Any]) -> None:
    url = job["url"]

    kafka_conf = base_conf.get("kafka", {})
    kafka_topic = job.get(
        "kafka_topic",
        os.getenv(
            "KAFKA_TOPIC_RAW",
            os.getenv(
                "KAFKA_TOPIC_ITS_TRAFFIC_RAW",
                kafka_conf.get("topics", {}).get("its_traffic_raw", "its_traffic_raw"),
            ),
        )
    )

    datalake_conf = base_conf.get("datalake", {})
    ingestion_conf = base_conf.get("ingestion", {})

    file_type = job.get(
        "file_type",
        os.getenv(
            "INGESTION_FILE_TYPE",
            ingestion_conf.get("file_type", "zip")
        )
    ).lower()

    lake_prefix = job.get(
        "lake_prefix",
        os.getenv(
            "INGESTION_LAKE_PREFIX",
            ingestion_conf.get("lake_prefix", datalake_conf.get("prefix", "traffic/raw"))
        )
    )

    minio_conf = base_conf.get("minio", {})
    minio_settings = {
        "endpoint": os.getenv("MINIO_ENDPOINT", minio_conf.get("endpoint", "minio:9000")),
        "access_key": os.getenv("MINIO_ACCESS_KEY", minio_conf.get("access_key", "")),
        "secret_key": os.getenv("MINIO_SECRET_KEY", minio_conf.get("secret_key", "")),
        "secure": _env_bool("MINIO_SECURE", bool(minio_conf.get("secure", False))),
        "bucket": job.get(
            "bucket",
            os.getenv("MINIO_BUCKET", minio_conf.get("bucket", "its"))
        ),
    }

    date_part = job.get("date") or infer_date_from_url(url)
    logger.info(f"[{job['name']}] Inferred date partition: {date_part}")

    kafka_bootstrap = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        kafka_conf.get("bootstrap_servers", "broker:29092")
    )

    file_bytes = download_bytes(url)
    object_key_template = job.get("object_key_template")
    if file_type == "zip":
        objects = _expand_zip(
            zip_bytes=file_bytes,
            lake_prefix=lake_prefix,
            date_partition=date_part,
            object_key_template=object_key_template
        )
    else:
        filename = job.get("filename") or _filename_from_url(url)
        objects = _wrap_single_file(
            file_bytes=file_bytes,
            lake_prefix=lake_prefix,
            date_partition=date_part,
            filename=filename,
            object_key_template=object_key_template
        )

    extra_meta = {"date": date_part, "file_type": file_type, "source_url": url}

    _upload_objects(
        minio_conf=minio_settings,
        kafka_conf={"bootstrap_servers": kafka_bootstrap},
        kafka_topic=kafka_topic,
        objects=objects,
        metadata=extra_meta,
    )
    logger.info(f"[{job['name']}] Ingestion completed.")


def ingest_from_url(
    job_name: str,
    url: str,
    *,
    file_type: Optional[str] = None,
    lake_prefix: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    bucket: Optional[str] = None,
    object_key_template: Optional[str] = None,
    date: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    setup_logging()
    logger.info(f"Starting file ingestion from {url}...")

    base_conf = load_base_config()
    job = {
        "name": job_name,
        "url": url,
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

    run_job(job, base_conf)

    logger.info("All file ingestion jobs completed.")
