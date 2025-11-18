# src/ingestion/file_ingestor/main.py
import io
import logging
import zipfile
import requests
from src.common.logging_utils import setup_logging
from src.common.config_loader import load_base_config, load_ingestion_file_config
from src.common.kafka_utils import get_kafka_producer, send_event
from minio import Minio
from typing import Optional, Dict, Any, List

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
    logger.info(f"Downloaded {len(resp.content):,} bytes from {url}")
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

def run_job(job: Dict[str, Any], base_conf: Dict[str, Any]) -> None:
    url = job["url"]
    kafka_topic = job.get(
        "kafka_topic",
        base_conf["kafka"]["topics"]["its_traffic_raw"]
    )
    lake_prefix = job.get("lake_prefix", base_conf["datalake"]["prefix"])

    date_part = infer_date_from_url(url)
    logger.info(f"[{job['name']}] Inferred date partition: {date_part}")

    zip_bytes = download_zip_in_memory(url)
    upload_zip_to_datalake(
        zip_bytes=zip_bytes,
        date_partition=date_part,
        lake_prefix=lake_prefix,
        minio_conf={
            **base_conf["minio"],
            "bucket": base_conf["minio"]["bucket"]
        },
        kafka_conf={
            "bootstrap_servers": base_conf["kafka"]["bootstrap_servers"]
        },
        kafka_topic=kafka_topic
    )
    logger.info(f"[{job['name']}] Ingestion completed.")

def ingest_from_url(job_name: str, url: str) -> None:
    """
    Convenience wrapper so other components (e.g. Airflow) can trigger
    a single ingestion job without relying on the YAML fan-out.
    """
    base_conf = load_base_config()
    job = {"name": job_name, "url": url}
    run_job(job, base_conf)

def main(job_name: Optional[str] = None) -> None:
    setup_logging()
    logger.info("Starting file ingestion...")

    base_conf = load_base_config()
    ingest_conf = load_ingestion_file_config()
    jobs: List[Dict[str, Any]] = ingest_conf.get("jobs", [])

    if job_name:
        jobs = [j for j in jobs if j["name"] == job_name]

    if not jobs:
        logger.warning("No jobs found to run.")
        return
    
    for job in jobs:
        run_job(job, base_conf)
    
    logger.info("All file ingestion jobs completed.")

if __name__ == "__main__":
    main()
