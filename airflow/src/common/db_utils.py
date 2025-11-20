# airflow/src/common/db_utils.py
import psycopg2
import pandas as pd
from minio import Minio


def _get_minio_client(cfg) -> Minio:
    client = Minio(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=cfg.minio_secure,
    )
    if not client.bucket_exists(cfg.minio_bucket):
        client.make_bucket(cfg.minio_bucket)
    return client


def _upload_bytes_to_minio(
    client: Minio,
    cfg,
    data_bytes: bytes,
    object_name: str,
) -> str:
    data_stream = io.BytesIO(data_bytes)
    length = len(data_bytes)

    client.put_object(
        bucket_name=cfg.minio_bucket,
        object_name=object_name,
        data=data_stream,
        length=length,
    )

    return f"s3a://{cfg.minio_bucket}/{object_name}"


def load_gold_from_postgres(cfg, query) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=cfg.pg_host,
        port=cfg.pg_port,
        dbname=cfg.pg_db,
        user=cfg.pg_user,
        password=cfg.pg_password
    )

    try:
        params = (cfg.start_date, cfg.end_date)
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()

    return df