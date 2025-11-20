# airflow/src/common/spark_session.py
import os
from pyspark.sql import SparkSession


def create_spark(app_name: str) -> SparkSession:
    s3_endpoint = os.getenv("S3_ENDPOINT", "minio:9000")
    s3_access_key = os.getenv("S3_ACCESS_KEY", "minio")
    s3_secret_key = os.getenv("S3_SECRET_KEY", "miniostorage")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "64"))
        .config("spark.sql.files.maxPartitionBytes", os.getenv("SPARK_MAX_PART_BYTES", "128m"))
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", s3_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", s3_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "org.postgresql:postgresql:42.7.4"
        )
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark