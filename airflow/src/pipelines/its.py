# airflow/src/pipelines/its.py
import os
import psycopg2
from datetime import datetime, timedelta
from pyspark.sql import functions as F 
from src.common.config_loader import load_base_config
from src.common.spark_session import create_spark


_BASE_CONF = load_base_config()
_DATALAKE_PATHS = _BASE_CONF.get("datalake", {}).get("paths", {})
_POSTGRES_CONF = _BASE_CONF.get("postgres", {})

LINKIDS_ALL = [
    "1920161400", "1920161500",
    "1920121301", "1920121401",
    "1920161902", "1920162205", "1920162400",
    "1920000702", "1920000801", "1920121000", "1920121302", "1920121402",
    "1920235801", "1920189001", "1920139400", "1920161801", "1920162207",
    "1920162304", "1920162500", "1920171200", "1920171600", "1920188900", "1920138500"
]


def _datalake_path(env_key: str, zone: str, default: str) -> str:
    return os.getenv(env_key, _DATALAKE_PATHS.get(zone, default))


BASE_RAW = _datalake_path("DATALAKE_RAW_PATH", "raw", "s3a://its/traffic/raw")
BASE_BRONZE = _datalake_path("DATALAKE_BRONZE_PATH", "bronze", "s3a://its/traffic/bronze")
BASE_SILVER = _datalake_path("DATALAKE_SILVER_PATH", "silver", "s3a://its/traffic/silver")


def its_raw_to_bronze(start_date: str, end_date: str) -> None:
    spark = create_spark("RAW_TO_BRONZE_ITS_TRAFFIC_5MIN")
    
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{BASE_RAW}/date={dt}/*.csv"
        output_path = f"{BASE_BRONZE}/date={dt}"
        
        print(f"[RAW→BRONZE] {dt} 읽는 중: {input_path}")

        try:
            df_raw = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(input_path)
            )

            df_clean = (
                df_raw
                .select(
                    F.col("CREATDE").cast("string"),
                    F.col("CREATHM").cast("string"),
                    F.col("LINKID").cast("string"),
                    F.col("ROADINSTTCD").cast("string"),
                    F.col("PASNGSPED").cast("double"),
                    F.col("PASNGTIME").cast("double")
                )
            )

            cnt = df_clean.count()
            print(f"[RAW→BRONZE] {dt} row 수: {cnt}")

            (
                df_clean
                .repartition(1)
                .write.mode("overwrite")
                .parquet(output_path)
            )
            print(f"[RAW→BRONZE] {dt} → {output_path} 저장 완료")

        except Exception as e:
            print(f"[RAW→BRONZE][WARN] {dt} 처리 실패: {e}")

        cur += timedelta(days=1)


def its_bronze_to_silver(start_date: str, end_date: str) -> None:
    spark = create_spark("BRONZE_TO_SILVER_ITS_TRAFFIC_5MIN")
    
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{BASE_BRONZE}/date={dt}"
        output_path = f"{BASE_SILVER}/date={dt}"
        
        print(f"[BRONZE→SILVER] {dt} 읽는 중: {input_path}")

        try:
            df_bronze = spark.read.parquet(input_path)

            df = (
                df_bronze
                .select(
                    F.col("CREATDE").cast("string"),
                    F.col("CREATHM").cast("string"),
                    F.col("LINKID").cast("string"),
                    F.col("PASNGSPED").cast("double")
                )
                .withColumn("CREATDE", F.lpad(F.col("CREATDE"), 8, "0"))
                .withColumn("CREATHM", F.lpad(F.col("CREATHM"), 4, "0"))
            )

            df = df.filter(F.col("LINKID").isin(LINKIDS_ALL))

            df = df.filter(
                (F.col("CREATDE").rlike(r"^\d{8}$")) &
                (F.col("CREATHM").rlike(r"^\d{4}$"))
            )

            df = df.withColumn(
                "datetime",
                F.to_timestamp(F.concat(F.col("CREATDE"), F.col("CREATHM")), "yyyyMMddHHmm")
            )

            df = df.withColumn(
                "minute",
                F.minute("datetime")
            ).withColumn(
                "minute_5",
                (F.col("minute") / 5).cast("int") * 5
            ).withColumn(
                "datetime_5min",
                F.concat_ws(
                    " ",
                    F.date_format("datetime", "yyyy-MM-dd"),
                    F.format_string(
                        "%02d:%02d:00",
                        F.hour("datetime"),
                        F.col("minute_5")
                    )
                ).cast("timestamp")
            )

            df = df.filter(F.col("CREATDE") == dt)

            df_silver = (
                df.groupBy("datetime_5min", "LINKID")
                .agg(F.avg("PASNGSPED").alias("self_mean"))
                .withColumnRenamed("LINKID", "linkid")
                .withColumnRenamed("datetime_5m", "datetime")
            )

            (
                df_silver
                .repartition(8, "datetime")
                .write.mode("overwrite")
                .parquet(output_path)
            )

            print(f"[BRONZE→SILVER] {dt} → {output_path} 저장 완료")

        except Exception as e:
            print(f"[BRONZE→SILVER][WARN] {dt} 처리 실패: {e}")

        cur += timedelta(days=1)


def _delete_existing_partition(
    dt: str,
    table: str,
    host: str,
    port: str,
    db: str,
    user: str,
    password: str,
    date_col: str = "date"
) -> None:
    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db,
            user=user,
            password=password
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = f'DELETE FROM "{table}" WHERE "{date_col}" = %s'
            cur.execute(sql, (dt, ))
        print(f"[SILVER→GOLD] {table}에서 date={dt} 기존 행 삭제 완료")
    except Exception as e:
        print(f"[SILVER→GOLD][WARN] date={dt} 삭제 중 오류 (무시하고 진행): {e}")
    finally:
        if conn is not None:
            conn.close()


def its_silver_to_gold(start_date: str, end_date: str) -> None:
    spark = create_spark("SILVER_TO_GOLD_ITS_TRAFFIC_5MIN")

    pg_host = os.getenv("PG_HOST", _POSTGRES_CONF.get("host", "postgres"))
    pg_port = os.getenv("PG_PORT", str(_POSTGRES_CONF.get("port", 5432)))
    pg_db = os.getenv("PG_DB", _POSTGRES_CONF.get("db", "mlops"))
    pg_user = os.getenv("PG_USER", _POSTGRES_CONF.get("user", "postgres"))
    pg_password = os.getenv("PG_PASSWORD", _POSTGRES_CONF.get("password", "postgres"))
    pg_table = os.getenv("ITS_TRAFFIC_GOLD_TABLE", _POSTGRES_CONF.get("table", "its_traffic_5min_gold"))

    jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"
    
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{BASE_SILVER}/date={dt}"
        
        print(f"[SILVER→GOLD] {dt} 읽는 중: {input_path}")

        try:
            df_silver = spark.read.parquet(input_path)

            cols = df_silver.columns

            rename_map = {}
            if "LINKID" in cols:
                rename_map["LINKID"] = "linkid"
            if "DATETIME_5MIN" in cols:
                rename_map["DATETIME_5MIN"] = "datetime"
            if "datetime_5min" in cols:
                rename_map["datetime_5min"] = "datetime"

            df_out = df_silver
            for src, dst in rename_map.items():
                df_out = df_out.withColumnRenamed(src, dst)
            
            df_out = df_out.withColumn("date", F.lit(dt).cast("string"))

            ordered_cols = []
            for c in ["date", "datetime", "linkid"]:
                if c in df_out.columns:
                    ordered_cols.append(c)
            ordered_cols += [c for c in df_out.columns if c not in ordered_cols]
            df_out = df_out.select(*ordered_cols)

            _delete_existing_partition(
                dt=dt,
                table=pg_table,
                host=pg_host,
                port=pg_port,
                db=pg_db,
                user=pg_user,
                password=pg_password,
                date_col="date"
            )

            (
                df_out.write
                .mode("append")
                .format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", pg_table)
                .option("user", pg_user)
                .option("password", pg_password)
                .option("driver", "org.postgresql.Driver")
                .save()
            )

            print(f"[SILVER→GOLD] {dt} → {pg_table} 저장 완료")

        except Exception as e:
            print(f"[SILVER→GOLD][WARN] {dt} 처리 실패: {e}")

        cur += timedelta(days=1)