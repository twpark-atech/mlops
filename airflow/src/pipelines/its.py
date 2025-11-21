# airflow/src/pipelines/its.py
import os
import psycopg2
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence
from pyspark.sql import functions as F 
from src.common.config_loader import load_base_config
from src.common.spark_session import create_spark


_BASE_CONF = load_base_config()
_DATALAKE_PATHS = _BASE_CONF.get("datalake", {}).get("paths", {})
_PIPELINE_CONF = _BASE_CONF.get("traffic_pipeline", {})
_PIPELINE_PATHS = _PIPELINE_CONF.get("paths", {})
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
    zone_default = _PIPELINE_PATHS.get(zone, _DATALAKE_PATHS.get(zone, default))
    return os.getenv(env_key, zone_default)


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_list(var_name: str, default: Optional[Iterable[str]]) -> Optional[List[str]]:
    raw = os.getenv(var_name)
    if raw is None:
        return list(default) if default else None
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    return parsed if parsed else None


@dataclass(frozen=True)
class RawSchema:
    date_col: str = "CREATDE"
    time_col: str = "CREATHM"
    link_id_col: str = "LINKID"
    speed_col: str = "PASNGSPED"
    extra_numeric_cols: Sequence[str] = field(default_factory=lambda: ("PASNGTIME",))
    rename_map: Dict[str, str] = field(default_factory=dict)

    def normalize(self, df):
        """Select and rename raw columns to canonical names: date, time, linkid, speed."""
        required = [self.date_col, self.time_col, self.link_id_col, self.speed_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Required columns missing from raw dataframe: {missing}")

        selects = [
            F.col(self.date_col).cast("string").alias("date"),
            F.col(self.time_col).cast("string").alias("time"),
            F.col(self.link_id_col).cast("string").alias("linkid"),
            F.col(self.speed_col).cast("double").alias("speed"),
        ]

        for src in self.extra_numeric_cols:
            if src not in df.columns:
                continue
            dst = self.rename_map.get(src, src)
            selects.append(F.col(src).cast("double").alias(dst))

        return df.select(*selects)


@dataclass(frozen=True)
class TrafficPipelineConfig:
    raw_schema: RawSchema = field(default_factory=RawSchema)
    link_ids: Optional[List[str]] = field(default_factory=list)
    filter_link_ids: bool = True
    base_raw: str = field(default_factory=lambda: "s3a://its/traffic/raw")
    base_bronze: str = field(default_factory=lambda: "s3a://its/traffic/bronze")
    base_silver: str = field(default_factory=lambda: "s3a://its/traffic/silver")
    postgres_table: str = "its_traffic_5min_gold"

    def should_filter_links(self) -> bool:
        return self.filter_link_ids and bool(self.link_ids)


def _build_raw_schema() -> RawSchema:
    raw_cols = _PIPELINE_CONF.get("raw_columns", {})
    rename_map = raw_cols.get("rename_map", {})
    extra_cols = raw_cols.get("extra_numeric", ("PASNGTIME",))
    return RawSchema(
        date_col=raw_cols.get("date", "CREATDE"),
        time_col=raw_cols.get("time", "CREATHM"),
        link_id_col=raw_cols.get("link_id", "LINKID"),
        speed_col=raw_cols.get("speed", "PASNGSPED"),
        extra_numeric_cols=extra_cols,
        rename_map=rename_map,
    )


def _build_pipeline_config() -> TrafficPipelineConfig:
    schema = _build_raw_schema()
    link_ids = _PIPELINE_CONF.get("link_ids", LINKIDS_ALL)
    link_ids = _env_list("TRAFFIC_LINK_IDS", link_ids)
    filter_links = _env_bool("TRAFFIC_FILTER_LINK_IDS", True)

    base_raw_default = _PIPELINE_PATHS.get("raw", "s3a://its/traffic/raw")
    base_bronze_default = _PIPELINE_PATHS.get("bronze", "s3a://its/traffic/bronze")
    base_silver_default = _PIPELINE_PATHS.get("silver", "s3a://its/traffic/silver")

    pg_table = (
        os.getenv("PIPELINE_GOLD_TABLE")
        or os.getenv("ITS_TRAFFIC_GOLD_TABLE")
        or _PIPELINE_CONF.get("gold_table")
        or _POSTGRES_CONF.get("table", "its_traffic_5min_gold")
    )

    return TrafficPipelineConfig(
        raw_schema=schema,
        link_ids=link_ids,
        filter_link_ids=filter_links,
        base_raw=_datalake_path("DATALAKE_RAW_PATH", "raw", base_raw_default),
        base_bronze=_datalake_path("DATALAKE_BRONZE_PATH", "bronze", base_bronze_default),
        base_silver=_datalake_path("DATALAKE_SILVER_PATH", "silver", base_silver_default),
        postgres_table=pg_table,
    )


PIPELINE_CONFIG = _build_pipeline_config()


BASE_RAW = PIPELINE_CONFIG.base_raw
BASE_BRONZE = PIPELINE_CONFIG.base_bronze
BASE_SILVER = PIPELINE_CONFIG.base_silver


def _resolve_column(df, *candidates: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the candidate columns exist in dataframe: {candidates}")


def its_raw_to_bronze(start_date: str, end_date: str, config: TrafficPipelineConfig | None = None) -> None:
    cfg = config or PIPELINE_CONFIG
    spark = create_spark("RAW_TO_BRONZE_ITS_TRAFFIC_5MIN")
    
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{cfg.base_raw}/date={dt}/*.csv"
        output_path = f"{cfg.base_bronze}/date={dt}"
        
        print(f"[RAW→BRONZE] {dt} 읽는 중: {input_path}")

        try:
            df_raw = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(input_path)
            )

            df_clean = cfg.raw_schema.normalize(df_raw)

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


def its_bronze_to_silver(start_date: str, end_date: str, config: TrafficPipelineConfig | None = None) -> None:
    cfg = config or PIPELINE_CONFIG
    spark = create_spark("BRONZE_TO_SILVER_ITS_TRAFFIC_5MIN")
    
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{cfg.base_bronze}/date={dt}"
        output_path = f"{cfg.base_silver}/date={dt}"
        
        print(f"[BRONZE→SILVER] {dt} 읽는 중: {input_path}")

        try:
            df_bronze = spark.read.parquet(input_path)

            date_col = _resolve_column(df_bronze, "date", cfg.raw_schema.date_col, "CREATDE")
            time_col = _resolve_column(df_bronze, "time", cfg.raw_schema.time_col, "CREATHM")
            link_col = _resolve_column(df_bronze, "linkid", cfg.raw_schema.link_id_col, "LINKID")
            speed_col = _resolve_column(df_bronze, "speed", cfg.raw_schema.speed_col, "PASNGSPED")

            df = (
                df_bronze.select(
                    F.col(date_col).cast("string").alias("date"),
                    F.col(time_col).cast("string").alias("time"),
                    F.col(link_col).cast("string").alias("linkid"),
                    F.col(speed_col).cast("double").alias("speed")
                )
                .withColumn("date", F.lpad(F.col("date"), 8, "0"))
                .withColumn("time", F.lpad(F.col("time"), 4, "0"))
            )

            df = df.filter(
                (F.col("date").rlike(r"^\d{8}$")) &
                (F.col("time").rlike(r"^\d{4}$"))
            )

            if cfg.should_filter_links():
                df = df.filter(F.col("linkid").isin(cfg.link_ids))

            df = df.withColumn(
                "datetime",
                F.to_timestamp(F.concat(F.col("date"), F.col("time")), "yyyyMMddHHmm")
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

            df = df.filter(F.col("date") == dt)

            df_silver = (
                df.groupBy("datetime_5min", "linkid")
                .agg(F.avg("speed").alias("self_mean"))
                .withColumnRenamed("datetime_5min", "datetime")
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


def its_silver_to_gold(start_date: str, end_date: str, config: TrafficPipelineConfig | None = None) -> None:
    cfg = config or PIPELINE_CONFIG
    spark = create_spark("SILVER_TO_GOLD_ITS_TRAFFIC_5MIN")

    pg_host = os.getenv("PG_HOST", _POSTGRES_CONF.get("host", "postgres"))
    pg_port = os.getenv("PG_PORT", str(_POSTGRES_CONF.get("port", 5432)))
    pg_db = os.getenv("PG_DB", _POSTGRES_CONF.get("db", "mlops"))
    pg_user = os.getenv("PG_USER", _POSTGRES_CONF.get("user", "postgres"))
    pg_password = os.getenv("PG_PASSWORD", _POSTGRES_CONF.get("password", "postgres"))
    pg_table = cfg.postgres_table

    jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"
    
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{cfg.base_silver}/date={dt}"
        
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
            if "date" in cols:
                rename_map["date"] = "date"
            if "time" in cols:
                rename_map["time"] = "time"

            df_out = df_silver
            for src, dst in rename_map.items():
                df_out = df_out.withColumnRenamed(src, dst)

            df_out = df_out.withColumn("date", F.lit(dt).cast("string"))

            # Gold 테이블이 self_mean 외 t1/t2/f1/f2_mean을 요구하므로 기본값으로 self_mean 복사
            if "self_mean" in df_out.columns:
                for extra_col in ["t1_mean", "t2_mean", "f1_mean", "f2_mean"]:
                    if extra_col not in df_out.columns:
                        df_out = df_out.withColumn(extra_col, F.col("self_mean"))

            ordered_cols = []
            for c in ["date", "datetime", "linkid", "self_mean", "t1_mean", "t2_mean", "f1_mean", "f2_mean"]:
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
