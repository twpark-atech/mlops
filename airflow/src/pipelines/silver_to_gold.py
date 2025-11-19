# src/pipelines/silver_to_gold.py
import os
import argparse
import psycopg2
from src.common.spark_session import create_spark
from pyspark.sql import functions as F
from datetime import datetime, timedelta

def _get_env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _delete_existing_partition(
    dt: str,
    table: str,
    host: str,
    port: str,
    db: str,
    user: str,
    password: str,
    date_col: str = "date",
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
            cur.execute(sql, (dt,))
        print(f'[SILVER→GOLD] {table}에서 date={dt} 기존 행 삭제 완료')
    except Exception as e:
        print(f'[SILVER→GOLD][WARN] date={dt} 삭제 중 오류 (무시하고 진행): {e}')
    finally:
        if conn is not None:
            conn.close()

def run_its_traffic_silver_to_gold(start_date: str, end_date: str) -> None:
    spark = create_spark("SILVER_TO_GOLD_ITS_TRAFFIC_5MIN")

    base_silver = "s3a://its/traffic/silver"

    pg_host = _get_env("PG_HOST", "localhost")
    pg_port = _get_env("PG_PORT", "5432")
    pg_db = _get_env("PG_DB", "mlops")
    pg_user = _get_env("PG_USER", "postgres")
    pg_password = _get_env("PG_PASSWORD", "postgres")
    pg_table = _get_env("ITS_TRAFFIC_GOLD_TABLE", "its_traffic_5min_gold")

    jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        silver_path = f"{base_silver}/date={dt}"

        print(f"[SILVER→GOLD] {dt} 읽는 중: {silver_path}")

        try:
            df_silver = spark.read.parquet(silver_path)
            
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

            print(f"[SILVER→GOLD] {dt} → {pg_table} 적재 완료")
        
        except Exception as e:
            print(f"[SILVER→GOLD][WARN] {dt} 처리 실패: {e}")
        
        cur += timedelta(days=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True, help="예: its_traffic_5min")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    args = parser.parse_args()

    if args.job_name == "its_traffic_5min":
        run_its_traffic_silver_to_gold(args.start_date, args.end_date)
    else:
        raise ValueError(f"지원하지 않는 job-name: {args.job_name}")
    
if __name__ == "__main__":
    main()