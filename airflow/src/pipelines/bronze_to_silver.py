# src/pipelines/bronze_to_silver.py
import argparse
from src.common.spark_session import create_spark
from pyspark.sql import functions as F
from datetime import datetime, timedelta

LINKIDS_ALL = [
    "1920161400", "1920161500",
    "1920121301", "1920121401",
    "1920161902", "1920162205", "1920162400",
    "1920000702", "1920000801", "1920121000", "1920121302", "1920121402",
    "1920235801", "1920189001", "1920139400", "1920161801", "1920162207",
    "1920162304", "1920162500", "1920171200", "1920171600", "1920188900", "1920138500"
]

def run_its_traffic_bronze_to_silver(start_date: str, end_date: str) -> None:
    print(f"[BRONZE→SILVER] job=its_traffic_5min, range={start_date} ~ {end_date}")

    spark = create_spark("BRONZE_TO_SILVER_ITS_TRAFFIC_5MIN")

    base_bronze = "s3a://its/traffic/bronze"
    base_silver = "s3a://its/traffic/silver"

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")

        input_path = f"{base_bronze}/date={dt}"
        output_path = f"{base_silver}/date={dt}"

        print(f"[BRONZE→SILVER] {dt} 읽는 중: {input_path}")

        try:
            df_bronze = spark.read.parquet(input_path)

            df = (
                df_bronze.select(
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
            
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_its_traffic_bronze_to_silver(args.start_date, args.end_date)