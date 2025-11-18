# src/common/its_traffic_5min.py
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Set
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from datetime import datetime, timedelta

LINKIDS_ALL: List[str] = [
    "1920161400", "1920161500",
    "1920121301", "1920121401",
    "1920161902", "1920162205", "1920162400",
    "1920000702", "1920000801", "1920121000", "1920121302", "1920121402",
    "1920235801", "1920189001", "1920139400", "1920161801", "1920162207",
    "1920162304", "1920162500", "1920171200", "1920171600", "1920188900", "1920138500"
]

TARGET_LINKS: List[str] = ["1920161400", "1920161500"]

RAW_BASE = "s3a://its/traffic_5min/raw"
BRONZE_BASE = "s3a://its/traffic_5min/bronze"
SILVER_BASE = "s3a://its/traffic_5min/silver"
GOLD_BASE = "s3a://its/traffic_5min/gold"

LINK_SHP_PATH = "/data/NODE_LINK/MOCT_LINK.shp"

def iter_dates(start: str, end: str):
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    cur = s
    while cur <= e:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)

def transform_raw_to_bronze(df: DataFrame) -> DataFrame:
    for c in ["CREATDE", "CREATHM", "LINKID", "PASNGSPED"]:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None))

    df = (
        df
        .withColumn("CREATDE", F.lpad(F.col("CREATDE").cast(T.StringType()), 8, "0"))
        .withColumn("CREATHM", F.lpad(F.col("CREATHM").cast(T.StringType()), 4, "0"))
        .withColumn("LINKID", F.col("LINKID").cast(T.StringType()))
    )

    mask_date = F.col("CREATDE").rlike(r"^[0-9]{8}$")
    mask_time = F.col("CREATHM").rlike(r"^[0-9]{4}$")
    df = df.where(mask_date & mask_time)

    df = df.withColumn(
        "PASNGSPED",
        F.when(
            F.col("PASNGSPED").cast(T.StringType()).rlike(r"^[0-9]+(\.[0-9]+)?$"),
            F.col("PASNGSPED").cast(T.DoulbeType())
        ).otherwise(F.lit(None).cast(T.DoubleType()))
    )

    df = df.withColumn(
        "DATETIME",
        F.to_timestamp(
            F.concat_ws("", F.col("CREATDE", "CREATHM")),
            "yyyyMMddHHmm"
        )
    )

    df = df.where(F.col("DATETIME").isNotNull() & F.col("LINKID").isNotNull())

    if "date" in df.columns:
        df = df.withColumn("date", F.col("date").cast(T.StringType()))
    else:
        df = df.withColumn("date", F.col("CREATDE"))

    keep_cols = [
        "date",
        "CREATDE",
        "CREATHM",
        "LINKID",
        "ROADINSTTCD",
        "PASNGSPED",
        "PASNGTIME",
        "DATETIME"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df.select(*keep_cols)

def transform_bronze_to_silver(df: DataFrame) -> DataFrame:
    required = ["date", "DATETIME", "LINKID", "PASNGSPED"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"bronze 데이터에 필수 컬럼 {c}가 없습니다.")
    
    agg = (
        df.groupBy("date", "DATETIME", "LINKID")
        .agg(
            F.avg("PASNGSPED").cast(T.FloatType()).alias("self_mean"),
            F.count(F.lit(1)).alias("n_obs")
        )
    )

    out = (
        agg.select(
            "date",
            F.col("DATETIME").alias("datetime"),
            F.col("LINKID").alias("linkid"),
            "self_mean",
            "n_obs"
        )
        .ordeyBy("datetime", "linkid")
    )
    return out

def ensure_link_columns(gdf_links: gpd.GeoDataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in gdf_links.columns}

    def pick(cands):
        for c in cands:
            if c in gdf_links.columns:
                return c
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        raise KeyError(f"필수 컬럼 없음: {cands}")
    
    link_col = pick(["LINK_ID", "LINKID", "LINK", "LINK_NO"])
    f_col = pick(["F_NODE", "FNODE", "F_NODEID", "F_NO"])
    t_col = pick(["T_NODE", "TNODE", "T_NODEID", "T_NO"])

    sub = gdf_links[[link_col, f_col, t_col]].copy()
    sub.columns = ["LINK_ID", "F_NODE", "T_NODE"]
    for c in ["LINK_ID", "F_NODE", "T_NODE"]:
        if sub[c].dtype != "object":
            sub[c] = sub[c].astype(str)
    return sub

def build_neighbor_maps(
    link_tab: pd.DataFrame,
    target_linkids: List[str]
) -> Dict[str, Dict[str, Set[str]]]:
    idx_by_t = link_tab.groupby("T_NODE")["LINK_ID"].apply(set).to_dict()
    idx_by_f = link_tab.groupby("F_NODE")["LINK_ID"].apply(set).to_dict()
    f_by_link = dict(zip(link_tab["LINK_ID"], link_tab["F_NODE"]))
    t_by_link = dict(zip(link_tab["LINK_ID"], link_tab["T_NODE"]))
    
    out: Dict[str, Dict[str, Set[str]]] = {}
    target_linkids = [str(x) for x in target_linkids]
    for lid in target_linkids:
        f0 = f_by_link.get(lid)
        t0 = t_by_link.get(lid)
        if f0 is None or t0 is None:
            out[lid] = {"t1": set(), "f1": set(), "t2": set(), "f2": set()}
            continue

        t1 = set(idx_by_t.get(t0, set()))
        f1 = set(idx_by_f.get(f0, set()))

        prev2_nodes = set(
            link_tab.loc[link_tab["LINK_ID"].isin(t1), "F_NODE"].unique().tolist()
        ) if t1 else set()
        next2_nodes = set(
            link_tab.loc[link_tab["LINK_ID"].isin(f1), "T_NODE"].unique().tolist()
        ) if f1 else set()
        
        t2: Set[str] = set()
        if prev2_nodes:
            for n in prev2_nodes:
                t2 |= set(idx_by_t.get(n, set()))
        
        f2: Set[str] = set()
        if next2_nodes:
            for n in next2_nodes:
                f2 |= set(idx_by_f.get(n, set()))

        out[lid] = {"t1": t1, "f1": f1, "t2": t2, "f2": f2}
    return out

def neighbor_long_df(neighbor_maps: Dict[str, Dict[str, Set[str]]]) -> pd.DataFrame:
    rows = []
    for lid, hops in neighbor_maps.items():
        for hop, s in hops.items():
            if not s:
                continue
            for nb in s:
                rows.append((lid, hop, nb))