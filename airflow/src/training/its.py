# airflow/src/training/its.py
from __future__ import annotations

import io
import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from src.common.config_loader import load_base_config
from src.common.db_utils import _get_minio_client, load_gold_from_postgres, _upload_bytes_to_minio
from src.common.train_utils import set_seed, _parse_datetime, SeqDataset, train_one_epoch_lstm, evaluate_lstm
from .model import TrafficConvLSTM


os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

FEATURE_COLS = ["t2_mean", "t1_mean", "self_mean", "f1_mean", "f2_mean"]

_BASE_CONF = load_base_config()
_TRAIN_CONF = _BASE_CONF.get("training", {})
_POSTGRES_CONF = _BASE_CONF.get("postgres", {})
_MINIO_CONF = _BASE_CONF.get("minio", {})
_MLFLOW_CONF = _BASE_CONF.get("mlflow", {})


@dataclass
class TrainConfig:
    job_name: str = field(default_factory=lambda: _TRAIN_CONF.get("job_name", "its_traffic_5min_convlstm"))
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # 학습 하이퍼파라미터
    epochs: int = field(default_factory=lambda: int(_TRAIN_CONF.get("epochs", 50)))
    seq_len: int = field(default_factory=lambda: int(_TRAIN_CONF.get("seq_len", 36)))
    pred_horizon: int = field(default_factory=lambda: int(_TRAIN_CONF.get("pred_horizon", 1)))
    batch_size: int = field(default_factory=lambda: int(_TRAIN_CONF.get("batch_size", 64)))
    lr: float = field(default_factory=lambda: float(_TRAIN_CONF.get("lr", 1e-3)))
    val_days: int = field(default_factory=lambda: int(_TRAIN_CONF.get("val_days", 30)))
    device: str = field(default_factory=lambda: _TRAIN_CONF.get("device", "auto"))
    num_workers: int = field(default_factory=lambda: int(_TRAIN_CONF.get("num_workers", 0)))
    grad_clip: float = field(default_factory=lambda: float(_TRAIN_CONF.get("grad_clip", 1.0)))
    skip_nan_batches: bool = field(default_factory=lambda: bool(_TRAIN_CONF.get("skip_nan_batches", True)))
    seed: int = field(default_factory=lambda: int(_TRAIN_CONF.get("seed", 42)))

    # F1 임계값
    cls_thresh_percentile: float = field(default_factory=lambda: float(_TRAIN_CONF.get("cls_thresh_percentile", 25.0)))

    # MLflow 설정
    mlflow_experiment: str = field(default_factory=lambda: _TRAIN_CONF.get("mlflow_experiment", "its_traffic_convlstm"))
    mlflow_run_name: Optional[str] = field(default_factory=lambda: _TRAIN_CONF.get("mlflow_run_name"))
    mlflow_register_name: Optional[str] = field(default_factory=lambda: _TRAIN_CONF.get("mlflow_register_name"))
    mlflow_tracking_uri: str = field(default_factory=lambda: _MLFLOW_CONF.get("tracking_uri", "http://mlflow:5000"))

    # Postgres
    pg_host: str = field(default_factory=lambda: _POSTGRES_CONF.get("host", "postgres"))
    pg_port: int = field(default_factory=lambda: int(_POSTGRES_CONF.get("port", 5432)))
    pg_db: str = field(default_factory=lambda: _POSTGRES_CONF.get("db", "mlops"))
    pg_user: str = field(default_factory=lambda: _POSTGRES_CONF.get("user", "postgres"))
    pg_password: str = field(default_factory=lambda: _POSTGRES_CONF.get("password", "postgres"))
    pg_table: str = field(default_factory=lambda: _POSTGRES_CONF.get("table", "its_traffic_5min_gold"))

    # MinIO 설정
    minio_endpoint: str = field(default_factory=lambda: _MINIO_CONF.get("endpoint", "minio:9000"))
    minio_access_key: str = field(default_factory=lambda: _MINIO_CONF.get("access_key", "minio"))
    minio_secret_key: str = field(default_factory=lambda: _MINIO_CONF.get("secret_key", "miniostorage"))
    minio_bucket: str = field(default_factory=lambda: _TRAIN_CONF.get("minio_bucket", _MINIO_CONF.get("bucket", "its")))
    minio_secure: bool = field(default_factory=lambda: bool(_MINIO_CONF.get("secure", False)))
    minio_prefix: str = field(default_factory=lambda: _TRAIN_CONF.get("minio_prefix", "its/traffic/its_traffic_convlstm"))


QUERY = f"""
    SELECT
        datetime, linkid,
        self_mean AS self_mean,
        self_mean AS t1_mean,
        self_mean AS t2_mean,
        self_mean AS f1_mean,
        self_mean AS f2_mean
    FROM {TrainConfig.pg_table}
    WHERE date BETWEEN %s AND %s
"""


def ensure_5min_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = _parse_datetime(df["datetime"])
    df = df.dropna(subset=["datetime", "linkid"])

    num_cols = [c for c in FEATURE_COLS if c in df.columns]
    agg = {c: "mean" for c in num_cols}
    df = (
        df.groupby(["linkid", "datetime"], as_index=False)
            .agg(agg)
            .sort_values(["linkid", "datetime"])
    )

    out_list = []
    for link, g in df.groupby("linkid", sort=False):
        if g.empty:
            continue
        g = g.sort_index("datetime").reset_index(drop=True)
        idx = pd.date_range(g["datetime"].min(), g["datetime"].max(), freq="5min")
        g2 = g.set_index("datetime").reindex(idx)
        g2["linkid"] = link
        g2.index.name = "datetime"
        out_list.append(g2.reset_index())

    if not out_list:
        raise ValueError("datetime 기준으로 파싱, 그룹핑된 row가 없습니다.")
    return pd.concat(out_list, axis=0, ignore_index=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = _parse_datetime(df["datetime"])
    df = df.dropna(subset=["datetime", "linkid"])
    df["weekday"] = df["datetime"].dt.weekday
    df["hhmm"] = df["datetime"].dt.strftime("%H:%M")
    df["month"] = df["datetime"].dt.month

    out = []
    for link, g in df.groupby("linkid", sort=False):
        g = g.sort_values("datetime").copy().set_index("datetime")
        for col in FEATURE_COLS:
            g[col + "_rfill"] = g[col].rolling("31min", center=True, min_periods=1).mean()
            g[col] = g[col].fillna(g[col + "_rfill"])
            g.drop(columns=[col + "_rfill"], inplace=True)
        out.append(g.reset_index())
    df = pd.concat(out, ignore_index=True)

    key_cols = ["linkid", "weekday", "hhmm"]
    base = df[key_cols + FEATURE_COLS + ["month"]].copy()
    grp = base.groupby(key_cols)[FEATURE_COLS].mean().reset_index()
    df = df.merge(grp, on=key_cols, how="left", suffixes=("", "_wkmean"))

    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col + "_wkmean"])
        df.drop(columns=[col + "_wkmean"], inplace=True)

    df.drop(columns=["weekday", "hhmm", "month"], inplace=True)
    return df

def build_sequences_by_last_time(
    df: pd.DataFrame,
    seq_len: int,
    pred_horizon: int,
    fit_scaler_on: Tuple[pd.Timestamp, pd.Timestamp],
    last_time_range: Tuple[pd.Timestamp, pd.Timestamp],
    allowed_links: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    t0, t1 = fit_scaler_on
    r0, r1 = last_time_range

    if allowed_links is not None:
        df = df[df["linkid"].isin(allowed_links)]

    scalers: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for link, g in df.groupby("linkid"):
        g_train = g[(g["datetime"] >= t0) & (g["datetime"] <= t1)]
        if g_train.empty:
            continue
        stats: Dict[str, Tuple[float, float]] = {}
        for col in FEATURE_COLS:
            mu = float(g_train[col].mean())
            sd = float(g_train[col].std(ddof=0))
            if not np.isfinite(mu):
                mu = 0.0
            if (not np.isfinite(sd)) or sd == 0.0:
                sd = 1.0
            stats[col] = (mu, sd)
        scalers[link] = stats

    X_list, y_list = [], []

    for link, g in df.groupby("linkid", sort=False):
        if link not in scalers:
            continue

        g = g.sort_values("datetime").reset_index(drop=True)

        for col in FEATURE_COLS:
            g[col + "_raw"] = g[col]

        for col in FEATURE_COLS:
            mu, sd = scalers[link][col]
            g[col] = (g[col] - mu) / sd

        mu_y, sd_y = scalers[link]["self_mean"]
        g["target"] = (g["self_mean_raw"].shift(-pred_horizon) - mu_y) / sd_y

        g = g.dropna(subset=["target"]).reset_index(drop=True)
        N = len(g)
        if N < seq_len:
            continue

        vals = g[FEATURE_COLS + ["target"]].values

        for i in range(N - seq_len + 1):
            end_idx = i + seq_len - 1
            end_time = g.loc[end_idx, "datetime"]
            if not (r0 <= end_time <= r1):
                continue
            window = vals[i : i + seq_len]
            feats = window[:, :5]
            x = feats.reshape(seq_len, 1, 5, 1).astype(np.float32)
            y = np.float32(window[-1, 5])

            if not np.isfinite(x).all() or not np.isfinite(y):
                continue

            X_list.append(x)
            y_list.append(y)

    if len(X_list) == 0:
        raise ValueError(
            "No sequences were generated for the given last_time_range. "
            "Consider reducing seq_len or checking data continuity."
        )

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def run_training(cfg: TrainConfig) -> None:
    if not cfg.start_date or not cfg.end_date:
        raise ValueError("start_date/end_date는 학습을 위해 필수 제공되어야 합니다.")
    
    set_seed(cfg.seed)
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    minio_client = _get_minio_client(cfg)

    run_name = cfg.mlflow_run_name or cfg.job_name
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "job_name": cfg.job_name,
            "pg_host": cfg.pg_host,
            "pg_port": cfg.pg_port,
            "pg_db": cfg.pg_db,
            "pg_table": cfg.pg_table,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "epochs": cfg.epochs,
            "seq_len": cfg.seq_len,
            "pred_horizon": cfg.pred_horizon,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "val_days": cfg.val_days,
            "grad_clip": cfg.grad_clip,
            "device": cfg.device,
            "minio_bucket": cfg.minio_bucket,
            "minio_prefix": cfg.minio_prefix,
        })

        df = load_gold_from_postgres(cfg, QUERY)
        df = df[["datetime", "linkid"] + FEATURE_COLS].copy()

        df = ensure_5min_grid(df)
        df = impute_missing(df)

        df["datetime"] = pd.to_datetime(df["datetime"])
        t_min = df["datetime"].min()
        t_max = df["datetime"].max()
        total_days = (t_max - t_min).days + 1

        val_start = t_max - pd.Timedelta(days=cfg.val_days)

        if total_days > cfg.val_days:
            df_train = df[df["datetime"] < val_start].copy()
            df_val   = df[df["datetime"] >= val_start].copy()
        else:
            df_sorted = df.sort_values("datetime").reset_index(drop=True)
            split_idx = max(1, int(len(df_sorted) * 0.8))
            df_train = df_sorted.iloc[:split_idx].copy()
            df_val   = df_sorted.iloc[split_idx:].copy()

        if df_train.empty or df_val.empty:
            raise ValueError(
                f"Train/Val split produced empty sets. "
                f"total_days={total_days}, len(df)={len(df)}, "
                f"len(train)={len(df_train)}, len(val)={len(df_val)}"
            )
        
        train_links = set(df_train["linkid"].unique())
        df_train = df_train[df_train["linkid"].isin(train_links)]
        df_val   = df_val[df_val["linkid"].isin(train_links)]

        t0, t1 = df_train["datetime"].min(), df_train["datetime"].max()
        v0, v1 = df_val["datetime"].min(), df_val["datetime"].max()

        X_train, y_train = build_sequences_by_last_time(
            df=df,
            seq_len=cfg.seq_len,
            pred_horizon=cfg.pred_horizon,
            fit_scaler_on=(t0, t1),
            last_time_range=(t0, t1),
            allowed_links=train_links,
        )
        X_val, y_val = build_sequences_by_last_time(
            df=df,
            seq_len=cfg.seq_len,
            pred_horizon=cfg.pred_horizon,
            fit_scaler_on=(t0, t1),
            last_time_range=(v0, v1),
            allowed_links=train_links,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("No sequences generated. Try smaller seq_len or check data continuity.")

        thr_z = float(np.percentile(y_train, cfg.cls_thresh_percentile))
        mlflow.log_param("cls_thresh_percentile", cfg.cls_thresh_percentile)
        mlflow.log_metric("thr_z", thr_z)

        ds_tr = SeqDataset(X_train, y_train)
        ds_va = SeqDataset(X_val, y_val)
        dl_tr = DataLoader(
            ds_tr,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=(cfg.device == "cuda"),
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=(cfg.device == "cuda"),
        )

        mlflow.log_metric("train_samples", len(ds_tr))
        mlflow.log_metric("val_samples", len(ds_va))

        device = torch.device(cfg.device)
        model = TrafficConvLSTM(dropout=0.2).to(device)
        loss_fn = nn.L1Loss()
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        best_val = float("inf")
        best_state = None
        best_weights_s3_path: Optional[str] = None

        for epoch in range(1, cfg.epochs + 1):
            tr_mae, tr_f1 = train_one_epoch_lstm(model, dl_tr, optim, loss_fn, device, cfg, thr_z)
            va_mae, va_f1 = evaluate_lstm(model, dl_va, loss_fn, device, thr_z)

            print(
                f"[{epoch:03d}/{cfg.epochs}] "
                f"train_mae={tr_mae:.6f}  train_f1={tr_f1:.4f}  "
                f"val_mae={va_mae:.6f}  val_f1={va_f1:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_mae": tr_mae,
                    "train_f1": tr_f1,
                    "val_mae": va_mae,
                    "val_f1": va_f1,
                    "lr": optim.param_groups[0]["lr"],
                },
                step=epoch,
            )

            ckpt_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "thr_z": thr_z,
            }
            ckpt_buffer = io.BytesIO()
            torch.save(ckpt_state, ckpt_buffer)
            ckpt_bytes = ckpt_buffer.getvalue()

            ckpt_object_name = f"{cfg.minio_prefix}/checkpoints/epoch_{epoch:03d}.pth"
            ckpt_s3_path = _upload_bytes_to_minio(
                minio_client,
                cfg,
                ckpt_bytes,
                ckpt_object_name,
            )
            mlflow.log_text(ckpt_s3_path, artifact_file=f"checkpoints/epoch_{epoch:03d}.pth.path")

            if np.isfinite(va_mae) and va_mae < best_val:
                best_val = va_mae
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                best_buffer = io.BytesIO()
                torch.save(best_state, best_buffer)
                best_bytes = best_buffer.getvalue()

                best_object_name = f"{cfg.minio_prefix}/weights_best.pth"
                best_weights_s3_path = _upload_bytes_to_minio(
                    minio_client,
                    cfg,
                    best_bytes,
                    best_object_name,
                )
                mlflow.log_param("best_weights_s3_path", best_weights_s3_path)

        mlflow.log_metric("best_val_mae", best_val)

        if best_state is not None:
            model.load_state_dict(best_state)

        if cfg.mlflow_register_name:
            mlflow.pytorch.log_model(
                model,
                artifact_path="pytorch-model",
                registered_model_name=cfg.mlflow_register_name,
            )
        else:
            mlflow.pytorch.log_model(
                model,
                artifact_path="pytorch-model",
            )

        if best_state is None:
            print("Warning: best_state is None. No best model saved.")
        else:
            print(f"Best model weights stored at: {best_weights_s3_path}")
