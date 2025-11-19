# src/training/train_its_traffic_convlstm.py

from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import psycopg2
import mlflow
import mlflow.pytorch

from .model import TrafficConvLSTM  # src/training/model.py

# ----------------------------
# 안정성 관련 권장 설정
# ----------------------------
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

FEATURE_COLS = ["t2_mean", "t1_mean", "self_mean", "f1_mean", "f2_mean"]


# ----------------------------
# Utils
# ----------------------------
def _parse_datetime(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    try:
        dt = pd.to_datetime(s, format="mixed", errors="coerce", cache=False)
    except TypeError:
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, cache=False)
    return dt


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
        g = g.sort_values("datetime").reset_index(drop=True)
        idx = pd.date_range(g["datetime"].min(), g["datetime"].max(), freq="5min")
        g2 = g.set_index("datetime").reindex(idx)
        g2["linkid"] = link
        g2.index.name = "datetime"
        out_list.append(g2.reset_index())

    if not out_list:
        raise ValueError("No valid rows after datetime parsing / grouping.")
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
    """
    시퀀스의 '마지막 시점(=window_end)'이 last_time_range 안에 드는 샘플만 생성.
    히스토리는 last_time_range 이전에서 끌어와도 허용.
    타깃은 원본 self_mean을 train 스케일러(mu, sd)로 표준화.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    t0, t1 = fit_scaler_on
    r0, r1 = last_time_range

    if allowed_links is not None:
        df = df[df["linkid"].isin(allowed_links)]

    # 스케일러 (train 구간)
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

        # 원본 보존
        for col in FEATURE_COLS:
            g[col + "_raw"] = g[col]

        # 입력 표준화
        for col in FEATURE_COLS:
            mu, sd = scalers[link][col]
            g[col] = (g[col] - mu) / sd

        # 타깃: 원본 self_mean → pred_horizon 시프트 → train 스케일러로 표준화
        mu_y, sd_y = scalers[link]["self_mean"]
        g["target"] = (g["self_mean_raw"].shift(-pred_horizon) - mu_y) / sd_y

        g = g.dropna(subset=["target"]).reset_index(drop=True)
        N = len(g)
        if N < seq_len:
            continue

        vals = g[FEATURE_COLS + ["target"]].values  # (N, 6)

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


# ----------------------------
# Dataset
# ----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx:idx+1])


# ----------------------------
# Train / Eval Config
# ----------------------------
@dataclass
class TrainConfig:
    # job 정보
    job_name: str = "its_traffic_5min_convlstm"
    start_date: Optional[str] = None  # YYYYMMDD
    end_date: Optional[str] = None    # YYYYMMDD

    # 학습 하이퍼파라미터
    epochs: int = 50
    seq_len: int = 36
    pred_horizon: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    val_days: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    grad_clip: float = 1.0
    skip_nan_batches: bool = True
    seed: int = 42

    # F1 임계값 (표준화된 z-스코어에서 하위 몇 %를 혼잡으로 볼지)
    cls_thresh_percentile: float = 25.0

    # MLflow 설정
    mlflow_experiment: str = "its_traffic_convlstm"
    mlflow_run_name: Optional[str] = None
    mlflow_register_name: Optional[str] = None  # 모델 레지스트리에 등록하고 싶으면 이름 지정

    # Postgres
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "mlops"
    pg_user: str = "postgres"
    pg_password: str = "postgres"
    pg_table: str = "its_traffic_5min_gold"

    # 출력/체크포인트
    output_dir: str = "outputs/its_traffic_convlstm"
    ckpt_dir: str = "outputs/its_traffic_convlstm/checkpoints"
    weight_path: str = "outputs/its_traffic_convlstm/weights_best.pth"


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_batch(xb: torch.Tensor, yb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    is_finite = torch.isfinite(xb).all() and torch.isfinite(yb).all()
    if not is_finite:
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
    return xb, yb, is_finite


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def train_one_epoch(model, loader, optim, loss_fn, device, cfg: TrainConfig, thr_z: float):
    model.train()
    total_mae, n = 0.0, 0
    TP = FP = FN = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        xb, yb, was_finite = _sanitize_batch(xb, yb)
        if not was_finite and cfg.skip_nan_batches:
            continue

        optim.zero_grad(set_to_none=True)
        pred = model(xb)  # (B,1)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        loss = loss_fn(pred, yb)  # MAE
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        bsz = xb.size(0)
        total_mae += float(loss.detach().cpu()) * bsz
        n += bsz

        y_true_bin = (yb <= thr_z).view(-1).to(torch.int32)
        y_pred_bin = (pred <= thr_z).view(-1).to(torch.int32)

        TP += int(((y_pred_bin == 1) & (y_true_bin == 1)).sum().item())
        FP += int(((y_pred_bin == 1) & (y_true_bin == 0)).sum().item())
        FN += int(((y_pred_bin == 0) & (y_true_bin == 1)).sum().item())

    train_mae = total_mae / max(n, 1)
    train_f1 = _f1_from_counts(TP, FP, FN)
    return train_mae, train_f1


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, thr_z: float):
    model.eval()
    total_mae, n = 0.0, 0
    TP = FP = FN = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb, yb, _ = _sanitize_batch(xb, yb)

        pred = model(xb)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        loss = loss_fn(pred, yb)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        bsz = xb.size(0)
        total_mae += float(loss.detach().cpu()) * bsz
        n += bsz

        y_true_bin = (yb <= thr_z).view(-1).to(torch.int32)
        y_pred_bin = (pred <= thr_z).view(-1).to(torch.int32)

        TP += int(((y_pred_bin == 1) & (y_true_bin == 1)).sum().item())
        FP += int(((y_pred_bin == 1) & (y_true_bin == 0)).sum().item())
        FN += int(((y_pred_bin == 0) & (y_true_bin == 1)).sum().item())

    val_mae = total_mae / max(n, 1)
    val_f1 = _f1_from_counts(TP, FP, FN)
    return val_mae, val_f1


# ----------------------------
# Postgres에서 GOLD 로딩
# ----------------------------
def load_gold_from_postgres(cfg: TrainConfig) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=cfg.pg_host,
        port=cfg.pg_port,
        dbname=cfg.pg_db,
        user=cfg.pg_user,
        password=cfg.pg_password,
    )

    try:
        base_query = f"""
            SELECT
                datetime,
                linkid,
                -- GOLD에 이미 있는 self_mean을 5채널로 복제해서 사용 (MVP용)
                self_mean AS self_mean,
                self_mean AS t1_mean,
                self_mean AS t2_mean,
                self_mean AS f1_mean,
                self_mean AS f2_mean
            FROM {cfg.pg_table}
            WHERE date BETWEEN %s AND %s
        """
        params = (cfg.start_date, cfg.end_date)
        df = pd.read_sql(base_query, conn, params=params)
    finally:
        conn.close()

    return df


def run_training(cfg: TrainConfig) -> None:
    if not cfg.start_date or not cfg.end_date:
        raise ValueError("start_date/end_date must be provided for training")

    # 기본 출력 폴더들
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.weight_path), exist_ok=True)

    set_seed(cfg.seed)

    # 🔹 MLflow 기본 설정
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    run_name = cfg.mlflow_run_name or cfg.job_name
    with mlflow.start_run(run_name=run_name):
        # --- Config 파라미터 로깅
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
        })

        # 🔹 GOLD 로드
        df = load_gold_from_postgres(cfg)
        df = df[["datetime", "linkid"] + FEATURE_COLS].copy()

        # Grid & Impute
        df = ensure_5min_grid(df)
        df = impute_missing(df)

        # ----------------------------
        # Train / Val split (유연하게)
        # ----------------------------
        df["datetime"] = pd.to_datetime(df["datetime"])
        t_min = df["datetime"].min()
        t_max = df["datetime"].max()
        total_days = (t_max - t_min).days + 1

        # 기본 로직: 마지막 cfg.val_days 일을 validation으로 사용
        val_start = t_max - pd.Timedelta(days=cfg.val_days)

        if total_days > cfg.val_days:
            # 데이터 기간이 충분히 길면 기존 방식 사용
            df_train = df[df["datetime"] < val_start].copy()
            df_val   = df[df["datetime"] >= val_start].copy()
        else:
            # 데이터 기간이 짧으면(예: 하루만 있는 경우) 강제로 80% / 20% 시간순 split
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

        # train에 존재하는 link만 사용
        train_links = set(df_train["linkid"].unique())
        df_train = df_train[df_train["linkid"].isin(train_links)]
        df_val   = df_val[df_val["linkid"].isin(train_links)]

        # Build sequences
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

        # 🔹 F1 임계값 (표준화된 z-스코어 공간)
        thr_z = float(np.percentile(y_train, cfg.cls_thresh_percentile))
        mlflow.log_param("cls_thresh_percentile", cfg.cls_thresh_percentile)
        mlflow.log_metric("thr_z", thr_z)

        # 🔹 Dataloaders
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

        # 🔹 Model / Loss / Optim
        device = torch.device(cfg.device)
        model = TrafficConvLSTM(dropout=0.2).to(device)
        loss_fn = nn.L1Loss()  # MAE
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # 🔹 Train loop
        best_val = float("inf")
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            tr_mae, tr_f1 = train_one_epoch(model, dl_tr, optim, loss_fn, device, cfg, thr_z)
            va_mae, va_f1 = evaluate(model, dl_va, loss_fn, device, thr_z)

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

            # 로컬 checkpoint
            ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "thr_z": thr_z},
                ckpt_path,
            )
            # 필요하면: mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

            # 베스트 모델 갱신
            if np.isfinite(va_mae) and va_mae < best_val:
                best_val = va_mae
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(best_state, cfg.weight_path)
                mlflow.log_artifact(cfg.weight_path, artifact_path="best_model")

        mlflow.log_metric("best_val_mae", best_val)

        # 🔹 전체 모델을 MLflow에 저장 (PyTorch 형식)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", default="its_traffic_5min_convlstm")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    args = parser.parse_args()

    cfg = TrainConfig(
        job_name=args.job_name,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
