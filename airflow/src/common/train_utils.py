# airflow/src/common/train_utils.py
import io
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx:idx+1])


def set_seed(seed: int = 42):
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


def train_one_epoch_lstm(model, loader, optim, loss_fn, device, cfg, thr_z: float):
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
def evaluate_lstm(model, loader, loss_fn, device, thr_z: float):
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


def _parse_datetime(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    try:
        dt = pd.to_datetime(s, format="mixed", errors="coerce", cache=False)
    except TypeError:
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, cache=False)
    return dt

