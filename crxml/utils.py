from __future__ import annotations
import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def have_iterstrat() -> bool:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: F401
        return True
    except Exception:
        return False


def get_mis_splits(df: pd.DataFrame, label_cols: list[str], n_splits: int = 5, seed: int = 42):
    """Yield val indices for MIS; fallback to bucketed KFold if unavailable."""
    try:
        if have_iterstrat():
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            X = df.index.values
            Y = df[label_cols].values
            for _, val_idx in mskf.split(X, Y):
                yield val_idx
            return
        else:
            raise ImportError("iterative-stratification not installed")
    except Exception as e:
        print("[INFO] MIS unavailable, falling back to bucketed KFold:", e)
        combos = (df[label_cols].astype(int).astype(str)).agg("".join, axis=1)
        buckets = combos.apply(lambda s: hash(s) % 1000)
        df2 = df.copy(); df2["_bucket"] = buckets
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for _, val_idx in skf.split(df2, df2["_bucket"].values):
            yield val_idx


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aucs = []
    for k in range(y_true.shape[1]):
        yt = y_true[:, k]; yp = y_pred[:, k]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            pass
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def print_fold_stats(tr_df: pd.DataFrame, va_df: pd.DataFrame, label_cols: list[str]):
    tr_pos = tr_df[label_cols].sum().astype(int)
    va_pos = va_df[label_cols].sum().astype(int)
    print("Train positives per label:", tr_pos.to_dict())
    print(" Val  positives per label:", va_pos.to_dict())