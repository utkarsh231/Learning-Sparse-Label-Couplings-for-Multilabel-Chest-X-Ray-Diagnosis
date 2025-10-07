from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from .utils import macro_auc


def train_one_epoch(model, refiner, loader, optimizer, scaler: GradScaler, criterion, device,
                    ema=None, scheduler=None, amp: bool = True):
    model.train()
    if refiner is not None:
        refiner.train()
    tot = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            if refiner is not None:
                logits, reg = refiner(logits)
                loss_sup = criterion(logits, y)
                loss = loss_sup + reg
            else:
                loss = criterion(logits, y)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected. Skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(refiner.parameters()) if refiner else []), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        tot += float(loss.detach().cpu())
    return tot / max(1, len(loader))


@torch.no_grad()
def validate(model, refiner, loader, device):
    model.eval()
    if refiner is not None:
        refiner.eval()
    Ps, Ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if refiner is not None:
            logits, _ = refiner(logits)
        Ps.append(torch.sigmoid(logits).cpu().numpy())
        Ys.append(y.numpy())
    P = np.concatenate(Ps, 0); Y = np.concatenate(Ys, 0)
    return macro_auc(Y, P)


@torch.no_grad()
def predict(model, refiner, loader, device, tta: bool = True):
    model.eval()
    if refiner is not None:
        refiner.eval()
    names_all, probs_all = [], []
    for names, x in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if refiner is not None:
            logits, _ = refiner(logits)
        p = torch.sigmoid(logits)

        if tta:
            x2 = torch.flip(x, dims=[3])
            logits2 = model(x2)
            if refiner is not None:
                logits2, _ = refiner(logits2)
            p = (p + torch.sigmoid(logits2)) / 2.0

        probs_all.append(p.cpu().numpy())
        names_all.extend(list(names))
    return names_all, np.concatenate(probs_all, 0)
