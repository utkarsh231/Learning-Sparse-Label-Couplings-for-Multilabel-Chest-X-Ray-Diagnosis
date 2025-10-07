from __future__ import annotations
import os, math, time, argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from .data import XRayDataset
from .engine import train_one_epoch, validate, predict
from .losses import AsymmetricLoss, make_bce_with_pos_weight
from .models import create_backbone, LabelGraphRefiner, ModelEMA
from .utils import seed_everything, get_mis_splits, print_fold_stats


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--test_dir",  type=str, required=True)
    ap.add_argument("--out_csv",   type=str, default="./submission.csv")
    ap.add_argument("--out_dir",   type=str, default="./outputs")

    default_labels = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
        'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
        'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
    ]
    ap.add_argument("--labels", type=str, default=",".join(default_labels))
    ap.add_argument("--model", type=str, default="seresnext101_32x4d",
                    help="e.g., seresnext101_32x4d | tf_efficientnet_b4_ns | senet154 | resnet200d | convnext_base")
    ap.add_argument("--img_size", type=int, default=380)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--loss", type=str, default="asl", choices=["asl","bce"])
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--color_jitter", action="store_true")
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision if you still see NaNs.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(args.train_csv)
    if "Image_name" not in df.columns:
        raise ValueError("train_csv must contain 'Image_name' column.")

    # labels
    label_cols = [c.strip() for c in args.labels.split(",") if c.strip() in df.columns]
    if len(label_cols) == 0:
        cand = [c for c in df.columns if c != "Image_name"]
        label_cols = [c for c in cand if set(pd.unique(pd.to_numeric(df[c], errors="coerce").dropna().clip(0,1))) <= {0,1}]
        print("[WARN] Using autodetected labels:", label_cols)
    assert len(label_cols) > 0, "No valid label columns found."
    df[label_cols] = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(0,1).astype(int)
    num_classes = len(label_cols)
    print(f"Using {num_classes} labels:", label_cols)

    # folds
    val_indices_list = list(get_mis_splits(df, label_cols, n_splits=args.folds, seed=args.seed))

    # test loader
    test_images = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    test_df = pd.DataFrame({"Image_name": test_images})
    test_ds = XRayDataset(test_df, args.test_dir, label_cols, args.img_size, train=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    all_fold_probs, names_ref = [], None

    for fold, val_idx in enumerate(val_indices_list, start=1):
        print(f"\n========== Fold {fold}/{args.folds} ==========")
        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[val_idx].reset_index(drop=True)
        print_fold_stats(tr_df, va_df, label_cols)

        train_ds = XRayDataset(tr_df, args.train_dir, label_cols, args.img_size, train=True,
                               label_smoothing=args.label_smooth, color_jitter=args.color_jitter)
        val_ds   = XRayDataset(va_df, args.train_dir, label_cols, args.img_size, train=False)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        # --- Model & Refiner ---
        model = create_backbone(args.model, num_classes=num_classes, drop_rate=0.4, pretrained=True).to(device)
        refiner = LabelGraphRefiner(num_classes, alpha=0.3, l1=1e-3).to(device)

        # --- Loss ---
        if args.loss == "asl":
            criterion = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        else:
            criterion = make_bce_with_pos_weight(tr_df, label_cols, device)

        # --- Optimizer includes both model and refiner params ---
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(refiner.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

        # ---- Per-step cosine with warmup ----
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = max(1, args.warmup_epochs * len(train_loader))
        scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

        scaler = GradScaler(enabled=not args.no_amp)
        ema = ModelEMA(model) if args.ema else None

        best_auc, best_path = -1.0, os.path.join(args.out_dir, f"{args.model.replace('/', '_')}_fold{fold}.pth")
        patience, no_improve = 3, 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, refiner, train_loader, optimizer, scaler, criterion, device,
                ema=ema, scheduler=scheduler, amp=(not args.no_amp)
            )
            eval_model = ema.ema if (ema is not None) else model
            val_auc = validate(eval_model, refiner, val_loader, device)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[Fold {fold} | Epoch {epoch}] loss={train_loss:.4f} valAUC={val_auc:.4f} lr={lr_now:.2e} time={(time.time()-t0):.1f}s")

            if val_auc > best_auc or (np.isnan(best_auc) and not np.isnan(val_auc)):
                best_auc = val_auc; no_improve = 0
                torch.save({"model": eval_model.state_dict(), "refiner": refiner.state_dict()}, best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping: best valAUC {best_auc:.4f}")
                    break

        # --- Load best and predict ---
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=False); model.to(device)
        refiner.load_state_dict(state["refiner"], strict=False); refiner.to(device)

        names, probs = predict(model, refiner, test_loader, device, tta=args.tta)
        if names_ref is None:
            names_ref = names
        all_fold_probs.append(probs)

    avg_probs = np.mean(all_fold_probs, axis=0)
    sub = pd.DataFrame({"Image_name": names_ref})
    sub[label_cols] = avg_probs
    sub.to_csv(args.out_csv, index=False)
    print(f"\nSaved submission to: {args.out_csv}")


if __name__ == "__main__":
    main()
