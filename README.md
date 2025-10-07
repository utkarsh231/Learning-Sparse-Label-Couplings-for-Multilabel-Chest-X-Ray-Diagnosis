# Learning-Sparse-Label-Couplings-for-Multilabel-Chest-X-Ray-Diagnosis


> A modular PyTorch codebase for multilabel chest X‑ray classification that **learns sparse label couplings** end‑to‑end to refine logits and improve macro‑AUC. Includes MIS K‑fold splits, AMP, EMA, cosine LR with warmup, ASL/BCE losses, and flip‑TTA.

<p align="center">
  <img src="docs/diagram_placeholder.png" alt="Model overview" width="620"/>
</p>

---

## Highlights
- **Sparse label couplings**: learn a coupling matrix \(A\) with L1 regularization and refine logits via
  \[ z' = z + \alpha\,\sigma(z)\,A \]
  yielding a simple, fast, and effective co‑occurrence prior.
- **Solid training loop**: per‑step cosine LR with warmup, AMP, gradient clipping, EMA.
- **Robust splitting**: MultilabelStratifiedKFold (MIS) with a graceful bucketed‑KFold fallback.
- **Loss choices**: Asymmetric Loss (ASL) or BCE with data‑driven `pos_weight`.
- **Inference niceties**: horizontal flip TTA and (optional) fold ensembling.
- **Timm backbones**: one‑liner switch among `seresnext101_32x4d`, `tf_efficientnet_b4_ns`, `resnet200d`, `convnext_base`, etc.

---

## Repository Layout

Learning-Sparse-Label-Couplings-for-Multilabel-Chest-X-Ray-Diagnosis/
├─ crxml/
│  ├─ init.py
│  ├─ train.py           # CLI entrypoint for K-fold train + test inference
│  ├─ engine.py          # train/validate/predict loops
│  ├─ data.py            # XRayDataset & transforms
│  ├─ losses.py          # ASL / BCE (pos_weight)
│  ├─ utils.py           # MIS splits, metrics, seeding, constants
│  └─ models/
│     ├─ init.py
│     ├─ backbone.py     # timm backbone factory
│     ├─ refiner.py      # LabelGraphRefiner (sparse couplings via L1)
│     └─ ema.py          # EMA wrapper
├─ scripts/
│  ├─ train.sh
│  └─ infer.sh
├─ requirements.txt
├─ README.md
├─ LICENSE (MIT)
└─ .gitignore

> If you fork this into a different package name, replace `crxml` below accordingly.

---

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: CUDA‑specific torch build from pytorch.org if needed

Requirements (key): torch, torchvision, timm==0.9.16, pandas, numpy, scikit-learn, opencv-python, Pillow, iterative-stratification.

⸻

Data Format
	•	train_csv must include a column Image_name and one or more binary label columns (0/1).
	•	train_dir and test_dir are folders of images referenced by Image_name.

Example (CSV header):

Image_name,Atelectasis,Cardiomegaly,Consolidation,Edema, ...
000001.jpg,0,1,0,0, ...
000002.jpg,1,0,1,0, ...

Any multilabel chest‑X‑ray dataset with this structure should work. Ensure labels are 0/1 (the loader coerces numerics and clips).

⸻

Quick Start

Train K folds and write test predictions to submission.csv:

python -m crxml.train \
  --train_csv /path/train.csv \
  --train_dir /path/train_images \
  --test_dir  /path/test_images \
  --out_csv   ./submission.csv \
  --labels "Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices" \
  --model tf_efficientnet_b4_ns \
  --img_size 380 --epochs 3 --batch_size 24 --lr 2e-4 --weight_decay 1e-4 \
  --warmup_epochs 1 --folds 3 --seed 42 --loss asl --ema --tta --num_workers 4 --color_jitter

Common switches
	•	--model: any timm model name (examples above)
	•	--loss {asl,bce}: choose ASL or BCE (BCE auto‑computes pos_weight per fold)
	•	--no_amp: disable mixed precision if you encounter NaNs
	•	--ema/--no-ema: enable/disable EMA tracking of weights
	•	--tta/--no-tta: enable/disable flip TTA at test time
	•	--label_smooth 0.05: optional label smoothing during training

Outputs:
	•	Best fold weights → ./outputs/{model}_fold{K}.pth
	•	Final predictions → submission.csv with columns Image_name + labels

⸻

Method: Sparse Label Couplings

We augment logits with a learned, sparse coupling matrix (A \in \mathbb{R}^{C\times C}) (zero diagonal). Given logits (z) for (C) labels and (p=\sigma(z)),
[
z’ = z + \alpha, p A, \qquad A_{ii}=0.
]
We apply an L1 penalty (\lambda \lVert A \rVert_1) to encourage sparsity, learning a small set of meaningful co‑occurrences (e.g., Edema ↔ Pleural Effusion) while keeping the module lightweight and inference‑friendly.

Why this works
	•	Exploits clinically plausible co‑occurrence structure without hardcoding rules.
	•	Robust to noisy/missing labels—refiner operates on probabilities, not ground‑truth.
	•	End‑to‑end differentiable; adds negligible computational overhead.

⸻

Visualize Learned Couplings

After a run, you can inspect the learned (A) matrix:

import torch, matplotlib.pyplot as plt
import pandas as pd

ckpt = torch.load("./outputs/tf_efficientnet_b4_ns_fold1.pth", map_location="cpu")
A = ckpt["refiner"]["A"].numpy()  # shape (C, C)
labels = [
  'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
  'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
  'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]
plt.imshow(A, cmap="bwr"); plt.colorbar(); plt.xticks(range(len(labels)), labels, rotation=90); plt.yticks(range(len(labels)), labels); plt.title("Learned label couplings (A)"); plt.tight_layout(); plt.show()

To list strongest couplings per label:

import numpy as np
for i, li in enumerate(labels):
    idx = np.argsort(-np.abs(A[i]))
    print(li, "→", [(labels[j], float(A[i,j])) for j in idx[:5] if j!=i])


⸻

Command‑Line Arguments

Argument	Type/Default	Description
--train_csv	str, required	CSV with Image_name + label columns
--train_dir	str, required	Directory of training images
--test_dir	str, required	Directory of test images
--out_csv	str, ./submission.csv	Output CSV path
--out_dir	str, ./outputs	Where to save best fold weights
--labels	str (comma‑sep)	Explicit list of label columns; auto‑detects if omitted
--model	str, seresnext101_32x4d	timm backbone name
--img_size	int, 380	Square resize/crop size
--epochs	int, 3	Training epochs per fold
--batch_size	int, 24	Batch size
--lr	float, 2e-4	Base learning rate (AdamW)
--weight_decay	float, 1e-4	Weight decay
--warmup_epochs	int, 1	Warmup over N train steps
--folds	int, 3	Number of folds (MIS or fallback KFold)
--seed	int, 42	RNG seed
--label_smooth	float, 0.0	Label smoothing epsilon
--loss	{asl,bce}	Loss selection
--ema / --no-ema	flag, True	Enable EMA tracking
--tta / --no-tta	flag, True	Enable flip TTA
--num_workers	int, 4	Dataloader workers
--color_jitter	flag, False	Light color jitter in training augs
--no_amp	flag, False	Disable mixed precision


⸻

Reproducing a Strong Baseline

A good starting configuration is the Quick Start above (B4‑NS @ 380px, ASL, EMA, TTA). For larger backbones (e.g., convnext_base), consider --batch_size 16 and/or a smaller --img_size if you hit OOM.

Ablations to try
	•	--loss bce (pos_weight) vs --loss asl
	•	--ema on/off
	•	Refiner on/off (edit code to skip initializing LabelGraphRefiner)
	•	--tta on/off

⸻

Tips & Troubleshooting
	•	NaNs: set --no_amp; double‑check input CSV for non‑numeric labels; try a lower LR (e.g., 1e-4).
	•	Class imbalance: prefer --loss bce which auto‑computes pos_weight per fold.
	•	Slow I/O: increase --num_workers, ensure images are local (not network‑mounted).
	•	Overfitting: increase aug strength (enable --color_jitter, raise rotation degrees in data.py), add dropout in backbone.py via drop_rate.
	•	Memory: reduce --img_size or --batch_size; try lighter backbones.

⸻

Development
	•	The code is structured for readability and swapping components.
	•	Add new backbones in crxml/models/backbone.py (or pass any timm name).
	•	To create an infer‑only CLI, copy the predict flow from train.py into infer.py and load a chosen *_foldK.pth.

⸻

Citation

If you use this repo, please cite:

 incoming
