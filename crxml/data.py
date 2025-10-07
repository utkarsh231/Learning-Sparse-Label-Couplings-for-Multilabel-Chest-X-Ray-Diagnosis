from __future__ import annotations
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .utils import IMAGENET_MEAN, IMAGENET_STD

class XRayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, label_cols: list[str], img_size: int,
                 train: bool = True, label_smoothing: float = 0.0, color_jitter: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.img_size = img_size
        self.train = train
        self.eps = label_smoothing

        if train:
            augs = [
                T.RandomResizedCrop(self.img_size, scale=(0.84, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
            if color_jitter:
                augs.append(T.ColorJitter(brightness=0.05, contrast=0.05))
            augs += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
            self.tfm = T.Compose(augs)
        else:
            self.tfm = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def _smooth(self, y: torch.Tensor) -> torch.Tensor:
        if self.eps <= 0:
            return y
        return y * (1.0 - self.eps) + 0.5 * self.eps

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["Image_name"])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tfm(Image.fromarray(img))
        if all(c in row.index for c in self.label_cols):
            y = row[self.label_cols].values.astype(np.float32)
            y = np.nan_to_num(y, nan=0.0)
            y = np.clip(y, 0.0, 1.0)
            y = torch.tensor(y)
            if self.train and self.eps > 0:
                y = self._smooth(y)
            return x, y
        else:
            return row["Image_name"], x