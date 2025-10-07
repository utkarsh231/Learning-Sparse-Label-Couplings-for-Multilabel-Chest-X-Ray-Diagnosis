import torch
import torch.nn as nn
import numpy as np

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))
        loss = targets * log_pos + (1 - targets) * log_neg
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gp * targets + self.gn * (1 - targets)
        loss *= (1 - pt).pow(gamma)
        return -loss.mean()


def make_bce_with_pos_weight(tr_df, label_cols, device):
    pos = tr_df[label_cols].sum(axis=0).values.astype(np.float32)
    neg = len(tr_df) - pos
    w = (neg + 1e-6) / (pos + 1e-6)
    w = np.clip(w, 1.0, 10.0)
    pos_weight = torch.tensor(w, dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)