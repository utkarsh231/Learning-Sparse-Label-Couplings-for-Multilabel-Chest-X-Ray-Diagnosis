import torch
import torch.nn as nn

class LabelGraphRefiner(nn.Module):
    """Learn a dense label coupling matrix A (with zero diagonal) and refine logits.
    logits' = logits + alpha * sigmoid(logits) @ A
    """
    def __init__(self, num_classes: int, alpha: float = 0.3, l1: float = 1e-3):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_classes, num_classes))
        self.alpha = alpha
        self.l1 = l1

    def forward(self, logits):
        A = self.A - torch.diag(torch.diag(self.A))  # zero diagonal
        p = torch.sigmoid(logits)                    # B x C
        msg = p @ A                                  # B x C
        refined = logits + self.alpha * msg
        reg = self.l1 * (A.abs().sum())
        return refined, reg