"""Loss and LASO metrics.

Loss (paper B.2, "same loss formulation as LASO"): BCE + Dice, equal weights. The Dice term follows the
official LASO HM_Loss (positive + negative Dice).

Metrics replicate the official LASO evaluation (eval_lyc.py::evaluate): per-sample aIoU (mean IoU over
20 thresholds in [0,1], GT binarised at 0.5), ROC-AUC, SIM (histogram intersection of the L1-normalised
maps) and MAE, averaged over test samples.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = logits.sigmoid()
    eps = 1e-6
    dice_pos = ((pred * target).sum(1) + eps) / ((pred + target).sum(1) + eps)
    dice_neg = (((1 - pred) * (1 - target)).sum(1) + eps) / ((2 - pred - target).sum(1) + eps)
    dice = (1.5 - dice_pos - dice_neg).mean()
    return bce + dice, bce.detach(), dice.detach()


# ---------------------------------------------------------------- official (numpy / sklearn)
_THRES = np.linspace(0, 1, 20)


def sample_metrics(pred, gt):
    """pred, gt: 1-D numpy arrays for one sample -> (aIoU, AUC, SIM, MAE); aIoU/AUC NaN if no positives."""
    sim = np.minimum(pred / (pred.sum() + 1e-12), gt / (gt.sum() + 1e-12)).sum()
    mae = np.abs(pred - gt).mean()
    t = (gt >= 0.5).astype(int)
    if t.sum() == 0:
        return np.nan, np.nan, sim, mae
    auc = roc_auc_score(t, pred)
    ious = []
    for th in _THRES:
        m = (pred >= th).astype(int)
        ious.append((m & t).sum() / (m | t).sum())
    return float(np.mean(ious)), auc, sim, mae


def batch_metrics(pred, gt):
    """pred, gt: (B, N) numpy -> array (B, 4)"""
    return np.array([sample_metrics(p, g) for p, g in zip(pred, gt)], dtype=np.float64)


# ---------------------------------------------------------------- fast GPU approximation (monitoring)
@torch.no_grad()
def batch_metrics_torch(pred, gt):
    """Same definitions on GPU (AUC via rank statistic). pred, gt: (B,N) tensors -> (B,4) tensor."""
    pred, gt = pred.float(), gt.float()
    sim = torch.minimum(pred / (pred.sum(1, keepdim=True) + 1e-12),
                        gt / (gt.sum(1, keepdim=True) + 1e-12)).sum(1)
    mae = (pred - gt).abs().mean(1)
    t = gt >= 0.5
    npos = t.sum(1).float()
    nneg = t.shape[1] - npos
    ranks = pred.argsort(1).argsort(1).float() + 1
    auc = ((ranks * t).sum(1) - npos * (npos + 1) / 2) / (npos * nneg).clamp(min=1)
    th = torch.linspace(0, 1, 20, device=pred.device).view(1, -1, 1)
    m = pred.unsqueeze(1) >= th
    inter = (m & t.unsqueeze(1)).sum(2).float()
    union = (m | t.unsqueeze(1)).sum(2).float()
    aiou = (inter / union.clamp(min=1)).mean(1)
    none = npos == 0
    aiou[none], auc[none] = float("nan"), float("nan")
    return torch.stack([aiou, auc, sim, mae], 1)


def summarize(m):
    """m: (S,4) array -> dict with the paper's units (aIoU/AUC in %, SIM in %, MAE raw)."""
    m = np.asarray(m, dtype=np.float64)
    return dict(aIoU=100 * np.nanmean(m[:, 0]), AUC=100 * np.nanmean(m[:, 1]),
                SIM=100 * np.nanmean(m[:, 2]), MAE=float(np.nanmean(m[:, 3])), n=int(len(m)))
