import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader
import evaluate
import os
import shutil
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
from metrics import Metrics
import pandas as pd
import argparse
import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib
import json
from my_datasets import CityscapesDataset
import matplotlib.pyplot as plt
import datasets
# ----------------------------------------------------------------------------------------------------------------------
class OHEMCrossEntropy2D(nn.Module):
    def __init__(self, topk=0.7, ignore_index=255, n_min=100000):
        super(OHEMCrossEntropy2D, self).__init__()
        self.topk = topk
        self.ignore_index = ignore_index
        self.n_min = n_min

    def forward(self, input, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        pixel_losses = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        pixel_losses_flatten = pixel_losses.view(-1)
        topk_num = max(int(self.topk * pixel_losses_flatten.numel()), self.n_min)
        top_k_losses, _ = torch.topk(pixel_losses_flatten, topk_num)
        return top_k_losses.mean()


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7,
                 aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        n_min = labels[labels != self.ignore_label].numel() // preds.shape[1]
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


def OHEM_loss(loss_basic, preds, labels, CM_Weights):
    n_classes = CM_Weights.shape[0]
    ignore_label = 255
    preds_per = preds.permute(0, 2, 3, 1)
    predictions = preds_per.reshape(-1, n_classes)
    lbl = torch.flatten(labels)
    mask = lbl != ignore_label
    predictions = predictions[mask]
    lbl = lbl[mask]
    return loss_basic(predictions, lbl)


def CAL_loss(loss_basic, preds, labels, CM_Weights):
    n_classes = CM_Weights.shape[0]
    ignore_label = 255
    eps = torch.tensor(1e-6).to(preds.device)
    preds_per = preds.permute(0, 2, 3, 1)
    predictions = preds_per.reshape(-1, n_classes)
    lbl = torch.flatten(labels)
    mask = lbl != ignore_label
    predictions = predictions[mask]
    lbl = lbl[mask]
    loss1 = loss_basic(predictions, lbl)
    if CM_Weights.shape[0] != n_classes:
        raise ValueError(f"CM_Weights must be {n_classes}x{n_classes}")
    CM_Weight_tensor = torch.from_numpy(CM_Weights).to(preds.device).float()
    mask_eye = torch.eye(n_classes).to(preds.device)
    mask_one_hot = mask_eye[lbl]
    mask_zero_hot = 1 - mask_one_hot
    loss_cal = -torch.sum(torch.log((1 - predictions) + eps) * mask_zero_hot * CM_Weight_tensor[lbl, :], dim=1)
    return loss1 + torch.mean(loss_cal)

# ----------------------------------------------------------------------------------------------------------------------
class OHEMCrossEntropy(nn.Module):
    """Standard OHEM Cross-Entropy for Semantic Segmentation. Expects raw logits."""

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh.to(loss.device)]
        if loss_hard.numel() < self.min_kept:
            loss_hard, _ = loss.topk(self.min_kept)
        return loss_hard.mean()


class CALoss(nn.Module):
    """
    Confusion-Aware Loss (CAL). Expects probabilities [B, C, H, W] + confusion matrix weights.
    """

    def __init__(self, n_classes=19, ignore_index=255, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds, labels, cm_weights):
        device = preds.device
        B, C, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.view(-1)
        mask = labels != self.ignore_index
        preds = preds[mask]
        labels = labels[mask]
        if labels.numel() == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)
        if not isinstance(cm_weights, torch.Tensor):
            cm_weights = torch.from_numpy(cm_weights).float().to(device)
        target_probs = preds.gather(1, labels.unsqueeze(1)).squeeze()
        loss_ce = -torch.log(target_probs + self.eps).mean()
        mask_eye = torch.eye(C, device=device)
        mask_zero_hot = 1 - mask_eye[labels]
        confusion_penalty = -torch.sum(
            torch.log((1 - preds) + self.eps) * mask_zero_hot * cm_weights[labels, :], dim=1)
        return loss_ce + confusion_penalty.mean()


# ----------------------------------------------------------------------------------------------------------------------
class TverskyLoss(nn.Module):
    """
    Tversky Loss for semantic segmentation. Expects probabilities [B, C, H, W].

    Generalization of Dice Loss with asymmetric FP/FN weighting:
        - alpha=0.5, beta=0.5  =>  Dice Loss
        - beta > alpha          =>  Penalizes FN more (good for rare classes)

    Args:
        alpha (float):      FP weight. Default: 0.3.
        beta (float):       FN weight. Default: 0.7.
        smooth (float):     Smoothing constant. Default: 1e-6.
        ignore_index (int): Ignored label (Cityscapes void = 255). Default: 255.
        n_classes (int):    Number of classes. Default: 19.
        reduction (str):    'mean' or 'sum'. Default: 'mean'.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6,
                 ignore_index: int = 255, n_classes: int = 19, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        device = preds.device
        B, C, H, W = preds.shape
        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = preds.softmax(dim=1)
        valid_mask = (labels != self.ignore_index)
        labels_clean = labels.clone()
        labels_clean[~valid_mask] = 0
        labels_one_hot = F.one_hot(labels_clean.long(), num_classes=C).permute(0, 3, 1, 2).float()
        valid_mask_4d = valid_mask.unsqueeze(1).float()
        preds = preds * valid_mask_4d
        labels_one_hot = labels_one_hot * valid_mask_4d
        preds_flat = preds.view(B, C, -1)
        targets_flat = labels_one_hot.view(B, C, -1)
        TP = (preds_flat * targets_flat).sum(dim=(0, 2))
        FP = (preds_flat * (1 - targets_flat)).sum(dim=(0, 2))
        FN = ((1 - preds_flat) * targets_flat).sum(dim=(0, 2))
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss_per_class = 1.0 - tversky_index
        return loss_per_class.mean() if self.reduction == 'mean' else loss_per_class.sum()


class TverskyCALoss(nn.Module):
    """
    Tversky Loss + Confusion-Aware Loss (CAL). Expects probabilities + confusion matrix.

    Args:
        alpha (float):      Tversky FP weight. Default: 0.3.
        beta (float):       Tversky FN weight. Default: 0.7.
        cal_weight (float): Scale factor for the CAL term. Default: 1.0.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, cal_weight: float = 1.0,
                 n_classes: int = 19, ignore_index: int = 255, smooth: float = 1e-6, eps: float = 1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth,
                                   ignore_index=ignore_index, n_classes=n_classes)
        self.cal = CALoss(n_classes=n_classes, ignore_index=ignore_index, eps=eps)
        self.cal_weight = cal_weight

    def forward(self, preds: Tensor, labels: Tensor, cm_weights=None) -> Tensor:
        loss_tversky = self.tversky(preds, labels)
        if cm_weights is not None:
            return loss_tversky + self.cal_weight * self.cal(preds, labels, cm_weights)
        return loss_tversky


# ----------------------------------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for semantic segmentation. Expects raw logits [B, C, H, W].

    Down-weights easy (well-classified) pixels and focuses on hard examples via
    a modulating factor (1 - p_t)^gamma applied to the cross-entropy loss.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma (float):       Focusing parameter. gamma=0 reduces to CE. Default: 2.0.
        alpha (Tensor|None): Per-class weight tensor [C] for class imbalance. Default: None.
        ignore_index (int):  Ignored label (Cityscapes void = 255). Default: 255.
        reduction (str):     'mean' or 'sum'. Default: 'mean'.
    """

    def __init__(self, gamma: float = 2.0, alpha: Tensor = None,
                 ignore_index: int = 255, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            preds:  Tensor [B, C, H, W] — raw logits.
            labels: Tensor [B, H, W]    — ground truth class indices.
        Returns:
            Scalar focal loss.
        """
        device = preds.device
        # Per-pixel CE (ignore_index handled natively)
        ce_loss = F.cross_entropy(
            preds, labels,
            weight=self.alpha.to(device) if self.alpha is not None else None,
            ignore_index=self.ignore_index,
            reduction='none'
        )  # [B, H, W]

        # Compute p_t: probability of the ground-truth class
        with torch.no_grad():
            probs = preds.softmax(dim=1)                              # [B, C, H, W]
            labels_clamped = labels.clone()
            labels_clamped[labels == self.ignore_index] = 0
            p_t = probs.gather(1, labels_clamped.unsqueeze(1)).squeeze(1)  # [B, H, W]

        # Focal modulation
        focal = ((1.0 - p_t) ** self.gamma) * ce_loss               # [B, H, W]

        # Zero out ignored pixels (ce_loss is already 0 there, but p_t may not be)
        valid_mask = (labels != self.ignore_index).float()
        focal = focal * valid_mask

        if self.reduction == 'mean':
            return focal.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            raise ValueError(f"Unknown reduction '{self.reduction}'")


# ----------------------------------------------------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation. Expects probabilities [B, C, H, W].

    Maximizes overlap between predicted and ground-truth masks. Equivalent to
    TverskyLoss with alpha=beta=0.5. Naturally handles class imbalance since it
    normalizes by region size, giving rare classes equal contribution.

    Formula per class c:
        Dice_c = (2*TP_c + smooth) / (2*TP_c + FP_c + FN_c + smooth)
        Loss_c = 1 - Dice_c

    Args:
        smooth (float):     Smoothing constant. Default: 1e-6.
        ignore_index (int): Ignored label (Cityscapes void = 255). Default: 255.
        n_classes (int):    Number of semantic classes. Default: 19.
        reduction (str):    'mean' or 'sum'. Default: 'mean'.
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255,
                 n_classes: int = 19, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            preds:  Tensor [B, C, H, W] — probabilities (after softmax) or raw logits.
                    Softmax is applied internally if values are outside [0, 1].
            labels: Tensor [B, H, W]    — ground truth class indices.
        Returns:
            Scalar Dice loss.
        """
        device = preds.device
        B, C, H, W = preds.shape

        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = preds.softmax(dim=1)

        valid_mask = (labels != self.ignore_index)
        labels_clean = labels.clone()
        labels_clean[~valid_mask] = 0

        labels_one_hot = F.one_hot(labels_clean.long(), num_classes=C).permute(0, 3, 1, 2).float()

        valid_mask_4d = valid_mask.unsqueeze(1).float()
        preds = preds * valid_mask_4d
        labels_one_hot = labels_one_hot * valid_mask_4d

        preds_flat   = preds.view(B, C, -1)           # [B, C, N]
        targets_flat = labels_one_hot.view(B, C, -1)   # [B, C, N]

        # Sum over batch and spatial dims -> [C]
        intersection = (preds_flat * targets_flat).sum(dim=(0, 2))
        pred_sum     = preds_flat.sum(dim=(0, 2))
        target_sum   = targets_flat.sum(dim=(0, 2))

        dice_coeff = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        loss_per_class = 1.0 - dice_coeff  # [C]

        if self.reduction == 'mean':
            return loss_per_class.mean()
        elif self.reduction == 'sum':
            return loss_per_class.sum()
        else:
            raise ValueError(f"Unknown reduction '{self.reduction}'")


# ----------------------------------------------------------------------------------------------------------------------
class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss for semantic segmentation. Expects raw logits [B, C, H, W].

    Combines the hard-example mining of Focal Loss with the class-imbalance
    handling of Dice Loss. This combination is used in nnU-Net and other
    state-of-the-art segmentation frameworks.

    Loss = FocalLoss(logits) + dice_weight * DiceLoss(softmax(logits))

    Args:
        gamma (float):       Focal focusing parameter. Default: 2.0.
        alpha (Tensor|None): Per-class weight tensor [C] for focal term. Default: None.
        dice_weight (float): Scale factor for the Dice term. Default: 1.0.
        smooth (float):      Dice smoothing constant. Default: 1e-6.
        n_classes (int):     Number of semantic classes. Default: 19.
        ignore_index (int):  Ignored label. Default: 255.
    """

    def __init__(self, gamma: float = 2.0, alpha: Tensor = None, dice_weight: float = 1.0,
                 smooth: float = 1e-6, n_classes: int = 19, ignore_index: int = 255):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
        self.dice  = DiceLoss(smooth=smooth, ignore_index=ignore_index, n_classes=n_classes)
        self.dice_weight = dice_weight

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            preds:  Tensor [B, C, H, W] — raw logits.
            labels: Tensor [B, H, W]    — ground truth class indices.
        Returns:
            Scalar combined focal + dice loss.
        """
        return self.focal(preds, labels) + self.dice_weight * self.dice(preds, labels)


# ----------------------------------------------------------------------------------------------------------------------
class ClassBalancedCrossEntropy(nn.Module):
    """
    Class-Balanced Cross-Entropy Loss for semantic segmentation.
    Expects raw logits [B, C, H, W].

    Addresses class imbalance by computing per-class pixel frequencies from the
    current batch and deriving inverse-frequency weights on the fly. This means
    no pre-computation of dataset statistics is needed — the weights adapt to
    whatever class distribution appears in each batch.

    Two weighting schemes are supported via the `mode` parameter:

        "inverse_freq"   — w_c = 1 / (freq_c + eps)
            Classic inverse-frequency weighting. Rare classes receive very large
            weights, which can sometimes destabilise training on highly imbalanced
            datasets. Recommended when class imbalance is moderate.

        "effective_num"  — w_c = (1 - beta) / (1 - beta^n_c)   [Cui et al., CVPR 2019]
            Effective Number of Samples weighting. Uses a hyperparameter beta
            (default 0.9999) to smooth the inverse-frequency curve, making it more
            robust when some classes have very few samples. Recommended for
            severely imbalanced datasets (e.g. Cityscapes rare classes).

    In both modes the weights are L1-normalised (sum to 1) before being passed to
    nn.CrossEntropyLoss, so the loss magnitude stays comparable to standard CE.

    Args:
        mode (str):         Weighting scheme: 'inverse_freq' or 'effective_num'. Default: 'effective_num'.
        beta (float):       Smoothing parameter for 'effective_num' mode. Default: 0.9999.
        ignore_index (int): Label to ignore (Cityscapes void = 255). Default: 255.
        eps (float):        Small constant to avoid division by zero. Default: 1e-6.
        n_classes (int):    Number of semantic classes. Default: 19.
    """

    def __init__(
        self,
        mode: str = 'effective_num',
        beta: float = 0.9999,
        ignore_index: int = 255,
        eps: float = 1e-6,
        n_classes: int = 19,
    ):
        super().__init__()
        valid_modes = ('inverse_freq', 'effective_num')
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        self.mode = mode
        self.beta = beta
        self.ignore_index = ignore_index
        self.eps = eps
        self.n_classes = n_classes

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            preds:  Tensor [B, C, H, W] — raw logits.
            labels: Tensor [B, H, W]    — ground truth class indices (long).
        Returns:
            Scalar class-balanced cross-entropy loss.
        """
        device = preds.device
        C = preds.shape[1]

        # --- Count valid pixels per class in this batch ---
        valid_mask = labels != self.ignore_index           # [B, H, W]
        labels_valid = labels[valid_mask]                  # [N_valid]

        # Bincount over classes; clamp to avoid log(0) in weight computation
        counts = torch.bincount(labels_valid, minlength=C).float()  # [C]

        # --- Compute per-class weights ---
        if self.mode == 'inverse_freq':
            # w_c = 1 / (freq_c + eps);  freq_c = counts_c / sum(counts)
            total = counts.sum().clamp(min=1.0)
            freq = counts / total
            weights = 1.0 / (freq + self.eps)

        else:  # 'effective_num'
            # w_c = (1 - beta) / (1 - beta^n_c),  n_c = pixel count for class c
            # For classes with zero pixels we assign weight 0 (they don't appear
            # in the batch, so their contribution to the loss would be 0 anyway).
            effective_num = 1.0 - torch.pow(self.beta, counts)      # [C]
            weights = torch.where(
                counts > 0,
                (1.0 - self.beta) / (effective_num + self.eps),
                torch.zeros_like(counts)
            )

        # --- L1-normalise weights so their sum == 1 ---
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        # If all weights are 0 (degenerate batch), fall back to uniform weights
        else:
            weights = torch.ones(C, device=device) / C

        weights = weights.to(device)

        # --- Weighted cross-entropy (ignore_index handled natively) ---
        return F.cross_entropy(
            preds, labels,
            weight=weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )


# ----------------------------------------------------------------------------------------------------------------------
def evaluate_model(model, dataloader, processor, metric, device, previous_cm):
    N_CLASSES_CITYSCAPES = 19
    metrics = Metrics(N_CLASSES_CITYSCAPES, 255, device)
    model.eval()
    total_eval_loss = 0.0
    num_batches = 0
    eval_loss_fn = OhemCrossEntropy()
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            img_resized = F.interpolate(pixel_values, size=(512, 1024), mode='bilinear', align_corners=False)
            img_padded = F.pad(img_resized, (0, 0, 256, 256), value=0)
            outputs = model(pixel_values=img_padded)
            logits = outputs.logits
            logits_unpadded = logits[:, :, 64:192, :]
            upsampled_logits = F.interpolate(
                logits_unpadded, size=labels.shape[-2:], mode="bicubic", align_corners=False
            ).softmax(dim=1)
            eval_loss = CAL_loss(eval_loss_fn, upsampled_logits, labels, previous_cm)
            total_eval_loss += eval_loss.item()
            metrics.update(upsampled_logits, labels)
            predicted_masks = upsampled_logits.argmax(dim=1)
            metric.add_batch(
                predictions=predicted_masks.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )
            num_batches += 1

    avg_eval_loss = total_eval_loss / num_batches if num_batches > 0 else 0.0
    ious, miou, CM = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    return miou, ious, CM, macc, acc, mf1, f1, avg_eval_loss

# ----------------------------------------------------------------------------------------------------------------------
def evaluate_cityscapes(model_name, dataloader, device="cuda"):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    metric = evaluate.load("mean_iou")
    num_classes = 19
    print(f"Starting evaluation for {model_name}...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            img_resized = F.interpolate(pixel_values, size=(512, 1024), mode='bilinear')
            img_padded = F.pad(img_resized, (0, 0, 256, 256), value=0)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=img_padded)
            logits = outputs.logits
            logits_unpadded = logits[:, :, 64:192, :]
            upsampled_logits = nn.functional.interpolate(
                logits_unpadded, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predictions = upsampled_logits.argmax(dim=1)
            metric.add_batch(
                predictions=predictions.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy())
    return metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=False)

# ----------------------------------------------------------------------------------------------------------------------
def evaluate_model_sw(model, dataloader, device, criterion, cm_weights):
    """Sliding Window Evaluation with Memory Optimization."""
    n_classes = 19
    metrics = Metrics(n_classes, 255, device)
    model.eval()
    total_eval_loss = 0.0
    num_batches = 0
    WINDOW_SIZE = 1024
    STRIDE = 512

    window_weight = torch.ones((1, 1, 1024, 1024), device=device)
    ramp = torch.linspace(0, 1, 128, device=device)
    window_weight[:, :, :128, :] *= ramp.view(1, 1, 128, 1)
    window_weight[:, :, -128:, :] *= ramp.flip(0).view(1, 1, 128, 1)

    progress_bar = tqdm(dataloader, desc="Evaluating (SW)", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            B, C, H, W = pixel_values.shape
            feat_h, feat_w = H // 4, W // 4
            win_w = WINDOW_SIZE // 4

            full_logits = torch.zeros((B, n_classes, feat_h, feat_w), device=device)
            count_mask = torch.zeros((B, 1, feat_h, feat_w), device=device)

            for x in range(0, W - WINDOW_SIZE + 1, STRIDE):
                window_input = pixel_values[:, :, :, x:x + WINDOW_SIZE]
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=window_input)
                    logits = outputs.logits
                lx = x // 4
                full_logits[:, :, :, lx:lx + win_w] += logits * window_weight[:, :, ::4, ::4]
                count_mask[:, :, :, lx:lx + win_w] += window_weight[:, :, ::4, ::4]

            full_logits /= (count_mask + 1e-6)
            upsampled_logits = F.interpolate(full_logits, size=labels.shape[-2:],
                                             mode="bilinear", align_corners=False)

            # -------------------------------------------------------------------
            # Loss routing:
            #   - CALoss, TverskyCALoss         -> probabilities + cm_weights
            #   - TverskyLoss, DiceLoss          -> probabilities only
            #   - FocalLoss, FocalDiceLoss,
            #     OHEMCrossEntropy               -> raw logits
            # -------------------------------------------------------------------
            if isinstance(criterion, (CALoss, TverskyCALoss)):
                loss = criterion(upsampled_logits.softmax(dim=1), labels, cm_weights)
            elif isinstance(criterion, (TverskyLoss, DiceLoss)):
                loss = criterion(upsampled_logits.softmax(dim=1), labels)
            else:  # FocalLoss, FocalDiceLoss, OHEMCrossEntropy, ClassBalancedCrossEntropy
                loss = criterion(upsampled_logits, labels)

            total_eval_loss += loss.item()
            metrics.update(upsampled_logits.softmax(dim=1), labels)
            num_batches += 1

            if num_batches % 10 == 0:
                torch.cuda.empty_cache()

    ious, miou, CM = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    return miou, ious, CM, macc, acc, mf1, f1, total_eval_loss / num_batches


# ----------------------------------------------------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['N_CLASSES_CITYSCAPES'] = int(config.get('N_CLASSES_CITYSCAPES', 19))
    config['NUM_EPOCHS'] = int(config.get('NUM_EPOCHS', 500))
    config['BATCH_SIZE'] = int(config.get('BATCH_SIZE', 8))
    return config


def log_config_to_file(log_file_path, config):
    """Logs the training configuration to the top of the CSV log file."""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    if content.startswith("# CONFIG START"):
        return
    config_lines = ["# CONFIG START"]
    for k, v in config.items():
        config_lines.append(f"# {k}: {v}")
    config_lines.append("# CONFIG END\n")
    header_line = ""
    lines = content.split('\n')
    for line in lines:
        if line.startswith("Epoch\t"):
            header_line = line
            break
    content_after_header = '\n'.join(lines[lines.index(header_line):]) if header_line else content
    with open(log_file_path, 'w') as f:
        f.write("\n".join(config_lines) + content_after_header)


def log_epoch_record(log_file_path, epoch, avg_train_loss, avg_eval_loss, epoch_time_seconds,
                     optimizer, current_miou, ious, macc, mf1):
    current_miou_scalar = current_miou.item() if isinstance(current_miou, (np.ndarray, torch.Tensor)) else current_miou
    macc_scalar = macc.item() if isinstance(macc, (np.ndarray, torch.Tensor)) else macc
    mf1_scalar  = mf1.item()  if isinstance(mf1,  (np.ndarray, torch.Tensor)) else mf1
    ious_list = np.asarray(ious).flatten().tolist()
    new_row = {
        'Epoch':      epoch + 1,
        'Train_Loss': f"{avg_train_loss:.4f}",
        'Eval_Loss':  f"{avg_eval_loss:.4f}",
        'mIoU':       f"{current_miou_scalar:.4f}",
        'mAcc':       f"{macc_scalar:.4f}",
        'AvgF1':      f"{mf1_scalar:.4f}",
        'Time_s':     f"{epoch_time_seconds:.2f}",
        'LR':         f"{optimizer.param_groups[0]['lr']:.6e}"
    }
    for i, v in enumerate(ious_list):
        new_row[f'Class_iou_{i}'] = f"{v:.4f}"
    df_new_row = pd.DataFrame([new_row])
    write_header = not log_file_path.exists()
    df_new_row.to_csv(log_file_path, mode='a', header=write_header, index=False, sep='\t')


def show_pixel_values(pixels_batch):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = np.transpose(np.clip((pixels_batch[0].cpu() * std + mean).numpy(), 0, 1), (1, 2, 0))
    plt.imshow((img_np * 255).astype(np.uint8))
    plt.axis('off')
    plt.show()


def clear_cache_if_full(max_gb=3):
    cache_path = Path.home() / ".cache/huggingface"
    if not cache_path.exists():
        return
    total_size = sum(f.stat().st_size for f in cache_path.glob('**/*') if f.is_file())
    if total_size / (1024 ** 3) > max_gb:
        print(f"\n[Cache Alert] Cache >{max_gb} GB. Clearing...")
        try:
            shutil.rmtree(cache_path)
            print("[Cache Alert] Cleared.")
        except Exception as e:
            print(f"Error clearing cache: {e}")


def Calc_Weights(confmat, confusion_type):
    if confusion_type == 'fn':
        TP_FN_FP = confmat.copy()
    elif confusion_type == 'fn_fp':
        TP_FN_FP = confmat + confmat.T
        np.fill_diagonal(TP_FN_FP, np.diag(confmat))
    else:
        raise ValueError(f"confusion_type must be 'fn' or 'fn_fp', got '{confusion_type}'")
    row_sums = TP_FN_FP.sum(axis=1, keepdims=True)
    return np.divide(TP_FN_FP, row_sums, out=np.zeros_like(TP_FN_FP), where=row_sums != 0)

# ----------------------------------------------------------------------------------------------------------------------
def main():

    datasets.disable_caching()
    print("Current path = ", os.getcwd())

    parser = argparse.ArgumentParser(description="SegFormer Cityscapes Training")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    CITYSCAPES_CLASS_NAMES = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
        "traffic_sign", "vegetation", "terrain", "sky", "person", "rider",
        "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    config = load_config(args.config_file)
    n_classes_cityscapes = len(CITYSCAPES_CLASS_NAMES)
    num_epochs     = config['NUM_EPOCHS']
    batch_size     = config['BATCH_SIZE']
    base_model_name = config['BASE_MODEL_NAME']
    experiment_name = config['EXPERIMENT_NAME']
    data_set_path  = config['DATA_SET_PATH']
    loss_type      = config['LOSS_TYPE']
    confusion_type = config['CONFUSION_TYPE']

    # -------------------------------------------------------------------
    # Loss selection
    # Supported LOSS_TYPE values:
    #   "cal_ohem"    — Confusion-Aware Loss (CE + CAL)          [probs + cm]
    #   "ohem"        — OHEM Cross-Entropy                       [logits]
    #   "tversky"     — Tversky Loss                             [probs]
    #   "tversky_cal" — Tversky + CAL                            [probs + cm]
    #   "focal"       — Focal Loss                               [logits]
    #   "dice"        — Dice Loss                                [probs]
    #   "focal_dice"    — Focal + Dice Loss                      [logits]
    #   "balanced_ce"   — Class-Balanced Cross-Entropy               [logits]
    #
    # Optional JSON keys (all have defaults if omitted):
    #   "TVERSKY_ALPHA":      FP weight (default 0.3)
    #   "TVERSKY_BETA":       FN weight (default 0.7)
    #   "TVERSKY_CAL_WEIGHT": CAL term weight in tversky_cal (default 1.0)
    #   "FOCAL_GAMMA":        Focusing parameter (default 2.0)
    #   "FOCAL_DICE_WEIGHT":  Dice term weight in focal_dice (default 1.0)
    #   "BALANCED_CE_MODE":  weighting scheme: inverse_freq or effective_num (default effective_num)
    #   "BALANCED_CE_BETA":  beta for effective_num mode (default 0.9999)
    # -------------------------------------------------------------------
    tversky_alpha     = float(config.get('TVERSKY_ALPHA', 0.3))
    tversky_beta      = float(config.get('TVERSKY_BETA',  0.7))
    tversky_cal_w     = float(config.get('TVERSKY_CAL_WEIGHT', 1.0))
    focal_gamma       = float(config.get('FOCAL_GAMMA', 2.0))
    focal_dice_weight   = float(config.get('FOCAL_DICE_WEIGHT', 1.0))
    balanced_ce_mode    = config.get('BALANCED_CE_MODE', 'effective_num')
    balanced_ce_beta    = float(config.get('BALANCED_CE_BETA', 0.9999))

    if loss_type == "cal_ohem":
        criterion = CALoss(n_classes=19)
    elif loss_type == "ohem":
        criterion = OHEMCrossEntropy()
    elif loss_type == "tversky":
        criterion = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                ignore_index=255, n_classes=n_classes_cityscapes)
        print(f"Using TverskyLoss  alpha={tversky_alpha}  beta={tversky_beta}")
    elif loss_type == "tversky_cal":
        criterion = TverskyCALoss(alpha=tversky_alpha, beta=tversky_beta, cal_weight=tversky_cal_w,
                                  n_classes=n_classes_cityscapes, ignore_index=255)
        print(f"Using TverskyCALoss  alpha={tversky_alpha}  beta={tversky_beta}  cal_weight={tversky_cal_w}")
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, ignore_index=255)
        print(f"Using FocalLoss  gamma={focal_gamma}")
    elif loss_type == "dice":
        criterion = DiceLoss(ignore_index=255, n_classes=n_classes_cityscapes)
        print("Using DiceLoss")
    elif loss_type == "focal_dice":
        criterion = FocalDiceLoss(gamma=focal_gamma, dice_weight=focal_dice_weight,
                                  n_classes=n_classes_cityscapes, ignore_index=255)
        print(f"Using FocalDiceLoss  gamma={focal_gamma}  dice_weight={focal_dice_weight}")
    elif loss_type == "balanced_ce":
        criterion = ClassBalancedCrossEntropy(
            mode=balanced_ce_mode,
            beta=balanced_ce_beta,
            ignore_index=255,
            n_classes=n_classes_cityscapes
        )
        print(f"Using ClassBalancedCrossEntropy  mode={balanced_ce_mode}  beta={balanced_ce_beta}")
    else:
        print(f"Invalid loss type: '{loss_type}'. "
              f"Valid options: cal_ohem, ohem, tversky, tversky_cal, focal, dice, focal_dice, balanced_ce")
        exit(1)

    experiment_dir = Path("./experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_file_path  = experiment_dir / "training_log.csv"
    checkpoint_path = experiment_dir / "last_checkpoint.pt"
    best_model_path = experiment_dir / "best_model.pt"

    print(f"Logging configuration to {log_file_path}")
    log_config_to_file(log_file_path, config)

    base_header = "Epoch\tTrain_Loss\tEval_Loss\tmIoU\tmAcc\tAvgF1\tTime_s\tLR"
    class_headers = "\t".join(CITYSCAPES_CLASS_NAMES)
    header_template = f"{base_header}\t{class_headers}\n"

    if not log_file_path.exists():
        with open(log_file_path, 'a') as f:
            f.write(header_template)
        print(f"Created new log file at {log_file_path}")
    else:
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
            if base_header.strip() not in content:
                with open(log_file_path, 'a') as f:
                    f.write(header_template)
        except Exception:
            pass
        print(f"Appending to existing log file at {log_file_path}")

    seg_config = SegformerConfig.from_pretrained(base_model_name)
    seg_config.num_labels = n_classes_cityscapes
    model = SegformerForSemanticSegmentation(seg_config)

    processor = SegformerImageProcessor.from_pretrained(base_model_name)
    processor.do_resize = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_epoch = 0
    best_miou = 0.0
    CM = np.eye(n_classes_cityscapes, dtype=float)

    initial_lr = 0.00001
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    metric = evaluate.load("mean_iou")

    train_dataset = CityscapesDataset(root_dir=data_set_path, processor=processor, split="train", fine_annotation=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CityscapesDataset(root_dir=data_set_path, processor=processor, split="val", fine_annotation=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=total_steps, lr_end=0.0, power=0.9
    )

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['best_miou']
        CM = checkpoint['CM']
        scheduler.T_max = num_epochs
        print(f"Resuming from Epoch {start_epoch + 1} with Best mIoU: {best_miou:.4f}")
    else:
        try:
            state_dict = SegformerForSemanticSegmentation.from_pretrained(base_model_name).state_dict()
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pre-trained weights for {base_model_name}.")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights. Error: {e}")
            print("Starting from scratch.")

    model.to(device)

    CM_Weight = Calc_Weights(CM, confusion_type)
    miou, ious, CM, macc, acc, mf1, f1, avg_loss = evaluate_model_sw(
        model, val_dataloader, device, criterion, CM_Weight)
    print(f"  Initial mIoU: {miou:.4f}")

    if not checkpoint_path.exists():
        log_epoch_record(log_file_path, 0, 0, avg_loss, 0, optimizer, miou, ious, macc, mf1)

    ema_cm = CM.copy()
    alpha_ema = 0.5

    for epoch in range(start_epoch, num_epochs):
        t_start_epoch = time.time()
        ema_cm = alpha_ema * ema_cm + (1 - alpha_ema) * CM
        CM_Weight = Calc_Weights(ema_cm, confusion_type)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}: LR = {current_lr:.8f}")

        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        avg_data_time = 0.0

        progress_bar_train = tqdm(train_dataloader,
                                  desc=f"Epoch {epoch + 1}/{num_epochs} (Train)", ncols=150)
        t_batch_start = time.time()

        for i, batch in enumerate(progress_bar_train):
            data_load_time = time.time() - t_batch_start
            avg_data_time = data_load_time if num_train_batches == 0 \
                else avg_data_time * 0.9 + data_load_time * 0.1

            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear")

            # -------------------------------------------------------------------
            # Loss routing (same logic as evaluate_model_sw)
            if isinstance(criterion, (CALoss, TverskyCALoss)):
                loss = criterion(logits.softmax(dim=1), labels, CM_Weight)
            elif isinstance(criterion, (TverskyLoss, DiceLoss)):
                loss = criterion(logits.softmax(dim=1), labels)
            else:  # FocalLoss, FocalDiceLoss, OHEMCrossEntropy, ClassBalancedCrossEntropy — raw logits
                loss = criterion(logits, labels)
            # -------------------------------------------------------------------

            total_train_loss += loss.item()
            num_train_batches += 1

            progress_bar_train.set_postfix({
                "loss": f"{loss.item():.4f}",
                "data_t": f"{avg_data_time:.4f}s",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}"
            })

            loss.backward()
            optimizer.step()
            scheduler.step()
            t_batch_start = time.time()
            clear_cache_if_full(max_gb=3)

        t_end_epoch = time.time()
        epoch_time_seconds = t_end_epoch - t_start_epoch
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs} finished. Evaluating...")
        current_miou, ious, CM, macc, acc, mf1, f1, valid_loss = evaluate_model_sw(
            model, val_dataloader, device, criterion, CM_Weight)

        print(f"Epoch {epoch + 1} Report:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Eval Loss:  {valid_loss:.4f}")
        print(f"  Current mIoU:   {current_miou:.4f}")

        log_epoch_record(log_file_path, epoch, avg_train_loss, valid_loss,
                         epoch_time_seconds, optimizer, current_miou, ious, macc, mf1)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
            'CM': CM.copy(),
        }, checkpoint_path)

        if current_miou > best_miou:
            best_miou = current_miou
            shutil.copy(checkpoint_path, best_model_path)
            print(f"New best mIoU: {best_miou:.4f}. Saved to {best_model_path}")
        else:
            print("mIoU did not improve.")

        clear_cache_if_full(max_gb=3)
        print(f"Epoch {epoch + 1} done. mIoU: {current_miou:.4f}")


if __name__ == '__main__':
    main()