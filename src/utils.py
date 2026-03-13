"""
utils.py — Helper functions for metrics, bounding box operations, and visualisation.
"""

import numpy as np
import torch


# ─────────────────────────────────────────────
# Bounding Box Utilities
# ─────────────────────────────────────────────

def compute_iou(box_pred: torch.Tensor, box_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between predicted and ground-truth bounding boxes.

    Args:
        box_pred: (N, 4) normalised [x1, y1, x2, y2]
        box_gt:   (N, 4) normalised [x1, y1, x2, y2]

    Returns:
        iou: (N,) IoU per sample
    """
    x1 = torch.max(box_pred[:, 0], box_gt[:, 0])
    y1 = torch.max(box_pred[:, 1], box_gt[:, 1])
    x2 = torch.min(box_pred[:, 2], box_gt[:, 2])
    y2 = torch.min(box_pred[:, 3], box_gt[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_pred = (box_pred[:, 2] - box_pred[:, 0]) * (box_pred[:, 3] - box_pred[:, 1])
    area_gt   = (box_gt[:, 2]   - box_gt[:, 0])   * (box_gt[:, 3]   - box_gt[:, 1])
    union = area_pred + area_gt - inter

    return inter / union.clamp(min=1e-6)


def denormalise_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Convert normalised [0,1] box coordinates to pixel coordinates.

    Args:
        boxes:  (N, 4) normalised [x1, y1, x2, y2]
        img_w:  image width in pixels
        img_h:  image height in pixels

    Returns:
        boxes in pixel coordinates (N, 4)
    """
    scale = torch.tensor([img_w, img_h, img_w, img_h],
                         dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


# ─────────────────────────────────────────────
# Segmentation Metrics
# ─────────────────────────────────────────────

def compute_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                 threshold: float = 0.5) -> torch.Tensor:
    """
    Compute Dice score between predicted and ground-truth binary masks.

    Args:
        pred_mask: (N, 1, H, W) raw logits or probabilities
        gt_mask:   (N, 1, H, W) binary ground-truth
        threshold: binarisation threshold for predictions

    Returns:
        dice: scalar mean Dice score
    """
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * gt_mask).sum(dim=(1, 2, 3))
    dice = (2 * intersection) / (pred_bin.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3)) + 1e-6)
    return dice.mean()


def compute_seg_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                    threshold: float = 0.5) -> torch.Tensor:
    """
    Compute mean segmentation IoU over a batch.

    Args:
        pred_mask: (N, 1, H, W) raw logits or probabilities
        gt_mask:   (N, 1, H, W) binary ground-truth
        threshold: binarisation threshold

    Returns:
        iou: scalar mean IoU
    """
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * gt_mask).sum(dim=(1, 2, 3))
    union = (pred_bin + gt_mask).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = intersection / (union + 1e-6)
    return iou.mean()


# ─────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Top-1 classification accuracy.

    Args:
        logits: (N, C) raw class logits
        labels: (N,)  ground-truth class indices

    Returns:
        accuracy as a float in [0, 1]
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_macro_f1(preds: np.ndarray, labels: np.ndarray,
                     num_classes: int = 10) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        preds:       (N,) predicted class indices
        labels:      (N,) ground-truth class indices
        num_classes: number of gesture classes

    Returns:
        macro F1 score as float
    """
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


# ─────────────────────────────────────────────
# Depth Preprocessing
# ─────────────────────────────────────────────

def normalise_depth(depth: np.ndarray,
                    min_mm: float = 0.0,
                    max_mm: float = 5000.0) -> np.ndarray:
    """
    Clip and normalise raw 16-bit depth values to [0, 1].

    Args:
        depth:  (H, W) raw depth array in millimetres
        min_mm: minimum valid depth in mm
        max_mm: maximum valid depth in mm

    Returns:
        normalised depth array (H, W) in [0, 1]
    """
    depth = np.clip(depth, min_mm, max_mm)
    return (depth - min_mm) / (max_mm - min_mm + 1e-6)
