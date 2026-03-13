"""
evaluate.py
-----------
Evaluation script for the multi-task Hand Gesture Model.

Computes all metrics required by the coursework:

  Detection:
    - Detection accuracy @ 0.5 IoU
    - Mean bounding-box IoU

  Segmentation:
    - Mean IoU (hand vs background)
    - Dice coefficient

  Classification:
    - Top-1 accuracy
    - Macro-averaged F1 score
    - Confusion matrix (saved as PNG)

Usage:
    # Evaluate on validation set
    python evaluate.py --data_root ./dataset --checkpoint ./weights/best.pth

    # Evaluate on test set (after 27 Feb release)
    python evaluate.py --data_root ./test_dataset --checkpoint ./weights/best.pth --test
"""

import os
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)

from dataloader import HandGestureDataset, IDX_TO_CLASS, GESTURE_CLASSES
from model import HandGestureModel
from train import collate_fn, make_splits


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Evaluate Hand Gesture Model')
    p.add_argument('--data_root',   type=str,   required=True)
    p.add_argument('--checkpoint',  type=str,   required=True)
    p.add_argument('--results_dir', type=str,   default='./results')
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--img_h',       type=int,   default=480)
    p.add_argument('--img_w',       type=int,   default=640)
    p.add_argument('--val_ratio',   type=float, default=0.2)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--no_depth',    action='store_true')
    p.add_argument('--test',        action='store_true',
                   help='Evaluate on full dataset (test mode, no split)')
    p.add_argument('--num_workers', type=int,   default=4)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between predicted and ground-truth bounding boxes.
    Both tensors: (N, 4) format [x1, y1, x2, y2].
    Returns: (N,) IoU values.
    """
    x1 = torch.max(pred[:, 0], gt[:, 0])
    y1 = torch.max(pred[:, 1], gt[:, 1])
    x2 = torch.min(pred[:, 2], gt[:, 2])
    y2 = torch.min(pred[:, 3], gt[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter   = inter_w * inter_h

    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * \
                (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_gt   = (gt[:, 2]   - gt[:, 0]).clamp(min=0)  * \
                (gt[:, 3]   - gt[:, 1]).clamp(min=0)

    union = area_pred + area_gt - inter
    return inter / (union + 1e-6)


def mask_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Compute mean IoU and Dice coefficient for binary segmentation masks.
    pred_mask, gt_mask : (N, H, W) bool/binary numpy arrays.
    Returns: mean_iou (float), dice (float)
    """
    eps = 1e-6
    pred_flat = pred_mask.reshape(len(pred_mask), -1).astype(bool)
    gt_flat   = gt_mask.reshape(len(gt_mask),     -1).astype(bool)

    inter = (pred_flat & gt_flat).sum(axis=1).astype(float)
    union = (pred_flat | gt_flat).sum(axis=1).astype(float)
    sum_  = (pred_flat.sum(axis=1) + gt_flat.sum(axis=1)).astype(float)

    iou  = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (sum_  + eps)

    return iou.mean(), dice.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, use_depth, results_dir, split_name='val'):
    model.eval()
    os.makedirs(results_dir, exist_ok=True)

    # Accumulators
    all_bbox_iou   = []   # per-sample bbox IoU (annotated frames only)
    all_pred_masks = []   # predicted binary masks (annotated frames only)
    all_gt_masks   = []   # ground-truth masks
    all_pred_cls   = []   # predicted class indices (all frames)
    all_gt_cls     = []   # ground-truth class indices

    for batch in loader:
        rgb   = batch['rgb'].to(device)
        depth = batch['depth'].to(device) if use_depth else \
                torch.zeros_like(batch['depth']).to(device)

        preds = model(rgb, depth)

        # ── Classification (all frames) ──────────────────────────────────
        pred_cls = preds['logits'].argmax(dim=1).cpu()
        gt_cls   = batch['label']
        all_pred_cls.extend(pred_cls.tolist())
        all_gt_cls.extend(gt_cls.tolist())

        # ── Detection & Segmentation (annotated frames only) ─────────────
        has_mask = batch['has_mask']
        if has_mask.any():
            # Detection
            pred_bbox = preds['bbox'][has_mask].cpu()
            gt_bbox   = batch['bbox'][has_mask]
            ious = bbox_iou(pred_bbox, gt_bbox)
            all_bbox_iou.extend(ious.tolist())

            # Segmentation — threshold sigmoid output at 0.5
            pred_mask = (torch.sigmoid(preds['mask'][has_mask]) > 0.5) \
                        .squeeze(1).cpu().numpy()   # (N, H, W) bool
            gt_mask   = (batch['mask'][has_mask] > 0.5) \
                        .squeeze(1).numpy()         # (N, H, W) bool
            all_pred_masks.append(pred_mask)
            all_gt_masks.append(gt_mask)

    # ── Aggregate ────────────────────────────────────────────────────────────
    all_pred_cls = np.array(all_pred_cls)
    all_gt_cls   = np.array(all_gt_cls)

    # Detection metrics
    mean_bbox_iou = float(np.mean(all_bbox_iou)) if all_bbox_iou else 0.0
    det_acc_at_05 = float(np.mean(np.array(all_bbox_iou) >= 0.5)) \
                    if all_bbox_iou else 0.0

    # Segmentation metrics
    if all_pred_masks:
        pred_masks_cat = np.concatenate(all_pred_masks, axis=0)
        gt_masks_cat   = np.concatenate(all_gt_masks,   axis=0)
        mean_seg_iou, mean_dice = mask_iou_dice(pred_masks_cat, gt_masks_cat)
    else:
        mean_seg_iou = mean_dice = 0.0

    # Classification metrics
    top1_acc = float((all_pred_cls == all_gt_cls).mean())
    macro_f1 = float(f1_score(all_gt_cls, all_pred_cls,
                               average='macro', zero_division=0))
    cm = confusion_matrix(all_gt_cls, all_pred_cls,
                          labels=list(range(len(GESTURE_CLASSES))))

    # ── Print summary ─────────────────────────────────────────────────────────
    class_names = [g.split('_')[1] for g in GESTURE_CLASSES]   # short names
    print(f"\n{'='*55}")
    print(f"  Evaluation results — {split_name.upper()}")
    print(f"{'='*55}")
    print(f"  [Detection]")
    print(f"    Accuracy @ 0.5 IoU : {det_acc_at_05*100:.2f}%")
    print(f"    Mean bbox IoU      : {mean_bbox_iou:.4f}")
    print(f"  [Segmentation]")
    print(f"    Mean IoU           : {float(mean_seg_iou):.4f}")
    print(f"    Dice coefficient   : {float(mean_dice):.4f}")
    print(f"  [Classification]")
    print(f"    Top-1 accuracy     : {top1_acc*100:.2f}%")
    print(f"    Macro F1           : {macro_f1:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(
        all_gt_cls, all_pred_cls,
        target_names=class_names, zero_division=0
    ))

    # ── Save confusion matrix ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
    ax.set_title(f'Confusion Matrix — {split_name}', fontsize=14)
    plt.tight_layout()
    cm_path = Path(results_dir) / f'confusion_matrix_{split_name}.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    # ── Save metrics as JSON ──────────────────────────────────────────────────
    metrics = {
        'split'        : split_name,
        'det_acc_at_05': round(det_acc_at_05,        4),
        'mean_bbox_iou': round(mean_bbox_iou,         4),
        'mean_seg_iou' : round(float(mean_seg_iou),   4),
        'dice'         : round(float(mean_dice),       4),
        'top1_acc'     : round(top1_acc,              4),
        'macro_f1'     : round(macro_f1,              4),
        'n_samples'    : len(all_pred_cls),
        'n_annotated'  : len(all_bbox_iou),
    }
    json_path = Path(results_dir) / f'metrics_{split_name}.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved       → {json_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative visualisation  (mask + bbox overlays on a few images)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualise_predictions(model, dataset, device, use_depth,
                           results_dir, split_name='val', n_samples=12):
    """
    Saves a grid of images showing RGB | GT mask | Predicted mask + bbox.
    Picks one example per gesture class where possible.
    """
    model.eval()
    os.makedirs(results_dir, exist_ok=True)

    # Randomly select a start index, then scan forward wrapping around.
    # This balances random sampling with sequential disk access.
    n_total    = len(dataset)
    start_idx  = random.randint(0, n_total - 1)
    indices    = list(range(start_idx, n_total)) + list(range(0, start_idx))

    class_seen     = set()
    samples_to_viz = []

    for idx in indices:
        sample = dataset[idx]

        if not sample['has_mask']:
            continue

        label_val = sample['label'].item() \
                    if isinstance(sample['label'], torch.Tensor) \
                    else sample['label']

        if label_val not in class_seen:
            class_seen.add(label_val)
            samples_to_viz.append(sample)

        if len(samples_to_viz) >= n_samples:
            break

    n = len(samples_to_viz)
    if n == 0:
        print("  No annotated samples found for visualisation.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]   # ensure 2D indexing

    # ImageNet denormalisation for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, sample in enumerate(samples_to_viz):
        rgb_t   = sample['rgb'].unsqueeze(0).to(device)
        depth_t = sample['depth'].unsqueeze(0).to(device) if use_depth else \
                  torch.zeros(1, 1, *sample['rgb'].shape[1:]).to(device)

        gt_mask = sample['mask'].squeeze()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.numpy()

        gt_bbox = sample['bbox']
        if isinstance(gt_bbox, torch.Tensor):
            gt_bbox = gt_bbox.numpy()

        preds      = model(rgb_t, depth_t)
        pred_mask  = (torch.sigmoid(preds['mask']) > 0.5).squeeze().cpu().numpy()
        pred_bbox  = preds['bbox'].squeeze().cpu().numpy()
        pred_label = preds['logits'].argmax(dim=1).item()

        # Denormalise RGB for display
        rgb_show = (sample['rgb'] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        label_val  = sample['label'].item() \
                     if isinstance(sample['label'], torch.Tensor) \
                     else sample['label']
        gesture_short = IDX_TO_CLASS[label_val].split('_')[1]
        pred_short    = IDX_TO_CLASS[pred_label].split('_')[1]
        correct       = '✓' if label_val == pred_label else '✗'

        print(f"  Visualising sample {row+1}/{n} — "
              f"GT: {IDX_TO_CLASS[label_val]} | Pred: {IDX_TO_CLASS[pred_label]}")

        # Denormalise bbox coordinates [0,1] → pixel coordinates
        H_img, W_img = rgb_show.shape[:2]
        px1, py1, px2, py2 = pred_bbox
        px1, px2 = px1 * W_img, px2 * W_img
        py1, py2 = py1 * H_img, py2 * H_img
        gx1, gy1, gx2, gy2 = gt_bbox
        gx1, gx2 = gx1 * W_img, gx2 * W_img
        gy1, gy2 = gy1 * H_img, gy2 * H_img

        # Column 0: RGB + predicted bbox (green) + GT bbox (dashed red)
        axes[row, 0].imshow(rgb_show)
        axes[row, 0].add_patch(patches.Rectangle(
            (px1, py1), px2 - px1, py2 - py1,
            linewidth=2, edgecolor='lime', facecolor='none'
        ))
        axes[row, 0].add_patch(patches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2, edgecolor='tomato', facecolor='none', linestyle='--'
        ))
        axes[row, 0].set_title(
            f'GT: {gesture_short}  Pred: {pred_short} {correct}', fontsize=9
        )
        axes[row, 0].axis('off')

        # Column 1: Ground-truth mask
        axes[row, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('GT mask', fontsize=9)
        axes[row, 1].axis('off')

        # Column 2: Predicted mask
        axes[row, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title('Pred mask', fontsize=9)
        axes[row, 2].axis('off')

    plt.suptitle(f'Qualitative results — {split_name}', fontsize=13, y=1.01)
    plt.tight_layout()
    viz_path = Path(results_dir) / f'qualitative_{split_name}.png'
    plt.savefig(viz_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Qualitative viz saved → {viz_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args      = get_args()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_depth = not args.no_depth
    img_size  = (args.img_h, args.img_w)

    print(f"Device    : {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use depth : {use_depth}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = HandGestureModel(img_size=img_size, use_depth=use_depth).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?'):.4f})")

    # ── Build dataset ─────────────────────────────────────────────────────────
    if args.test:
        # Test mode: evaluate on the entire dataset folder as-is
        dataset = HandGestureDataset(
            args.data_root, use_depth=use_depth,
            keyframes_only=True, augment=False, img_size=img_size
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )
        split_name  = 'test'
        viz_dataset = dataset

    else:
        # Val mode: use the same student-level split as training
        _, val_set = make_splits(
            args.data_root, args.val_ratio, args.seed, img_size, use_depth
        )
        loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )
        split_name  = 'val'
        viz_dataset = val_set

    # ── Quantitative evaluation ───────────────────────────────────────────────
    metrics = evaluate(
        model, loader, device, use_depth, args.results_dir, split_name
    )

    # ── Qualitative visualisation ─────────────────────────────────────────────
    visualise_predictions(
        model, viz_dataset, device, use_depth,
        args.results_dir, split_name, n_samples=10
    )

    print(f"\nAll results saved to: {args.results_dir}/")


if __name__ == '__main__':
    main()
