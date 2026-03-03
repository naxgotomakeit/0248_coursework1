"""
visualise.py
------------
Visualisation utilities for the Hand Gesture project.

Functions:
  plot_training_curves   — loss curves from train_log.csv
  plot_confusion_matrix  — styled confusion matrix from predictions
  overlay_predictions    — RGB + GT/Pred mask + bbox side-by-side grid
  plot_class_distribution— bar chart of class counts in dataset

Usage (standalone):
    python visualise.py --log results/train_log.csv
    python visualise.py --mode overlay --checkpoint weights/best.pth \
                        --data_root ./dataset
"""

import os
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataloader import HandGestureDataset, GESTURE_CLASSES, IDX_TO_CLASS
from model import HandGestureModel
from train import collate_fn, make_splits


# ── Shared style ──────────────────────────────────────────────────────────────
CLASS_NAMES = [g.split('_')[1] for g in GESTURE_CLASSES]   # short labels
CMAP_CONF   = 'Blues'
DPI         = 150

# ImageNet denorm tensors (for RGB display)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """Denormalise a (3,H,W) tensor and return (H,W,3) numpy for imshow."""
    return (t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_csv: str, save_path: str = None):
    """
    Reads the CSV produced by train.py and plots train vs val loss curves
    (total + each sub-task).

    Args:
        log_csv   : path to train_log.csv
        save_path : where to save the PNG (default: same dir as csv)
    """
    df = pd.read_csv(log_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training & Validation Loss Curves', fontsize=14, fontweight='bold')

    pairs = [
        ('train_total', 'val_total', 'Total Loss',          axes[0, 0]),
        ('train_det',   'val_det',   'Detection Loss',      axes[0, 1]),
        ('train_seg',   'val_seg',   'Segmentation Loss',   axes[1, 0]),
        ('train_cls',   'val_cls',   'Classification Loss', axes[1, 1]),
    ]

    for train_col, val_col, title, ax in pairs:
        ax.plot(df['epoch'], df[train_col], label='Train',
                color='steelblue', linewidth=1.5)
        ax.plot(df['epoch'], df[val_col],   label='Val',
                color='tomato',    linewidth=1.5, linestyle='--')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = str(Path(log_csv).parent / 'training_curves.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           title: str = 'Confusion Matrix',
                           save_path: str = './results/confusion_matrix.png'):
    """
    Plots and saves a styled confusion matrix.

    Args:
        y_true    : (N,) ground-truth class indices
        y_pred    : (N,) predicted class indices
        title     : plot title
        save_path : output PNG path
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GESTURE_CLASSES))))

    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap=CMAP_CONF, colorbar=True,
              xticks_rotation=45, values_format='d')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Ground Truth', fontsize=11)

    # Highlight diagonal
    for i in range(len(CLASS_NAMES)):
        ax.add_patch(patches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            linewidth=2, edgecolor='gold', facecolor='none'
        ))

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Overlay predictions (RGB | GT mask | Pred mask + bbox)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def overlay_predictions(model, dataset, device, use_depth,
                         save_path: str = './results/overlay.png',
                         n_per_class: int = 1):
    """
    For each gesture class, picks up to n_per_class annotated samples and
    plots a 3-column grid:
        Col 0: RGB image + predicted bounding box (green)
                          + ground-truth bbox (red dashed)
        Col 1: Ground-truth mask (white = hand)
        Col 2: Predicted mask   (white = hand)

    Correct classifications are marked ✓ (green), wrong ones ✗ (red).

    Args:
        model       : trained HandGestureModel (already on device)
        dataset     : HandGestureDataset or Subset
        device      : torch device
        use_depth   : whether to pass depth to model
        save_path   : output PNG path
        n_per_class : how many examples per gesture class
    """
    model.eval()

    # ── Collect indices ───────────────────────────────────────────────────────
    # ── Collect indices ───────────────────────────────────────────────────────
    class_buckets = {i: [] for i in range(len(GESTURE_CLASSES))}

    n_total = len(dataset)
    # 🎲 随机起点魔法加在这里！
    start_idx = random.randint(0, n_total - 1)
    # 拼出一个从 start_idx 开始，最后绕回开头的完美顺序列表
    indices = list(range(start_idx, n_total)) + list(range(0, start_idx))

    # 用咱们的新 indices 替换掉原来死板的 range(len(dataset))
    for idx in indices:
        if hasattr(dataset, 'dataset'):
            raw_idx = dataset.indices[idx]
            s = dataset.dataset.samples[raw_idx]
        else:
            s = dataset.samples[idx]
            
        if s['has_mask'] and len(class_buckets[s['label']]) < n_per_class:
            class_buckets[s['label']].append(idx)

    rows = [idx for bucket in class_buckets.values() for idx in bucket]
    n    = len(rows)
    if n == 0:
        print("  No annotated samples found for overlay.")
        return

    # ── Plot: 4 columns now ───────────────────────────────────────────────────
    #   Col 0: RGB + GT bbox (red dashed) + Pred bbox (green solid)
    #   Col 1: Classification bar chart (top-5 softmax scores)
    #   Col 2: GT mask
    #   Col 3: Pred mask
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.2 * n),
                              gridspec_kw={'wspace': 0.12, 'hspace': 0.4,
                                           'width_ratios': [2, 1.6, 1, 1]})
    if n == 1:
        axes = axes[None, :]

    col_titles = ['RGB + BBox', 'Classification (top-5)', 'GT Mask', 'Pred Mask']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=10, fontweight='bold', pad=6)

    for row, idx in enumerate(rows):
        sample = dataset[idx]

        rgb_t   = sample['rgb'].unsqueeze(0).to(device)
        depth_t = sample['depth'].unsqueeze(0).to(device) if use_depth else \
                  torch.zeros(1, 1, *sample['rgb'].shape[1:]).to(device)
        gt_mask  = sample['mask'].squeeze().numpy()
        gt_bbox  = sample['bbox'].numpy()
        
        
        gt_label = sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label']
        print(f"idx={idx}, s_label={s['label']}, gt_label={gt_label}, gesture={s['gesture']}")

        preds      = model(rgb_t, depth_t)
        pred_mask  = (torch.sigmoid(preds['mask']) > 0.5).squeeze().cpu().numpy()
        pred_bbox  = preds['bbox'].squeeze().cpu().numpy()

        # Softmax scores for all 10 classes
        scores     = torch.softmax(preds['logits'], dim=1).squeeze().cpu().numpy()
        pred_label = int(scores.argmax())
        confidence = float(scores[pred_label])

        rgb_show   = _denorm(sample['rgb'])
        # 在这行下面： rgb_show   = _denorm(sample['rgb'])

        H_img, W_img = rgb_show.shape[:2]
        
        # 反归一化 Predicted bbox
        px1, py1, px2, py2 = pred_bbox
        px1, px2 = px1 * W_img, px2 * W_img
        py1, py2 = py1 * H_img, py2 * H_img
        
        # 反归一化 GT bbox
        gx1, gy1, gx2, gy2 = gt_bbox
        gx1, gx2 = gx1 * W_img, gx2 * W_img
        gy1, gy2 = gy1 * H_img, gy2 * H_img
        correct    = pred_label == gt_label
        tick       = '✓' if correct else '✗'
        tick_color = 'limegreen' if correct else 'tomato'
        gt_name    = CLASS_NAMES[gt_label]
        pred_name  = CLASS_NAMES[pred_label]

        # ── Col 0: RGB + bboxes ──────────────────────────────────────────────
        ax = axes[row, 0]
        ax.imshow(rgb_show)

        # Predicted bbox (solid green)
        px1, py1, px2, py2 = pred_bbox
        ax.add_patch(patches.Rectangle(
            (px1, py1), px2 - px1, py2 - py1,
            linewidth=2.5, edgecolor='limegreen', facecolor='none'
        ))
        # GT bbox (dashed red)
        gx1, gy1, gx2, gy2 = gt_bbox
        ax.add_patch(patches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2.5, edgecolor='tomato', facecolor='none', linestyle='--'
        ))
        ax.set_ylabel(
            f'GT: {gt_name}', fontsize=9,
            rotation=0, labelpad=55, va='center', color='dimgray'
        )
        ax.axis('off')

        # ── Col 1: Classification bar chart ──────────────────────────────────
        ax2 = axes[row, 1]

        # Show top-5 classes sorted by score
        top5_idx    = scores.argsort()[::-1][:5]
        top5_names  = [CLASS_NAMES[i] for i in top5_idx]
        top5_scores = scores[top5_idx]

        bar_colors = []
        for i in top5_idx:
            if i == gt_label and i == pred_label:
                bar_colors.append('limegreen')   # correct prediction
            elif i == gt_label:
                bar_colors.append('tomato')      # GT class but not top-1
            elif i == pred_label:
                bar_colors.append('orange')      # wrong top-1
            else:
                bar_colors.append('steelblue')   # other classes

        bars = ax2.barh(range(5), top5_scores, color=bar_colors,
                        edgecolor='white', linewidth=0.5)

        # Score labels on bars
        for bar, score in zip(bars, top5_scores):
            ax2.text(min(score + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                     f'{score:.2f}', va='center', fontsize=8)

        ax2.set_yticks(range(5))
        ax2.set_yticklabels(top5_names, fontsize=8)
        ax2.set_xlim(0, 1.15)
        ax2.set_xlabel('Confidence', fontsize=8)
        ax2.invert_yaxis()   # highest score on top
        ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.8)
        ax2.tick_params(axis='x', labelsize=7)

        # Title: show prediction result
        ax2.set_title(
            f'Pred: {pred_name} ({confidence:.0%}) {tick}',
            fontsize=9, color=tick_color, pad=4
        )
        ax2.spines[['top', 'right']].set_visible(False)

        # ── Col 2: GT mask ───────────────────────────────────────────────────
        axes[row, 2].imshow(gt_mask,   cmap='gray', vmin=0, vmax=1)
        axes[row, 2].axis('off')

        # ── Col 3: Pred mask ─────────────────────────────────────────────────
        axes[row, 3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].axis('off')

    # ── Colour legend (first row only) ────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    bbox_legend = [
        Line2D([0], [0], color='limegreen', linewidth=2,  label='Pred bbox'),
        Line2D([0], [0], color='tomato',    linewidth=2,
               linestyle='--', label='GT bbox'),
    ]
    axes[0, 0].legend(handles=bbox_legend, fontsize=7,
                      loc='lower right', framealpha=0.7)

    cls_legend = [
        Patch(color='limegreen', label='Correct (GT=Pred)'),
        Patch(color='tomato',    label='GT class (missed)'),
        Patch(color='orange',    label='Wrong top-1'),
        Patch(color='steelblue', label='Other'),
    ]
    axes[0, 1].legend(handles=cls_legend, fontsize=6.5,
                      loc='lower right', framealpha=0.7)

    fig.suptitle('Qualitative Predictions  —  Detection · Classification · Segmentation',
                 fontsize=13, fontweight='bold', y=1.005)

    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Overlay saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Class distribution bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(data_root: str,
                             save_path: str = './results/class_distribution.png'):
    """
    Counts annotated keyframes per gesture class across the whole dataset
    and saves a bar chart. Useful for the Dataset section of the report.
    """
    ds = HandGestureDataset(data_root, keyframes_only=True,
                             augment=False, use_depth=False)
    counts = np.zeros(len(GESTURE_CLASSES), dtype=int)
    for s in ds.samples:
        counts[s['label']] += 1

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(CLASS_NAMES, counts, color='steelblue', edgecolor='white',
                  linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel('Gesture Class', fontsize=11)
    ax.set_ylabel('Number of Annotated Frames', fontsize=11)
    ax.set_title('Dataset Class Distribution', fontsize=13, fontweight='bold')
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Class distribution saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Visualisation tools')
    p.add_argument('--mode', choices=['curves', 'overlay', 'dist', 'all'],
                   default='all')
    p.add_argument('--log',        type=str, default=None,
                   help='Path to train_log.csv (for curves mode)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to model checkpoint (for overlay mode)')
    p.add_argument('--data_root',  type=str, default='./dataset')
    p.add_argument('--results_dir',type=str, default='./results')
    # p.add_argument('--img_size',   type=int, default=320)
    p.add_argument('--img_h', type=int, default=480)
    p.add_argument('--img_w', type=int, default=640)
    p.add_argument('--val_ratio',  type=float, default=0.2)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--no_depth',   action='store_true')
    p.add_argument('--n_per_class',type=int, default=1)
    return p.parse_args()


def main():
    args      = get_args()
    results   = args.results_dir
    use_depth = not args.no_depth
    img_size = (args.img_h, args.img_w)
    os.makedirs(results, exist_ok=True)

    # ── Training curves ───────────────────────────────────────────────────────
    if args.mode in ('curves', 'all'):
        log = args.log
        if log is None:
            # Try to find automatically
            candidates = list(Path(results).glob('*_log.csv'))
            if candidates:
                log = str(candidates[0])
                print(f"Auto-detected log: {log}")
        if log and Path(log).exists():
            plot_training_curves(log, save_path=f'{results}/training_curves.png')
        else:
            print("  Skipping curves — no log CSV found (use --log path/to/log.csv)")

    # ── Class distribution ────────────────────────────────────────────────────
    if args.mode in ('dist', 'all'):
        plot_class_distribution(
            args.data_root,
            save_path=f'{results}/class_distribution.png'
        )

    # ── Overlay predictions ───────────────────────────────────────────────────
    if args.mode in ('overlay', 'all'):
        if args.checkpoint is None:
            print("  Skipping overlay — no checkpoint provided (use --checkpoint)")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt   = torch.load(args.checkpoint, map_location=device)
            model  = HandGestureModel(img_size=img_size,
                                      use_depth=use_depth).to(device)
            model.load_state_dict(ckpt['model_state'])

            _, val_set = make_splits(
                args.data_root, args.val_ratio, args.seed, img_size, use_depth
            )
            overlay_predictions(
                model, val_set, device, use_depth,
                save_path=f'{results}/overlay_val.png',
                n_per_class=args.n_per_class,
            )


if __name__ == '__main__':
    main()