"""
train.py
--------
Training loop for the multi-task RGB-D Hand Gesture Model.

Usage:
    python train.py --data_root ./dataset --epochs 80 --batch_size 8

Key features:
  - Trains detection, segmentation, and classification jointly
  - Saves best checkpoint based on validation loss
  - Logs per-epoch losses to results/train_log.csv
  - Supports RGB-only mode (--no_depth) for ablation study
  - Early stopping: halts training if val_loss does not improve by
    --es_min_delta over --es_patience consecutive epochs
"""

import os
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random

from dataloader import HandGestureDataset, CLASS_TO_IDX
from model import HandGestureModel, MultiTaskLoss


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Train Hand Gesture Model')
    p.add_argument('--data_root',    type=str,   default='./dataset')
    p.add_argument('--save_dir',     type=str,   default='./weights')
    p.add_argument('--results_dir',  type=str,   default='./results')
    p.add_argument('--epochs',       type=int,   default=80)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--val_ratio',    type=float, default=0.2)
    p.add_argument('--img_h',        type=int,   default=480)
    p.add_argument('--img_w',        type=int,   default=640)
    p.add_argument('--w_det',        type=float, default=1.0,
                   help='Detection loss weight')
    p.add_argument('--w_seg',        type=float, default=1.0,
                   help='Segmentation loss weight')
    p.add_argument('--w_cls',        type=float, default=1.0,
                   help='Classification loss weight')
    p.add_argument('--no_depth',     action='store_true',
                   help='RGB-only mode (ablation)')
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--es_patience',  type=int,   default=10,
                   help='Early stopping patience: number of epochs without improvement before halting')
    p.add_argument('--es_min_delta', type=float, default=1e-4,
                   help='Minimum improvement in val_loss to be considered a meaningful update')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopper
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopper:
    """
    Monitors val_loss and triggers early stopping when no improvement
    greater than min_delta is observed for `patience` consecutive epochs.

    Usage:
        stopper = EarlyStopper(patience=10, min_delta=1e-4)
        if stopper.step(val_loss):
            break
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0   # consecutive epochs without meaningful improvement

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            # Meaningful improvement — reset counter
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            # No meaningful improvement — increment counter
            self.counter += 1
            return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val split  (student-level split to avoid data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(root_dir: str, val_ratio: float, seed: int,
                img_size: tuple, use_depth: bool):
    """
    Splits by student folder so that no student's data appears in both
    train and val — prevents the model from memorising a specific hand's
    appearance.
    """
    root = Path(root_dir)
    student_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    random.seed(seed)
    random.shuffle(student_dirs)
    n_val = max(1, int(len(student_dirs) * val_ratio))
    val_students   = set(str(d) for d in student_dirs[:n_val])
    train_students = set(str(d) for d in student_dirs[n_val:])

    print(f"Train students: {len(train_students)}  |  Val students: {len(val_students)}")

    # Build full dataset twice: once with augmentation, once without
    full_aug = HandGestureDataset(root_dir, use_depth=use_depth,
                                  keyframes_only=False, augment=True,
                                  img_size=img_size)
    full_no_aug = HandGestureDataset(root_dir, use_depth=use_depth,
                                     keyframes_only=False, augment=False,
                                     img_size=img_size)

    def student_of(sample):
        # Resolve student folder robustly by taking the first path component
        # relative to root, handling arbitrary nesting inside student dirs.
        p = Path(sample['rgb_path'])
        student_folder_name = p.relative_to(root).parts[0]
        return str(root / student_folder_name)

    train_idx = [i for i, s in enumerate(full_aug.samples)
                 if student_of(s) in train_students]
    val_idx   = [i for i, s in enumerate(full_no_aug.samples)
                 if student_of(s) in val_students]

    train_set = Subset(full_aug,    train_idx)
    val_set   = Subset(full_no_aug, val_idx)

    print(f"Train frames : {len(train_set)}  |  Val frames : {len(val_set)}")
    return train_set, val_set


# ─────────────────────────────────────────────────────────────────────────────
# Collate function  (handles has_mask as bool tensor)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    rgb       = torch.stack([b['rgb']       for b in batch])
    depth     = torch.stack([b['depth']     for b in batch])
    depth_raw = torch.stack([b['depth_raw'] for b in batch])
    mask      = torch.stack([b['mask']      for b in batch])
    bbox      = torch.stack([b['bbox']      for b in batch])
    label     = torch.stack([b['label']     for b in batch])
    has_mask  = torch.tensor([b['has_mask'] for b in batch], dtype=torch.bool)
    return {
        'rgb'      : rgb,
        'depth'    : depth,
        'depth_raw': depth_raw,
        'mask'     : mask,
        'bbox'     : bbox,
        'label'    : label,
        'has_mask' : has_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, use_depth):
    model.train()
    totals = {'loss_total': 0, 'loss_det': 0, 'loss_seg': 0, 'loss_cls': 0}
    n_batches = 0

    for batch in loader:
        rgb   = batch['rgb'].to(device)
        depth = batch['depth'].to(device) if use_depth else \
                torch.zeros_like(batch['depth']).to(device)

        targets = {
            'bbox'    : batch['bbox'].to(device),
            'mask'    : batch['mask'].to(device),
            'label'   : batch['label'].to(device),
            'has_mask': batch['has_mask'].to(device),
        }

        optimizer.zero_grad()
        preds = model(rgb, depth)
        loss, loss_dict = criterion(preds, targets)
        loss.backward()

        # Gradient clipping to stabilise training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        for k in totals:
            totals[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, use_depth):
    model.eval()
    totals = {'loss_total': 0, 'loss_det': 0, 'loss_seg': 0, 'loss_cls': 0}
    n_batches = 0

    for batch in loader:
        rgb   = batch['rgb'].to(device)
        depth = batch['depth'].to(device) if use_depth else \
                torch.zeros_like(batch['depth']).to(device)

        targets = {
            'bbox'    : batch['bbox'].to(device),
            'mask'    : batch['mask'].to(device),
            'label'   : batch['label'].to(device),
            'has_mask': batch['has_mask'].to(device),
        }

        preds = model(rgb, depth)
        _, loss_dict = criterion(preds, targets)

        for k in totals:
            totals[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_depth = not args.no_depth
    img_size  = (args.img_h, args.img_w)

    print(f"\n{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Use depth   : {use_depth}")
    print(f"  Image size  : {img_size}")
    print(f"  Epochs      : {args.epochs}  |  Batch : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Loss weights — det:{args.w_det} seg:{args.w_seg} cls:{args.w_cls}")
    print(f"  Early stop  : patience={args.es_patience}, min_delta={args.es_min_delta}")
    print(f"{'='*55}\n")

    # ── Directories ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    run_name  = f"rgbd_{'depth' if use_depth else 'rgb'}_bs{args.batch_size}_lr{args.lr}"
    ckpt_best = Path(args.save_dir) / f"{run_name}_best.pth"
    ckpt_last = Path(args.save_dir) / f"{run_name}_last.pth"
    log_path  = Path(args.results_dir) / f"{run_name}_log.csv"

    # ── Data ─────────────────────────────────────────────────────────────────
    train_set, val_set = make_splits(
        args.data_root, args.val_ratio, args.seed, img_size, use_depth
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Model, loss, optimiser, scheduler ────────────────────────────────────
    model     = HandGestureModel(img_size=img_size, use_depth=use_depth).to(device)
    criterion = MultiTaskLoss(w_det=args.w_det, w_seg=args.w_seg, w_cls=args.w_cls)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing — smoothly reduces LR to near-zero over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    early_stopper = EarlyStopper(
        patience=args.es_patience,
        min_delta=args.es_min_delta,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # ── CSV log header ────────────────────────────────────────────────────────
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'lr',
            'train_total', 'train_det', 'train_seg', 'train_cls',
            'val_total',   'val_det',   'val_seg',   'val_cls',
        ])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_depth
        )
        val_losses = validate(
            model, val_loader, criterion, device, use_depth
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0

        # Early stopper counter shown for monitoring purposes
        es_info = f"  [ES {early_stopper.counter}/{early_stopper.patience}]"
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"lr={current_lr:.2e}  "
            f"train_loss={train_losses['loss_total']:.4f} "
            f"(det={train_losses['loss_det']:.3f} "
            f"seg={train_losses['loss_seg']:.3f} "
            f"cls={train_losses['loss_cls']:.3f})  "
            f"val_loss={val_losses['loss_total']:.4f} "
            f"(det={val_losses['loss_det']:.3f} "
            f"seg={val_losses['loss_seg']:.3f} "
            f"cls={val_losses['loss_cls']:.3f})  "
            f"[{elapsed:.1f}s]{es_info}"
        )

        # ── CSV log ───────────────────────────────────────────────────────────
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{current_lr:.2e}",
                f"{train_losses['loss_total']:.4f}",
                f"{train_losses['loss_det']:.4f}",
                f"{train_losses['loss_seg']:.4f}",
                f"{train_losses['loss_cls']:.4f}",
                f"{val_losses['loss_total']:.4f}",
                f"{val_losses['loss_det']:.4f}",
                f"{val_losses['loss_seg']:.4f}",
                f"{val_losses['loss_cls']:.4f}",
            ])

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_losses['loss_total'] < best_val_loss:
            best_val_loss = val_losses['loss_total']
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'   : best_val_loss,
                'args'       : vars(args),
            }, ckpt_best)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        # ── Early stopping check ──────────────────────────────────────────────
        if early_stopper.step(val_losses['loss_total']):
            print(
                f"\n  ✗ Early stopping triggered at epoch {epoch}  "
                f"(no significant improvement in val_loss for {args.es_patience} consecutive epochs, "
                f"best={early_stopper.best_loss:.4f})"
            )
            break

    # ── Save last checkpoint ──────────────────────────────────────────────────
    torch.save({
        'epoch'      : epoch,          # actual last epoch (may be < args.epochs if early stopped)
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'val_loss'   : val_losses['loss_total'],
        'args'       : vars(args),
    }, ckpt_last)

    print(f"\nTraining complete.")
    print(f"Best checkpoint : {ckpt_best}")
    print(f"Last checkpoint : {ckpt_last}")
    print(f"Training log    : {log_path}")


if __name__ == '__main__':
    main()
