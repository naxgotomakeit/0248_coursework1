"""
dataloader.py
-------------
Dataset and DataLoader for COMP0248 Hand Gesture RGB-D dataset.

Folder structure expected:
    dataset/
    └── <studentno>_<surname>/
        └── G01_call/
            └── clip01/
                ├── rgb/          frame_001.png, frame_002.png ...
                ├── depth/        frame_001.png, frame_002.png ...
                ├── depth_raw/    frame_001.npy, frame_002.npy ...
                └── annotation/   frame_00X.png, frame_00Y.png  (keyframes only)

Each returned sample:
    {
        'rgb'       : FloatTensor (3, H, W),   normalised to [0,1]
        'depth'     : FloatTensor (1, H, W),   normalised to [0,1]
        'depth_raw' : FloatTensor (1, H, W),   raw mm values
        'mask'      : FloatTensor (1, H, W),   binary 0/1  (0 if no annotation)
        'bbox'      : FloatTensor (4,)          [x1, y1, x2, y2] (0s if no mask)
        'label'     : int                       0-9 gesture class index
        'has_mask'  : bool                      True only for annotated keyframes
        'frame_id'  : str                       e.g. "frame_005"
    }
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


# ── Gesture class mapping ────────────────────────────────────────────────────
GESTURE_CLASSES = [
    'G01_call', 'G02_dislike', 'G03_like', 'G04_ok', 'G05_one',
    'G06_palm', 'G07_peace', 'G08_rock', 'G09_stop', 'G10_three'
]
CLASS_TO_IDX = {g: i for i, g in enumerate(GESTURE_CLASSES)}
IDX_TO_CLASS = {i: g for g, i in CLASS_TO_IDX.items()}


# ── Helper: derive bounding box from binary mask ─────────────────────────────
def mask_to_bbox(mask_np: np.ndarray, img_h: int, img_w: int):
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)
    if not rows.any():
        return [0.0, 0.0, 0.0, 0.0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [
        max(0.0, min(1.0, x1/img_w)),
        max(0.0, min(1.0, y1/img_h)),
        max(0.0, min(1.0, x2/img_w)),
        max(0.0, min(1.0, y2/img_h))
    ]


# ── Main Dataset ─────────────────────────────────────────────────────────────
class HandGestureDataset(Dataset):
    """
    Loads only annotated keyframes by default (has_mask=True).
    Set `keyframes_only=False` to also include unannotated frames
    (mask and bbox will be zero tensors for those).

    Args:
        root_dir       : path to dataset/ folder containing student subfolders
        use_depth      : if True, load depth PNG and depth_raw .npy
        keyframes_only : if True, only return frames that have annotation masks
        transform      : optional custom transform applied to rgb PIL image
        augment        : if True, apply built-in training augmentations
        img_size       : (H, W) to resize all images/masks to
        split_file     : optional path to a .txt file listing clip paths to use
                         (one per line, relative to root_dir). If None, use all.
    """

    def __init__(
        self,
        root_dir: str,
        use_depth: bool = True,
        keyframes_only: bool = True,
        augment: bool = False,
        img_size: tuple = (480, 640),
        split_file: str = None,
    ):
        self.root_dir = Path(root_dir)
        self.use_depth = use_depth
        self.keyframes_only = keyframes_only
        self.augment = augment
        self.img_size = img_size  # (H, W)

        self.samples = []   # list of dicts, one per frame
        self._build_index(split_file)

    # ── Index builder ─────────────────────────────────────────────────────
    def _build_index(self, split_file):
        """Walk dataset folder and collect all (frame, metadata) entries."""

        # Optionally restrict to a pre-defined list of clips
        allowed_clips = None
        # Walk: dataset/<student>/...中间可能有多层套娃.../<gesture>/<clip>/
        for student_dir in sorted(self.root_dir.iterdir()):
            if not student_dir.is_dir():
                continue

            # 🚨 核心修改点：把 iterdir() 改成 rglob('*')
            # rglob('*') 会像雷达一样递归搜索该学生文件夹下的所有子文件夹
            for gesture_dir in sorted(student_dir.rglob('*')):
                if not gesture_dir.is_dir():
                    continue
                
                gesture_name = gesture_dir.name  # 尝试获取文件夹名字
                if gesture_name not in CLASS_TO_IDX:
                    # 遇到中间的“套娃”文件夹（比如 25115816_Jin），静默跳过，继续往下挖
                    continue
                
                label = CLASS_TO_IDX[gesture_name]

                # --- 下面的部分完全保持你原来的样子，不需要动 ---
                for clip_dir in sorted(gesture_dir.iterdir()):
                    if not clip_dir.is_dir():
                        continue

                    # Check split file filter
                    clip_rel = str(clip_dir.relative_to(self.root_dir))
                    if allowed_clips is not None and clip_rel not in allowed_clips:
                        continue
                        
                    # ... 后面读取 rgb 目录的代码保持原样 ...

                    rgb_dir   = clip_dir / 'rgb'
                    dep_dir   = clip_dir / 'depth'
                    depraw_dir = clip_dir / 'depth_raw'
                    ann_dir   = clip_dir / 'annotation'

                    if not rgb_dir.exists():
                        continue

                    # Collect annotated keyframe names
                    annotated_frames = set()
                    if ann_dir.exists():
                        for ann_file in ann_dir.iterdir():
                            if ann_file.suffix.lower() == '.png':
                                # frame name without extension
                                annotated_frames.add(ann_file.stem)

                    # Iterate over all rgb frames in this clip
                    for rgb_file in sorted(rgb_dir.glob('*.png')):
                        frame_id = rgb_file.stem          # e.g. "frame_005"
                        has_mask = frame_id in annotated_frames

                        if self.keyframes_only and not has_mask:
                            continue

                        sample = {
                            'rgb_path'    : str(rgb_file),
                            'depth_path'  : str(dep_dir / f'{frame_id}.png')
                                            if dep_dir.exists() else None,
                            'depraw_path' : str(depraw_dir / f'{frame_id}.npy')
                                            if depraw_dir.exists() else None,
                            'ann_path'    : str(ann_dir / f'{frame_id}.png')
                                            if has_mask else None,
                            'label'       : label,
                            'has_mask'    : has_mask,
                            'frame_id'    : frame_id,
                            'gesture'     : gesture_name,
                        }
                        self.samples.append(sample)

        print(f"[HandGestureDataset] Found {len(self.samples)} frames "
              f"({'keyframes only' if self.keyframes_only else 'all frames'})")

    # ── Length ────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    # ── 增强版Augmentation (applied consistently to rgb, depth, mask) ──────────
    # def _augment(self, rgb_pil, depth_pil, mask_pil):
    # # 随机水平翻转
    #     if random.random() > 0.5:
    #         rgb_pil   = TF.hflip(rgb_pil)
    #         depth_pil = TF.hflip(depth_pil)
    #         mask_pil  = TF.hflip(mask_pil)

    #     # 随机旋转 ±15°
    #     angle = random.uniform(-15, 15)
    #     rgb_pil   = TF.rotate(rgb_pil,   angle)
    #     depth_pil = TF.rotate(depth_pil, angle)
    #     mask_pil  = TF.rotate(mask_pil,  angle)

    #     # ColorJitter只加在RGB上
    #     jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)
    #     rgb_pil = jitter(rgb_pil)

    #     return rgb_pil, depth_pil, mask_pil
   
#VER1-----只有jitter的-------------the basic avalible augmentation is implemented above, you can uncomment it to use.------------------
    def _augment(self, rgb_pil, depth_pil, mask_pil):
    
   
        jitter = T.ColorJitter(brightness=(0.5, 1.0), contrast=(0.5, 1.5), saturation=0.3, hue=0.05)
        rgb_pil = jitter(rgb_pil)

        return rgb_pil, depth_pil, mask_pil
    
# #VER2 ------带旋转的------------the advanced augmentation is implemented below, you can uncomment it to use.------------------
#     def _augment(self, rgb_pil, depth_pil, mask_pil):

#         # 旋转是安全的，不改变手的左右手性
#         angle = random.uniform(-15, 15)
#         rgb_pil   = TF.rotate(rgb_pil,   angle)
#         depth_pil = TF.rotate(depth_pil, angle)
#         mask_pil  = TF.rotate(mask_pil,  angle)

#         # ColorJitter只加在RGB上
#         jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)
#         rgb_pil = jitter(rgb_pil)

#         return rgb_pil, depth_pil, mask_pil
    # ── Main loader ───────────────────────────────────────────────────────
    def __getitem__(self, idx):
        s = self.samples[idx]
        H, W = self.img_size

        # ── RGB ────────────────────────────────────────────────────────
        rgb_pil = Image.open(s['rgb_path']).convert('RGB')

        # ── Depth (visualised PNG) ─────────────────────────────────────
        if self.use_depth and s['depth_path'] and os.path.exists(s['depth_path']):
            depth_pil = Image.open(s['depth_path']).convert('L')   # grayscale
        else:
            # Fallback: black depth image
            depth_pil = Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode='L')

        # ── Mask ───────────────────────────────────────────────────────
        if s['has_mask'] and s['ann_path'] and os.path.exists(s['ann_path']):
            mask_pil = Image.open(s['ann_path']).convert('L')
        else:
            mask_pil = Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode='L')

        # ── Resize to target size (before augment) ─────────────────────
        rgb_pil   = rgb_pil.resize((W, H), Image.BILINEAR)
        depth_pil = depth_pil.resize((W, H), Image.BILINEAR)
        mask_pil  = mask_pil.resize((W, H), Image.NEAREST)

        # ── Augmentation ───────────────────────────────────────────────
        if self.augment:
            rgb_pil, depth_pil, mask_pil = self._augment(rgb_pil, depth_pil, mask_pil)

        # ── Convert to tensors ─────────────────────────────────────────
        rgb_t   = TF.to_tensor(rgb_pil)                        # (3, H, W) float [0,1]
        # depth_t = TF.to_tensor(depth_pil)                      # (1, H, W) float [0,1]
        # 用depth_raw替代depth PNG
        if self.use_depth and s['depraw_path'] and os.path.exists(s['depraw_path']):
            raw_np = np.load(s['depraw_path']).astype(np.float32)
            raw_pil = Image.fromarray(raw_np).resize((W, H), Image.BILINEAR)
            raw_np = np.array(raw_pil, dtype=np.float32)
            raw_np = np.clip(raw_np, 0, 4000) / 4000.0   # 归一化到[0,1]
            depth_t = torch.from_numpy(raw_np).unsqueeze(0)
        else:
            depth_t = torch.zeros(1, H, W, dtype=torch.float32)

        mask_np = np.array(mask_pil, dtype=np.uint8)           # H x W, values 0 or 255
        mask_t  = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)  # (1,H,W)

        # ── depth_raw (.npy) ───────────────────────────────────────────
        if self.use_depth and s['depraw_path'] and os.path.exists(s['depraw_path']):
            raw_np  = np.load(s['depraw_path']).astype(np.float32)  # H x W, mm values
            # Resize to target size
            raw_pil = Image.fromarray(raw_np).resize((W, H), Image.BILINEAR)
            raw_np  = np.array(raw_pil, dtype=np.float32)
            depraw_t = torch.from_numpy(raw_np).unsqueeze(0)        # (1, H, W)
        else:
            depraw_t = torch.zeros(1, H, W, dtype=torch.float32)

        # ── Bounding box from mask ─────────────────────────────────────
        bbox = mask_to_bbox(mask_np, H, W)    # [x1,y1,x2,y2] normalised to [0,1]
        bbox_t = torch.tensor(bbox, dtype=torch.float32)  # (4,)
        # ── Normalise RGB (ImageNet mean/std) ──────────────────────────
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        rgb_t = normalize(rgb_t)
        # Modality Dropout: 30%概率关掉RGB，逼模型学习依赖Depth
        if self.augment and random.random() < 0.18:
            rgb_t = torch.zeros_like(rgb_t)

        return {
            'rgb'       : rgb_t,                        # (3, H, W)
            'depth'     : depth_t,                      # (1, H, W) normalised
            'depth_raw' : depraw_t,                     # (1, H, W) raw mm
            'mask'      : mask_t,                       # (1, H, W) binary
            'bbox'      : bbox_t,                       # (4,)
            'label'     : torch.tensor(s['label'], dtype=torch.long),
            'has_mask'  : s['has_mask'],
            'frame_id'  : s['frame_id'],
            'gesture'   : s['gesture'],
        }


# ── Train / Val split utility ────────────────────────────────────────────────
def split_dataset(dataset: HandGestureDataset, val_ratio: float = 0.2, seed: int = 42):
    """
    Randomly splits a HandGestureDataset into train and val subsets.
    Uses torch.utils.data.Subset so no data is copied.

    Returns: train_subset, val_subset
    """
    from torch.utils.data import Subset
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    split = int(n * (1 - val_ratio))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


# ── DataLoader factory ───────────────────────────────────────────────────────
def get_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    val_ratio: float = 0.2,
    use_depth: bool = True,
    img_size: tuple = (480, 640),
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Convenience function that returns train and val DataLoaders.

    Args:
        root_dir   : path to dataset/ folder
        batch_size : batch size for training loader
        val_ratio  : fraction of data used for validation
        use_depth  : whether to load depth images
        img_size   : (H, W) resize target
        num_workers: DataLoader workers
        seed       : random seed for reproducibility

    Returns:
        train_loader, val_loader
    """
    # 修改 get_dataloaders 函数中的切分逻辑
    n = len(train_dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    split = int(n * (1 - val_ratio))
    
    # 纯手动提取索引
    train_idx, val_idx = indices[:split], indices[split:]

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    # 用关闭了 augment 的 dataset，但是用提取出来的 val_idx
    val_subset = torch.utils.data.Subset(val_dataset, val_idx) 

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,  # 这里的 shuffle 是 batch 层面的，没问题
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,    # 这里传包裹好的 Subset
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_subset, val_subset = split_dataset(train_dataset, val_ratio, seed)
    # Use the same split indices on val_dataset (no augment)
    from torch.utils.data import Subset
    _, val_indices = split_dataset(val_dataset, val_ratio, seed)
    # Re-use the same indices
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_subset)} samples | "
          f"Val: {len(val_indices)} samples")

    return train_loader, val_loader


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else './dataset'

    print("=== Testing HandGestureDataset ===")
    ds = HandGestureDataset(
        root_dir=root,
        use_depth=True,
        keyframes_only=True,
        augment=False,
    )

    if len(ds) == 0:
        print("No samples found. Check your dataset path.")
    else:
        sample = ds[0]
        print(f"\nSample keys     : {list(sample.keys())}")
        print(f"rgb shape       : {sample['rgb'].shape}")
        print(f"depth shape     : {sample['depth'].shape}")
        print(f"depth_raw shape : {sample['depth_raw'].shape}")
        print(f"mask shape      : {sample['mask'].shape}")
        print(f"bbox            : {sample['bbox']}")
        print(f"label           : {sample['label']} ({IDX_TO_CLASS[sample['label'].item()]})")
        print(f"has_mask        : {sample['has_mask']}")
        print(f"frame_id        : {sample['frame_id']}")

        print("\n=== Testing DataLoader ===")
        # ---- Step1: label / gesture consistency check ----
        from collections import Counter, defaultdict

        print("\n=== Label distribution (first pass) ===")
        cnt = Counter([s["label"] for s in ds.samples])
        print({IDX_TO_CLASS[k]: v for k, v in sorted(cnt.items())})

        print("\n=== Check gesture-name vs label mapping (first 50 samples) ===")
        bad = 0
        for s in ds.samples[:50]:
            expected = CLASS_TO_IDX.get(s["gesture"], None)
            if expected != s["label"]:
                bad += 1
                print("Mismatch!", s["gesture"], "label=", s["label"], "path=", s["rgb_path"])
        print("mismatches:", bad)

        print("\n=== Check per-gesture folder has consistent labels ===")
        per_gesture = defaultdict(set)
        for s in ds.samples:
            per_gesture[s["gesture"]].add(s["label"])
        for g, labs in per_gesture.items():
            if len(labs) != 1:
                print("Gesture has multiple labels:", g, labs)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"Batch rgb shape : {batch['rgb'].shape}")
        print(f"Batch labels    : {batch['label']}")
        print("All good!")
