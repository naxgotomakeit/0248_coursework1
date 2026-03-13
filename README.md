# COMP0248 Coursework 1 — Multi-task RGB-D Hand Gesture Recognition

**Student:** Xi Nan · 25194692  
**GitHub:** https://github.com/naxgotomakeit/0248_coursework1

---

## Overview

A dual-stream multi-task CNN that jointly performs hand gesture **detection**, **segmentation**, and **classification** from RGB-D input captured by an Intel RealSense D455.

Three tasks are solved in a single forward pass:
- **Detection** — normalised bounding box `[x1, y1, x2, y2]`
- **Segmentation** — binary hand mask `(1, H, W)`
- **Classification** — 10-class gesture logits `(B, 10)`

Two model variants are provided:

| Variant | Key features |
|---|---|
| **Baseline** | RGB encoder (ResBlock×4) + lightweight Depth encoder (DepthwiseSepConv×4), Late Fusion, RGB-only U-Net skip connections |
| **Improved** | + Raw depth input (.npy), + Modality Dropout (18%), + Dual-stream depth skip connections (d1/d2/d3 concatenated with s1/s2/s3) |

---

## Project Structure

```text
project_<studentno>_<surname>/
├── dataset/
│   └── <studentno>_<surname>/      
├── src/
│   ├── dataloader.py              
│   ├── model.py                    
│   ├── train.py                    
│   ├── evaluate.py                
│   ├── visualise.py               
│   └── utils.py                   
├── weights/                        
├── results/                        
├── requirements.txt
├── improvement   --- the innovation model
│   └──results/  
│   └──weights/
│   └──src/
│      ├── dataloader.py              
│      ├── model.py
└── README.md                
---

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti (24 GB) |
| CUDA | 13.1 |
| Framework | PyTorch |
| Python | 3.10+ |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the dataset under `./dataset/` with the following structure:

```
dataset/
└── <studentno>_<surname>/
    └── G01_call/
        └── clip01/
            ├── rgb/          # frame_001.png, frame_002.png ...
            ├── depth/        # frame_001.png, frame_002.png ...
            ├── depth_raw/    # frame_001.npy, frame_002.npy ...
            └── annotation/   # keyframes only (frame_00X.png)
```

10 gesture classes: `G01_call`, `G02_dislike`, `G03_like`, `G04_ok`, `G05_one`, `G06_palm`, `G07_peace`, `G08_rock`, `G09_stop`, `G10_three`

- 31 students (2 excluded due to corrupted folder structures)
- ~3,100 annotated keyframes total
- Captured at 640×480, 3 FPS
- Classes are perfectly balanced — no class weighting needed

---

## CLI Arguments

### `train.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_root` | str | `./dataset` | Path to dataset folder |
| `--save_dir` | str | `./weights` | Directory for saved checkpoints |
| `--results_dir` | str | `./results` | Directory for logs and outputs |
| `--epochs` | int | `80` | Maximum number of training epochs |
| `--batch_size` | int | `8` | Training batch size |
| `--lr` | float | `1e-3` | Initial learning rate |
| `--val_ratio` | float | `0.2` | Fraction of students used for validation |
| `--img_h` | int | `480` | Input image height |
| `--img_w` | int | `640` | Input image width |
| `--w_det` | float | `1.0` | Detection loss weight |
| `--w_seg` | float | `1.0` | Segmentation loss weight |
| `--w_cls` | float | `1.0` | Classification loss weight |
| `--no_depth` | flag | `False` | RGB-only mode (ablation study) |
| `--num_workers` | int | `4` | DataLoader worker processes |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--es_patience` | int | `10` | Early stopping patience (epochs) |
| `--es_min_delta` | float | `1e-4` | Minimum val_loss improvement to reset patience |

### `evaluate.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_root` | str | *(required)* | Path to dataset folder |
| `--checkpoint` | str | *(required)* | Path to model checkpoint `.pth` |
| `--results_dir` | str | `./results` | Directory for output files |
| `--batch_size` | int | `8` | Evaluation batch size |
| `--img_h` | int | `480` | Input image height |
| `--img_w` | int | `640` | Input image width |
| `--val_ratio` | float | `0.2` | Must match the value used during training |
| `--seed` | int | `42` | Must match the value used during training |
| `--no_depth` | flag | `False` | RGB-only mode |
| `--test` | flag | `False` | Evaluate on full dataset (no split) |
| `--num_workers` | int | `4` | DataLoader worker processes |

### `visualise.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | str | `all` | One of `curves`, `overlay`, `dist`, `all` |
| `--log` | str | `None` | Path to `train_log.csv` (for `curves` mode) |
| `--checkpoint` | str | `None` | Path to model checkpoint (for `overlay` mode) |
| `--data_root` | str | `./dataset` | Path to dataset folder |
| `--results_dir` | str | `./results` | Directory for output figures |
| `--img_h` | int | `480` | Input image height |
| `--img_w` | int | `640` | Input image width |
| `--val_ratio` | float | `0.2` | Must match the value used during training |
| `--seed` | int | `42` | Must match the value used during training |
| `--no_depth` | flag | `False` | RGB-only mode |
| `--n_per_class` | int | `1` | Number of overlay examples per gesture class |

---

## Training

```bash
python train.py \
  --data_root ./dataset \
  --epochs 80 \
  --batch_size 32 \
  --lr 3e-5 \
  --img_h 480 \
  --img_w 640 \
  --w_det 1.0 \
  --w_seg 1.0 \
  --w_cls 5.0 \
  --es_patience 10 \
  --es_min_delta 1e-4
```

Key training details:
- **Optimiser:** AdamW, weight decay 1e-4
- **LR schedule:** Cosine Annealing (min 1e-6)
- **Gradient clipping:** max norm 5.0
- **Split:** Student-level (no student appears in both train and val)
- **Augmentation:** ColorJitter on RGB only; Modality Dropout (18% probability of zeroing RGB) during training
- **Loss:** `L_total = w_det·L_det + w_seg·L_seg + w_cls·L_cls`
  - `L_det`: Smooth L1
  - `L_seg`: BCE + Dice
  - `L_cls`: Cross-entropy

For RGB-only ablation:
```bash
python train.py --data_root ./dataset --no_depth
```

---

## Evaluation

```bash
# Validation set (same student split as training)
python evaluate.py \
  --data_root ./dataset \
  --checkpoint ./weights/<run_name>_best.pth

# Full test set
python evaluate.py \
  --data_root ./test_dataset \
  --checkpoint ./weights/<run_name>_best.pth \
  --test
```

Metrics reported:
- Detection: Accuracy @ IoU=0.5, Mean BBox IoU
- Segmentation: Mean IoU, Dice coefficient
- Classification: Top-1 accuracy, Macro F1, Confusion matrix (PNG)

---

## Visualisation

```bash
# Training loss curves
python visualise.py --mode curves --log results/<run_name>_log.csv

# Qualitative overlay (RGB + GT/Pred mask + bbox + classification bar)
python visualise.py --mode overlay \
  --checkpoint ./weights/<run_name>_best.pth \
  --data_root ./dataset

# Dataset class distribution
python visualise.py --mode dist --data_root ./dataset

# All of the above
python visualise.py --mode all \
  --checkpoint ./weights/<run_name>_best.pth \
  --data_root ./dataset
```

---

## Results

### Quantitative

| Metric | Baseline Val | Baseline Test | Baseline Clip15 | Improved Val | Improved Test | Improved Clip15 |
|---|---|---|---|---|---|---|
| Detection @ IoU=0.5 | 95.50% | 88.35% | 0.00% | 6.67% | 7.57% | 16.00% |
| Mean BBox IoU | 77.54% | 70.35% | 19.77% | 17.15% | 15.63% | 36.18% |
| Mean Seg IoU | 93.37% | 84.54% | 0.01% | 80.91% | 78.98% | 23.77% |
| Dice Score | 96.35% | 88.52% | 0.02% | 88.33% | 85.89% | 35.35% |
| Top-1 Accuracy | 97.96% | 86.41% | 10.00% | 80.78% | 76.96% | 10.00% |
| Macro F1 | 97.96% | 86.58% | 1.99% | 80.84% | 77.07% | 1.82% |

### Key Findings

- The baseline achieves strong normal-illumination performance but fails completely on the dark clip15 (0% detection, 0.01% seg IoU).
- The improved model raises clip15 seg IoU from 0.01% to 23.77% and Dice from 0.02% to 35.35%, confirming that raw depth input + depth skip connections + modality dropout meaningfully improve low-illumination robustness.
- The 18% modality dropout probability introduces a trade-off: while it forces the model to learn depth features, it disrupts RGB texture cues needed for detection, causing a sharp drop in detection accuracy on normal-illumination scenes.

---

## References

1. V. Villani et al., "Survey on human–robot collaboration in industrial settings," *Mechatronics*, vol. 55, 2018.
2. W. Wang, D. Tran, and M. Feiszli, "What makes training multi-modal classification networks hard?" *CVPR*, 2020.
