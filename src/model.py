"""
model.py
--------
Multi-task RGB-D Hand Gesture Model.

Architecture overview:
                                          ┌─────────────────────┐
    RGB  (3, H, W)  ──► RGB Encoder  ────┤                      │
                        (Custom CNN)      │  Late Fusion         ├──► Fused Feature (256, H/16, W/16)
    Depth(1, H, W)  ──► Depth Encoder ───┤  (concat + conv)     │
                        (Custom CNN)      └─────────────────────┘
                                                    │
                          ┌─────────────────────────┼──────────────────────┐
                          ▼                         ▼                      ▼
                  Detection Head            Segmentation Head       Classification Head
                  (bbox regression)         (U-Net decoder          (GAP + FC)
                  → (4,) [x1,y1,x2,y2]      with skip conns)        → (10,) logits
                                            → (1, H, W) mask

Innovation points:
  1. Custom CNN encoders written from scratch with torch.nn.Module — fully
     compliant with coursework requirements (no high-level frameworks).
  2. Late fusion: RGB and Depth encoded separately, then concatenated and
     fused — each modality learns its own representation before merging.
  3. Shared fused feature feeds all three task heads jointly — multi-task
     learning regularises the shared representation.
  4. U-Net skip connections from RGB encoder for fine-grained segmentation.
  5. Depth encoder uses depthwise-separable convolutions to stay lightweight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 10


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    """Conv → BN → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    """
    Residual block: two 3×3 convs + skip connection.
    Handles channel/stride mismatch with a 1×1 projection on the skip.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class DepthwiseSepConv(nn.Module):
    """
    Depthwise-separable convolution — cheaper than standard conv.
    Used in the depth encoder to stay lightweight.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                              groups=in_ch, bias=False)
        self.pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))


# ─────────────────────────────────────────────────────────────────────────────
# RGB Encoder
# ─────────────────────────────────────────────────────────────────────────────

class RGBEncoder(nn.Module):
    """
    Custom CNN encoder for RGB images. Returns 4 feature maps at different
    scales for use as U-Net skip connections.

    Input : (B, 3, H, W)
    Output:
        s1 : (B, 64,  H/2,  W/2)
        s2 : (B, 128, H/4,  W/4)
        s3 : (B, 256, H/8,  W/8)
        s4 : (B, 256, H/16, W/16)  ← main feature, fed to fusion
    """
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(          # 3 → 64,  H → H/2
            conv_bn_relu(3,  32, stride=1),
            conv_bn_relu(32, 64, stride=2),
            ResBlock(64, 64),
        )
        self.stage2 = nn.Sequential(          # 64 → 128, H/2 → H/4
            ResBlock(64,  128, stride=2),
            ResBlock(128, 128),
        )
        self.stage3 = nn.Sequential(          # 128 → 256, H/4 → H/8
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )
        self.stage4 = nn.Sequential(          # 256 → 256, H/8 → H/16
            ResBlock(256, 256, stride=2),
            ResBlock(256, 256),
        )

    def forward(self, x):
        s1 = self.stage1(x)    # (B, 64,  H/2,  W/2)
        s2 = self.stage2(s1)   # (B, 128, H/4,  W/4)
        s3 = self.stage3(s2)   # (B, 256, H/8,  W/8)
        s4 = self.stage4(s3)   # (B, 256, H/16, W/16)
        return s1, s2, s3, s4


# ─────────────────────────────────────────────────────────────────────────────
# Depth Encoder
# ─────────────────────────────────────────────────────────────────────────────

class DepthEncoder(nn.Module):
    """
    Lightweight custom CNN for the single-channel depth map.
    Depthwise-separable convs reduce parameters since depth carries geometry,
    not texture — fewer feature channels are needed.

    Input : (B, 1, H, W)
    Output: (B, 128, H/16, W/16)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(1,  32,  stride=2),             # H/2
            DepthwiseSepConv(32,  64,  stride=2),        # H/4
            DepthwiseSepConv(64,  128, stride=2),        # H/8
            DepthwiseSepConv(128, 128, stride=2),        # H/16
        )

    def forward(self, d):
        return self.net(d)    # (B, 128, H/16, W/16)


# ─────────────────────────────────────────────────────────────────────────────
# Late Fusion
# ─────────────────────────────────────────────────────────────────────────────

class LateFusion(nn.Module):
    """
    Concatenates RGB s4 (256ch) + Depth features (128ch) → fused (256ch).

    Late fusion means each modality encodes independently first, preserving
    modality-specific representations before they are merged.
    """
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            conv_bn_relu(256 + 128, 256, kernel=3, padding=1),
            conv_bn_relu(256,       256, kernel=1, padding=0),
        )

    def forward(self, rgb_feat, depth_feat):
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            depth_feat = F.interpolate(
                depth_feat, size=rgb_feat.shape[-2:],
                mode='bilinear', align_corners=False
            )
        x = torch.cat([rgb_feat, depth_feat], dim=1)   # (B, 384, H/16, W/16)
        return self.fuse(x)                             # (B, 256, H/16, W/16)


# ─────────────────────────────────────────────────────────────────────────────
# Detection Head VER lr = 4e-4
# ─────────────────────────────────────────────────────────────────────────────

# class DetectionHead(nn.Module):
#     """
#     保留 4x4 的空间结构，让网络知道物体大概在哪个象限，再进行 BBox 回归
#     """
#     def __init__(self, in_ch: int = 256, img_size: tuple = (480,640)):
#         super().__init__()
        
#         # 4x4 = 16. 保留了 16 个空间区块的位置信息
#         self.spatial_size = 4 
        
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(self.spatial_size), 
#             nn.Flatten(),
#             # 展平后维度是 256 * 4 * 4 = 4096
#             nn.Linear(in_ch * self.spatial_size * self.spatial_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(128, 4),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.head(x)

# ─────────────────────────────────────────────────────────────────────────────
# Detection Head VER lr = 5e-4
# ─────────────────────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    def __init__(self, in_ch, img_size: tuple = (480,640)):
        super().__init__()
        
        # 创新点 1: 引入连续的 3x3 卷积提取空间几何特征 (边缘、角点)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(), # GELU 比 ReLU 在复杂回归任务上表现更平滑
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # 创新点 2: 提升空间网格分辨率 (从 4x4 提升到 7x7)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.flatten = nn.Flatten()
        
        # 创新点 3: 更深的非线性回归器 + Dropout 防过拟合
        self.regressor = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.GELU(),
            nn.Dropout(0.2), # 防止网络死记硬背训练集的坐标
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 4),
            nn.Sigmoid() # 物理外挂：强制保证输出坐标永远在 [0, 1] 之间！
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        x = self.flatten(x)
        bbox = self.regressor(x)
        return bbox

# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Head (U-Net decoder)
# ─────────────────────────────────────────────────────────────────────────────

class UNetDecoderBlock(nn.Module):
    """Upsample ×2 → concat skip → conv ×2."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch // 2 + skip_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SegmentationHead(nn.Module):
    """
    U-Net decoder using skip connections from the RGB encoder.

    For 320×320 input:
        fused (256, 20, 20)
            → dec3 → (128, 40, 40)   skip=s3 (256, 40, 40)
            → dec2 → (64,  80, 80)   skip=s2 (128, 80, 80)
            → dec1 → (32, 160, 160)  skip=s1 (64, 160, 160)
            → up   → (16, 320, 320)
            → 1×1  → (1,  320, 320)  raw logits
    """
    def __init__(self):
        super().__init__()
        self.dec3     = UNetDecoderBlock(256, 256, 128)
        self.dec2     = UNetDecoderBlock(128, 128, 64)
        self.dec1     = UNetDecoderBlock(64,  64,  32)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.out      = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, fused, s1, s2, s3):
        x = self.dec3(fused, s3)    # (B, 128, H/8,  W/8)
        x = self.dec2(x,     s2)    # (B, 64,  H/4,  W/4)
        x = self.dec1(x,     s1)    # (B, 32,  H/2,  W/2)
        x = self.up_final(x)        # (B, 16,  H,    W)
        return self.out(x)          # (B, 1,   H,    W)


# ─────────────────────────────────────────────────────────────────────────────
# Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        # 不用GAP，改用卷积进一步提特征再分类
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch, 128, kernel=3, padding=1),
            conv_bn_relu(128, 64, kernel=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(2)   # 保留2×2空间结构而不是1×1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)    # (B, 64, 20, 20)
        x = self.pool(x)    # (B, 64, 2, 2)
        return self.fc(x)   # (B, 10)

# ─────────────────────────────────────────────────────────────────────────────
# Full Multi-task Model
# ─────────────────────────────────────────────────────────────────────────────

class HandGestureModel(nn.Module):
    """
    Full multi-task model: detection + segmentation + classification from RGB-D.

    Args:
        img_size  : (H, W) — must match DataLoader img_size.
        use_depth : if False, bypasses depth encoder (RGB-only ablation mode).

    Forward input:
        rgb   : (B, 3, H, W)
        depth : (B, 1, H, W)

    Forward output (dict):
        'bbox'   : (B, 4)        [x1, y1, x2, y2] pixel coords
        'mask'   : (B, 1, H, W)  raw logits  → sigmoid for probability
        'logits' : (B, 10)       class logits → softmax for probability
    """
    def __init__(self, img_size: tuple = (320, 320), use_depth: bool = True):
        super().__init__()
        self.use_depth = use_depth

        self.rgb_encoder   = RGBEncoder()
        self.depth_encoder = DepthEncoder() if use_depth else None
        self.fusion        = LateFusion()   if use_depth else None

        if not use_depth:
            # Keep channel count consistent for the heads
            self.rgb_only_proj = conv_bn_relu(256, 256, kernel=1, padding=0)

        self.det_head = DetectionHead(in_ch=256, img_size=img_size)
        self.seg_head = SegmentationHead()
        self.cls_head = ClassificationHead(in_ch=256)

    def forward(self, rgb, depth):
        s1, s2, s3, s4 = self.rgb_encoder(rgb)

        if self.use_depth and self.depth_encoder is not None:
            depth_feat = self.depth_encoder(depth)
            fused      = self.fusion(s4, depth_feat)
        else:
            fused = self.rgb_only_proj(s4)

        bbox   = self.det_head(fused)
        mask   = self.seg_head(fused, s1, s2, s3)

        
        # logits = self.cls_head(s3)
        logits = self.cls_head(fused)

        return {'bbox': bbox, 'mask': mask, 'logits': logits}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task Loss
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    L_total = w_det * L_det  +  w_seg * L_seg  +  w_cls * L_cls

    L_det : Smooth L1 on bounding box              (annotated frames only)
    L_seg : BCE + Dice on segmentation mask        (annotated frames only)
    L_cls : Cross-entropy on 10-class gesture      (all frames)

    Detection and segmentation losses are only computed on keyframes that
    have ground-truth annotations (has_mask=True in the batch).
    """
    def __init__(self, w_det: float = 1.0, w_seg: float = 1.0, w_cls: float = 1.0):
        super().__init__()
        self.w_det     = w_det
        self.w_seg     = w_seg
        self.w_cls     = w_cls
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce       = nn.BCEWithLogitsLoss()
        self.ce        = nn.CrossEntropyLoss()

    # @staticmethod
    # def dice_loss(pred_logits, target, eps=1e-6):
    #     pred  = torch.sigmoid(pred_logits)
    #     p     = pred.view(pred.size(0), -1)
    #     t     = target.view(target.size(0), -1)
    #     inter = (p * t).sum(dim=1)
    #     dice  = (2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    #     return 1 - dice.mean()

    @staticmethod
    def dice_loss(pred_logits, target, eps=1e-5):
        # 稍微调大一点 eps 防止除 0
        pred  = torch.sigmoid(pred_logits)
        
        # 展平张量
        p = pred.view(-1)
        t = target.view(-1)
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice
    def forward(self, preds: dict, targets: dict):
        """
        targets keys:
            'bbox'     : (B, 4)
            'mask'     : (B, 1, H, W)  binary float
            'label'    : (B,)          long
            'has_mask' : (B,)          bool
        """
        has_mask = targets['has_mask']

        l_cls = self.ce(preds['logits'], targets['label'])

        if has_mask.any():
            l_det = self.smooth_l1(
                preds['bbox'][has_mask], targets['bbox'][has_mask]
            )
            mp = preds['mask'][has_mask]
            mg = targets['mask'][has_mask]
            l_seg = self.bce(mp, mg) + self.dice_loss(mp, mg)
        else:
            l_det = torch.tensor(0.0, device=preds['bbox'].device)
            l_seg = torch.tensor(0.0, device=preds['mask'].device)

        total = self.w_det * l_det + self.w_seg * l_seg + self.w_cls * l_cls
        return total, {
            'loss_total': total.item(),
            'loss_det'  : l_det.item(),
            'loss_seg'  : l_seg.item(),
            'loss_cls'  : l_cls.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    H, W, B = 320, 320, 2
    model     = HandGestureModel(img_size=(H, W), use_depth=True).to(device)
    criterion = MultiTaskLoss()

    rgb   = torch.randn(B, 3, H, W).to(device)
    depth = torch.randn(B, 1, H, W).to(device)
    targets = {
        'bbox'    : torch.rand(B, 4).to(device),
        'mask'    : (torch.rand(B, 1, H, W) > 0.5).float().to(device),
        'label'   : torch.randint(0, NUM_CLASSES, (B,)).to(device),
        'has_mask': torch.tensor([True, True]).to(device),
    }

    preds = model(rgb, depth)
    print(f"bbox   : {preds['bbox'].shape}")     # (2, 4)
    print(f"mask   : {preds['mask'].shape}")     # (2, 1, 320, 320)
    print(f"logits : {preds['logits'].shape}")   # (2, 10)

    total, losses = criterion(preds, targets)
    print(f"\nLosses : {losses}")

    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n:,}")
