# -*- coding: utf-8 -*-
"""
Backend ML - Slum Detection via Satellite Imagery
Architecture: SatMAE-style ViT-B + PSP + FPN + Segmentation Head
Checkpoint keys confirm:
  encoder: ViT-B (embed_dim=768, depth=12, patch_size=16, in_chans=6, 3D-conv patch embed)
  decoder: 4x PSP pools + bottleneck(1792→256) + 11 lateral(768→256) + 11 FPN(256→256)
           + fpn_bottleneck(3072→256)
  head:    Conv2d(256, 2, 1)   [2 classes: non-slum, slum]
"""
import io
import math
import json
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ──────────────────────────────────────────────────────────────
# 1. MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────

class ConvNormLayer(nn.Module):
    """Conv + BatchNorm (no bias in conv)."""
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)), inplace=True)


class PatchEmbed3D(nn.Module):
    """3-D Conv patch embedding: (B, T, C, H, W) → (B, N, embed_dim)."""
    def __init__(self, in_chans=6, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(1, patch_size, patch_size),
                              stride=(1, patch_size, patch_size))

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)           # (B, C, T, H, W)
        x = self.proj(x)                        # (B, embed_dim, T, h, w)
        _, D, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, T*h*w, embed_dim)
        return x, t, h, w


class ScaledEmbed(nn.Module):
    """Learnable scalar-scaled positional embedding (temporal / location)."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, info=None):
        # info unused at inference; returns x unchanged (no positional offset needed)
        return x * (1.0 + self.scale)


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT-B encoder with 3-D patch embedding, temporal + location embeddings."""
    def __init__(self, in_chans=6, embed_dim=768, depth=12,
                 num_heads=12, patch_size=16, img_size=224):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2   # 196
        self.patch_embed       = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.cls_token         = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed         = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_embed_enc  = ScaledEmbed()
        self.location_embed_enc  = ScaledEmbed()
        self.blocks            = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm              = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x : (B, T, C, H, W)
        Returns list of intermediate feature maps [h×w spatial, embed_dim channels]
        """
        B, T, C, H, W = x.shape
        tokens, t, h, w = self.patch_embed(x)   # (B, T*h*w, 768)

        # Tile pos_embed for T time steps (exclude cls token slot)
        pos = self.pos_embed[:, 1:, :]            # (1, h*w, 768)
        pos = pos.repeat(1, t, 1)                 # (1, T*h*w, 768)
        tokens = tokens + pos

        tokens = self.temporal_embed_enc(tokens)
        tokens = self.location_embed_enc(tokens)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # prepend cls

        intermediates = []
        for blk in self.blocks:
            tokens = blk(tokens)
            # collect patch tokens (drop cls) → (B, embed_dim, h, w)
            feat = tokens[:, 1:, :]               # (B, T*h*w, 768)
            # take only first time-step slice for spatial features
            feat = feat[:, :h*w, :]               # (B, h*w, 768)
            feat = feat.transpose(1, 2).reshape(B, -1, h, w)  # (B,768,h,w)
            intermediates.append(feat)

        tokens = self.norm(tokens)
        return intermediates, h, w


# ── Decoder ───────────────────────────────────────────────────

class PPMModule(nn.Module):
    """Pyramid Pooling Module (PSP)."""
    def __init__(self, in_ch=768, out_ch=256, pool_size=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        # index [1] in checkpoint: psp_modules.i.1
        self._1 = ConvNormLayer(in_ch, out_ch, kernel=1)  # named as attribute '1' via state_dict

    def forward(self, x):
        h, w = x.shape[-2:]
        feat = self.pool(x)
        feat = self._1(feat)
        return F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)


class PSPDecoder(nn.Module):
    """
    Full PSP + FPN decoder matching the checkpoint layout:
      psp_modules:   4 × PPM pools → 4 × (B,256,h,w)
      bottleneck:    cat(original + 4 pools) = 768+4*256=1792 → 256   [3×3 conv]
      lateral_convs: 11 × (768→256, 1×1)
      fpn_convs:     11 × (256→256, 3×3)
      fpn_bottleneck: cat of 12 levels: 12×256=3072 → 256             [3×3 conv]
    """
    NUM_POOLS = 4      # psp_modules 0-3
    NUM_LATERAL = 11   # lateral_convs 0-10  (blocks 1-11)

    def __init__(self, in_ch=768, out_ch=256):
        super().__init__()
        pool_sizes = [1, 2, 3, 6]
        self.psp_modules = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(ps), ConvNormLayer(in_ch, out_ch, kernel=1))
            for ps in pool_sizes
        ])
        # bottleneck: in = in_ch + 4*out_ch = 768 + 1024 = 1792
        self.bottleneck   = ConvNormLayer(in_ch + out_ch * self.NUM_POOLS, out_ch, kernel=3, padding=1)
        self.lateral_convs = nn.ModuleList([ConvNormLayer(in_ch, out_ch, kernel=1)
                                            for _ in range(self.NUM_LATERAL)])
        self.fpn_convs    = nn.ModuleList([ConvNormLayer(out_ch, out_ch, kernel=3, padding=1)
                                           for _ in range(self.NUM_LATERAL)])
        # fpn_bottleneck: cat (NUM_LATERAL+1)=12 levels each 256 → 3072 → 256
        self.fpn_bottleneck = ConvNormLayer((self.NUM_LATERAL + 1) * out_ch, out_ch, kernel=3, padding=1)

    def forward(self, intermediates, h, w):
        """
        intermediates: list of 12 feature maps (B,768,h,w) from ViT blocks 0-11
        Returns segmentation feature map (B,256,H_out,W_out)
        """
        # --- PSP on last block output ---
        last = intermediates[-1]    # (B,768,h,w)
        psp_feats = [last]
        for pool_mod in self.psp_modules:
            pooled = pool_mod[0](last)      # AdaptiveAvgPool
            projected = pool_mod[1](pooled) # ConvNormLayer
            projected = F.interpolate(projected, size=(h, w),
                                      mode='bilinear', align_corners=False)
            psp_feats.append(projected)
        psp_out = torch.cat(psp_feats, dim=1)  # (B, 1792, h, w)
        psp_out = self.bottleneck(psp_out)      # (B, 256, h, w)

        # --- FPN on blocks 1-11 (lateral_convs 0-10) ---
        fpn_feats = [psp_out]   # base level from PSP output
        for i, (lat, fpn) in enumerate(zip(self.lateral_convs, self.fpn_convs)):
            feat = intermediates[i + 1]         # blocks 1..11
            feat = lat(feat)                    # (B,256,h,w)
            feat = fpn(feat)
            fpn_feats.append(feat)

        # Upsample all to same spatial size and cat
        target_h, target_w = h, w
        feats_up = []
        for f in fpn_feats:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            feats_up.append(f)

        concat = torch.cat(feats_up, dim=1)     # (B, 12*256=3072, h, w)
        out = self.fpn_bottleneck(concat)       # (B, 256, h, w)
        return out


class SegHead(nn.Module):
    """Segmentation head matching head.head.2.weight [2, 256, 1, 1]."""
    def __init__(self, in_ch=256, num_classes=2):
        super().__init__()
        # head.head is a Sequential; only index 2 has weights
        # indices 0 and 1 are activation / dropout → no params
        # We reconstruct as: Dropout → ReLU → Conv1x1
        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class SlumSegModel(nn.Module):
    """Full slum segmentation model: ViT-B + PSP + FPN + Head."""
    def __init__(self):
        super().__init__()
        self.encoder = ViTEncoder(in_chans=6, embed_dim=768, depth=12,
                                  num_heads=12, patch_size=16, img_size=224)
        self.decoder = PSPDecoder(in_ch=768, out_ch=256)
        self.head    = SegHead(in_ch=256, num_classes=2)

    def forward(self, x):
        """x: (B, T, C, H, W)  → logits (B, 2, H, W)"""
        intermediates, h, w = self.encoder(x)
        feat  = self.decoder(intermediates, h, w)     # (B, 256, h, w)
        logits = self.head(feat)                       # (B, 2, h, w)
        # Upsample to input spatial size
        H, W = x.shape[-2], x.shape[-1]
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits


# ──────────────────────────────────────────────────────────────
# 2. LOAD CHECKPOINT
# ──────────────────────────────────────────────────────────────

device = torch.device('cpu')
model  = None

try:
    checkpoint   = torch.load('model.pt', map_location=device)
    state_dict   = checkpoint.get('model_state_dict', checkpoint)

    m = SlumSegModel()
    missing, unexpected = m.load_state_dict(state_dict, strict=False)

    # Only warn about truly missing keys (not just PSP pool-size sub-modules)
    critical_missing = [k for k in missing if 'num_batches_tracked' not in k]
    if critical_missing:
        print(f"⚠️  Missing keys ({len(critical_missing)}): {critical_missing[:5]}...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    m.eval()
    model = m

    meta = {
        'epoch':       checkpoint.get('epoch', 'N/A'),
        'val_iou_slum': float(checkpoint.get('val_iou_slum', 0)),
        'miou':         float((checkpoint.get('val_metrics') or {}).get('miou', 0)),
    }
    print(f"[OK] Model loaded - epoch {meta['epoch']}, IoU slum {meta['val_iou_slum']:.4f}, mIoU {meta['miou']:.4f}")
except Exception as e:
    print(f"[FAIL] Model load failed: {e}")
    model = None
    meta  = {}


# ──────────────────────────────────────────────────────────────
# 3. PREPROCESSING HELPERS
# ──────────────────────────────────────────────────────────────

# Sentinel-2 band stats (approx. for normalisation)
# Bands: B2, B3, B4, B8A, B11, B12
BAND_MEAN = [485.0,  559.0,  615.0,  2624.0, 1702.0, 1221.0]
BAND_STD  = [254.0,  244.0,  282.0,   572.0,  502.0,  476.0]

IMG_SIZE  = 224
NUM_BANDS = 6


def normalize_bands(arr: np.ndarray) -> np.ndarray:
    """arr shape (C, H, W), returns float32 normalised."""
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(min(arr.shape[0], NUM_BANDS)):
        out[i] = (arr[i].astype(np.float32) - BAND_MEAN[i]) / (BAND_STD[i] + 1e-8)
    return out


def prepare_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    arr: (C, H, W)  float32 normalised
    → tensor (1, 1, C, IMG_SIZE, IMG_SIZE)  [batch=1, T=1, C, H, W]
    """
    arr = arr[:NUM_BANDS]   # keep only first 6 bands

    # Resize to IMG_SIZE×IMG_SIZE
    t = torch.from_numpy(arr).unsqueeze(0)          # (1, C, H, W)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                      mode='bilinear', align_corners=False)  # (1, C, 224, 224)
    t = t.unsqueeze(0).unsqueeze(0)                 # (1, 1, C, 224, 224)
    return t


def mask_to_geojson(mask: np.ndarray, bbox: list) -> dict:
    """
    Convert binary segmentation mask → GeoJSON FeatureCollection.
    mask: (H, W) uint8  (1 = slum)
    bbox: [west, south, east, north]  (lon/lat)
    """
    west, south, east, north = bbox
    H, W = mask.shape

    # Calculate pixel-to-geo transform
    px_w = (east  - west)  / W
    px_h = (north - south) / H

    features = []
    # Simple approach: find connected blobs via run-length scan rows
    slum_pixels = np.argwhere(mask == 1)
    if len(slum_pixels) == 0:
        return {
            "type": "FeatureCollection", 
            "features": [],
            "metadata": {
                "slum_pixels": 0,
                "total_pixels": int(mask.size),
                "slum_ratio": 0.0,
            }
        }

    # Group contiguous rows into rough polygons (simplified, no cv2 needed)
    # We emit one rectangle per contiguous row-segment of slum pixels
    visited = np.zeros_like(mask, dtype=bool)
    for (r, c) in slum_pixels:
        if visited[r, c]:
            continue
        # find run extent in this row
        c_end = c
        while c_end + 1 < W and mask[r, c_end + 1] == 1:
            c_end += 1
        visited[r, c:c_end+1] = True

        lon0 = west  + c     * px_w
        lon1 = west  + (c_end + 1) * px_w
        lat0 = north - r     * px_h
        lat1 = north - (r + 1) * px_h

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon0, lat0], [lon1, lat0],
                    [lon1, lat1], [lon0, lat1],
                    [lon0, lat0],
                ]]
            },
            "properties": {"class": "slum", "row": int(r), "col_start": int(c)}
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "slum_pixels": int(np.sum(mask)),
            "total_pixels": int(mask.size),
            "slum_ratio": float(np.sum(mask) / mask.size),
        }
    }


# ──────────────────────────────────────────────────────────────
# 4. FASTAPI APP
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="Slum Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_meta": meta,
        "rasterio": HAS_RASTERIO,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    west:  float = Form(...),
    south: float = Form(...),
    east:  float = Form(...),
    north: float = Form(...),
    threshold: float = Form(0.5),
):
    """
    Accepts a GeoTIFF or multi-band image upload.
    Requires: west, south, east, north (bounding box in WGS84)
    Returns GeoJSON FeatureCollection with slum polygons.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    raw = await file.read()

    # ── Parse image ─────────────────────────────────────────
    arr = None
    if HAS_RASTERIO:
        try:
            with rasterio.open(io.BytesIO(raw)) as src:
                arr = src.read().astype(np.float32)  # (bands, H, W)
        except Exception:
            pass

    if arr is None and HAS_PIL:
        try:
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            rgb = np.array(img).transpose(2, 0, 1).astype(np.float32)  # (3,H,W)
            # Pad RGB to 6 bands by repeating last band
            extra = np.repeat(rgb[[-1]], NUM_BANDS - rgb.shape[0], axis=0)
            arr = np.concatenate([rgb, extra], axis=0)
        except Exception as e:
            raise HTTPException(400, f"Cannot parse image: {e}")

    if arr is None:
        raise HTTPException(400, "Cannot parse uploaded file (rasterio and PIL unavailable)")

    # ── Normalise & run inference ────────────────────────────
    arr_norm = normalize_bands(arr)
    tensor   = prepare_tensor(arr_norm).to(device)

    with torch.no_grad():
        logits = model(tensor)                   # (1, 2, H, W)
        probs  = torch.softmax(logits, dim=1)    # (1, 2, H, W)
        slum_prob = probs[0, 1].cpu().numpy()    # (H, W)  prob of slum class

    mask = (slum_prob >= threshold).astype(np.uint8)

    bbox    = [west, south, east, north]
    geojson = mask_to_geojson(mask, bbox)
    geojson["metadata"]["slum_confidence_mean"] = float(slum_prob[mask == 1].mean()) if mask.any() else 0.0

    return JSONResponse(content=geojson)


@app.get("/predict/demo")
def predict_demo(lat: float = -6.2, lon: float = 106.8, radius_km: float = 1.0):
    """
    Demo endpoint (no upload required).
    Returns a synthetic GeoJSON for testing the frontend.
    Uses the model on a zero-initialised tensor to test the pipeline.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    deg_per_km = 1 / 111.0
    r = radius_km * deg_per_km
    bbox = [lon - r, lat - r, lon + r, lat + r]

    # Synthetic 6-band input
    dummy = torch.zeros(1, 1, NUM_BANDS, IMG_SIZE, IMG_SIZE, device=device)

    with torch.no_grad():
        logits = model(dummy)
        probs  = torch.softmax(logits, dim=1)
        slum_prob = probs[0, 1].cpu().numpy()

    threshold = 0.45
    mask = (slum_prob >= threshold).astype(np.uint8)

    geojson = mask_to_geojson(mask, bbox)
    geojson["metadata"]["mode"] = "demo"
    geojson["metadata"]["center"] = {"lat": lat, "lon": lon}
    return JSONResponse(content=geojson)