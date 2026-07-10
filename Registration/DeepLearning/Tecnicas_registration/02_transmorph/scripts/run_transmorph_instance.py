from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


ROOT = Path.cwd()
DL_ROOT = ROOT / "Registration" / "DeepLearning"
BASELINE_DIR = DL_ROOT / "Tecnicas_registration" / "00_baseline_clasico" / "outputs"
METHOD_DIR = DL_ROOT / "Tecnicas_registration" / "02_transmorph"


def set_seed(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def signed_distance(mask: np.ndarray, clip: float = 40.0) -> np.ndarray:
    mask = mask.astype(bool)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    sdf = inside - outside
    sdf = np.clip(sdf, -clip, clip) / clip
    return sdf.astype(np.float32)


def resize_float(arr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(arr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def make_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)


def warp_tensor(image: torch.Tensor, flow_norm: torch.Tensor) -> torch.Tensor:
    b, _, h, w = image.shape
    grid = make_grid(b, h, w, image.device) + flow_norm.permute(0, 2, 3, 1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return dx + dy


def dice_loss_from_soft_mask(warped_mask: torch.Tensor, fixed_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (warped_mask * fixed_mask).sum()
    denom = warped_mask.sum() + fixed_mask.sum()
    return 1.0 - (2.0 * intersection + eps) / (denom + eps)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBottleneck2D(nn.Module):
    def __init__(self, channels: int, feature_size: int, depth: int, heads: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.channels = channels
        self.feature_size = feature_size
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_size * feature_size, channels))
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.feature_size or w != self.feature_size:
            raise ValueError(f"Expected {self.feature_size}x{self.feature_size} bottleneck, got {h}x{w}.")
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TransMorphLite2D(nn.Module):
    def __init__(
        self,
        input_size: int = 192,
        max_disp_norm: float = 0.12,
        embed_dim: int = 64,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
    ):
        super().__init__()
        if input_size % 8 != 0:
            raise ValueError("input_size must be divisible by 8.")
        self.max_disp_norm = max_disp_norm
        self.enc1 = ConvBlock(2, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 48)
        self.enc4 = ConvBlock(48, embed_dim)
        self.transformer = TransformerBottleneck2D(
            channels=embed_dim,
            feature_size=input_size // 8,
            depth=transformer_depth,
            heads=transformer_heads,
        )
        self.dec3 = ConvBlock(embed_dim + 48, 48)
        self.dec2 = ConvBlock(48 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.flow = nn.Conv2d(16, 2, 3, padding=1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([moving, fixed], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.enc4(F.avg_pool2d(e3, 2))
        b = self.transformer(b)
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.tanh(self.flow(d1)) * self.max_disp_norm


def normalized_flow_to_pixels(flow_norm: np.ndarray, height: int, width: int) -> np.ndarray:
    flow = np.zeros_like(flow_norm, dtype=np.float32)
    flow[0] = flow_norm[0] * (width - 1) / 2.0
    flow[1] = flow_norm[1] * (height - 1) / 2.0
    return flow


def warp_image_cv2(image: np.ndarray, flow_px: np.ndarray, interpolation: int) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    map_x = xx + flow_px[0]
    map_y = yy + flow_px[1]
    return cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def flow_metrics(flow_px: np.ndarray) -> dict:
    magnitude = np.sqrt(flow_px[0] ** 2 + flow_px[1] ** 2)
    dx_x = np.gradient(flow_px[0], axis=1)
    dx_y = np.gradient(flow_px[0], axis=0)
    dy_x = np.gradient(flow_px[1], axis=1)
    dy_y = np.gradient(flow_px[1], axis=0)
    jacobian_det = (1.0 + dx_x) * (1.0 + dy_y) - dx_y * dy_x
    return {
        "flow_mean_px": float(np.mean(magnitude)),
        "flow_p50_px": float(np.percentile(magnitude, 50)),
        "flow_p95_px": float(np.percentile(magnitude, 95)),
        "flow_max_px": float(np.max(magnitude)),
        "flow_smoothness_px": float(np.mean(np.abs(dx_x)) + np.mean(np.abs(dx_y)) + np.mean(np.abs(dy_x)) + np.mean(np.abs(dy_y))),
        "jacobian_det_min": float(np.min(jacobian_det)),
        "jacobian_det_p01": float(np.percentile(jacobian_det, 1)),
        "jacobian_det_negative_fraction": float(np.mean(jacobian_det <= 0)),
    }


def contour_overlay(fixed_rgb: np.ndarray, moving_rgb: np.ndarray, fixed_mask: np.ndarray, moving_mask: np.ndarray) -> np.ndarray:
    canvas = cv2.addWeighted(fixed_rgb, 0.65, moving_rgb, 0.35, 0)
    fixed_contours, _ = cv2.findContours(fixed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_contours, _ = cv2.findContours(moving_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, fixed_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(canvas, moving_contours, -1, (255, 0, 255), 2)
    return canvas


def load_subject(subject: str) -> dict:
    image_dir = DL_ROOT / "Imagenes_a_escala" / subject
    affine_dir = BASELINE_DIR / "outputs_registro_afin_mascaras" / subject
    rigid_dir = BASELINE_DIR / "outputs_registro_rigido_mascaras" / subject

    affine_metrics_path = affine_dir / f"{subject}_affine_metrics.json"
    affine_accepted = False
    if affine_metrics_path.exists():
        with affine_metrics_path.open(encoding="utf-8") as f:
            affine_accepted = bool(json.load(f).get("affine_accepted", False))

    if affine_accepted:
        moving_image_path = affine_dir / f"{subject}_he_affine_to_hsi.png"
        moving_mask_path = affine_dir / f"{subject}_he_mask_affine_to_hsi.png"
        init_stage = "affine"
    else:
        moving_image_path = rigid_dir / f"{subject}_he_rigid_to_hsi.png"
        moving_mask_path = rigid_dir / f"{subject}_he_mask_rigid_to_hsi.png"
        init_stage = "rigid"

    fixed_image_path = image_dir / f"{subject}_hsi.png"
    fixed_mask_path = image_dir / f"{subject}_hsi_mask.png"

    return {
        "moving_rgb": read_rgb(moving_image_path),
        "moving_mask": read_mask(moving_mask_path),
        "fixed_rgb": read_rgb(fixed_image_path),
        "fixed_mask": read_mask(fixed_mask_path),
        "moving_image_path": moving_image_path,
        "moving_mask_path": moving_mask_path,
        "fixed_image_path": fixed_image_path,
        "fixed_mask_path": fixed_mask_path,
        "init_stage": init_stage,
    }


def run(
    subject: str,
    iterations: int,
    size: int,
    lr: float,
    smooth_weight: float,
    dice_weight: float,
    max_disp_norm: float,
    transformer_depth: int,
    transformer_heads: int,
    output_tag: str | None = None,
) -> dict:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_subject(subject)

    moving_mask = data["moving_mask"]
    fixed_mask = data["fixed_mask"]
    if moving_mask.shape != fixed_mask.shape:
        raise ValueError(f"Mask shapes differ after baseline init: {moving_mask.shape} vs {fixed_mask.shape}")

    moving_sdf = signed_distance(moving_mask)
    fixed_sdf = signed_distance(fixed_mask)
    moving_sdf_small = resize_float(moving_sdf, size)
    fixed_sdf_small = resize_float(fixed_sdf, size)
    moving_mask_small = resize_mask(moving_mask, size).astype(np.float32)
    fixed_mask_small = resize_mask(fixed_mask, size).astype(np.float32)

    moving_t = torch.from_numpy(moving_sdf_small)[None, None].to(device)
    fixed_t = torch.from_numpy(fixed_sdf_small)[None, None].to(device)
    moving_mask_t = torch.from_numpy(moving_mask_small)[None, None].to(device)
    fixed_mask_t = torch.from_numpy(fixed_mask_small)[None, None].to(device)

    model = TransMorphLite2D(
        input_size=size,
        max_disp_norm=max_disp_norm,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []

    for step in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        flow = model(moving_t, fixed_t)
        warped_sdf = warp_tensor(moving_t, flow)
        warped_mask = warp_tensor(moving_mask_t, flow)
        image_loss = F.mse_loss(warped_sdf, fixed_t)
        overlap_loss = dice_loss_from_soft_mask(torch.clamp(warped_mask, 0, 1), fixed_mask_t)
        reg_loss = smoothness_loss(flow)
        loss = image_loss + dice_weight * overlap_loss + smooth_weight * reg_loss
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == iterations:
            with torch.no_grad():
                small_mask = (torch.clamp(warped_mask, 0, 1)[0, 0].detach().cpu().numpy() > 0.5)
                d = dice(small_mask, fixed_mask_small > 0.5)
            history.append(
                {
                    "iteration": step,
                    "loss": float(loss.detach().cpu()),
                    "sdf_mse": float(image_loss.detach().cpu()),
                    "dice_loss": float(overlap_loss.detach().cpu()),
                    "smoothness": float(reg_loss.detach().cpu()),
                    "small_dice": d,
                }
            )
            print(history[-1])

    with torch.no_grad():
        flow_small = model(moving_t, fixed_t)[0].detach().cpu().numpy()

    h, w = moving_mask.shape
    flow_px_small = normalized_flow_to_pixels(flow_small, size, size)
    flow_px = np.stack(
        [
            cv2.resize(flow_px_small[0], (w, h), interpolation=cv2.INTER_LINEAR) * (w / size),
            cv2.resize(flow_px_small[1], (w, h), interpolation=cv2.INTER_LINEAR) * (h / size),
        ],
        axis=0,
    ).astype(np.float32)

    warped_rgb = warp_image_cv2(data["moving_rgb"], flow_px, cv2.INTER_LINEAR)
    warped_mask = warp_image_cv2(data["moving_mask"].astype(np.uint8) * 255, flow_px, cv2.INTER_NEAREST) > 127
    initial_dice = dice(data["moving_mask"], data["fixed_mask"])
    final_dice = dice(warped_mask, data["fixed_mask"])
    deformation_metrics = flow_metrics(flow_px)

    suffix = f"_{output_tag}" if output_tag else ""
    output_dir = METHOD_DIR / "outputs" / f"outputs_transmorph_instance{suffix}_{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(warped_rgb).save(output_dir / f"{subject}_he_transmorph_instance_to_hsi.png")
    Image.fromarray((warped_mask.astype(np.uint8) * 255)).save(output_dir / f"{subject}_he_mask_transmorph_instance_to_hsi.png")
    np.savez_compressed(output_dir / f"{subject}_transmorph_instance_flow_px.npz", flow_px=flow_px)

    overlay = contour_overlay(data["fixed_rgb"], warped_rgb, data["fixed_mask"], warped_mask)
    Image.fromarray(overlay).save(output_dir / f"{subject}_overlay_contours_transmorph_instance.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(contour_overlay(data["fixed_rgb"], data["moving_rgb"], data["fixed_mask"], data["moving_mask"]))
    axes[0].set_title(f"Initial {data['init_stage']} | Dice={initial_dice:.3f}")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"TransMorph-style | Dice={final_dice:.3f}")
    axes[1].axis("off")
    axes[2].plot([h["iteration"] for h in history], [h["small_dice"] for h in history])
    axes[2].set_title("Training small-mask Dice")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("Dice")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject}_transmorph_instance_summary.png", dpi=180)
    plt.close(fig)

    summary = {
        "subject": subject,
        "method": "TransMorph-style instance-specific 2D hybrid CNN-Transformer",
        "device": str(device),
        "init_stage": data["init_stage"],
        "input_size": size,
        "iterations": iterations,
        "learning_rate": lr,
        "smooth_weight": smooth_weight,
        "dice_weight": dice_weight,
        "max_disp_norm": max_disp_norm,
        "transformer_depth": transformer_depth,
        "transformer_heads": transformer_heads,
        "output_tag": output_tag,
        "initial_dice": initial_dice,
        "final_dice": final_dice,
        **deformation_metrics,
        "moving_image": str(data["moving_image_path"]),
        "moving_mask": str(data["moving_mask_path"]),
        "fixed_image": str(data["fixed_image_path"]),
        "fixed_mask": str(data["fixed_mask_path"]),
        "history": history,
    }
    with open(output_dir / f"{subject}_transmorph_instance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model, optimizer, moving_t, fixed_t, moving_mask_t, fixed_mask_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instance-specific TransMorph-style 2D registration.")
    parser.add_argument("--subject", default="SB013")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--smooth-weight", type=float, default=0.60)
    parser.add_argument("--dice-weight", type=float, default=0.45)
    parser.add_argument("--max-disp-norm", type=float, default=0.10)
    parser.add_argument("--transformer-depth", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(
        subject=args.subject,
        iterations=args.iterations,
        size=args.size,
        lr=args.lr,
        smooth_weight=args.smooth_weight,
        dice_weight=args.dice_weight,
        max_disp_norm=args.max_disp_norm,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        output_tag=args.output_tag,
    )
    print(json.dumps(result, indent=2))
