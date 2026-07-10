from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from spectral import open_image


ROOT = Path.cwd()
DL_ROOT = ROOT / "Registration" / "DeepLearning"
METHOD_DIR = DL_ROOT / "Tecnicas_registration" / "05_feature_based_matching"
OUTPUT_DIR = METHOD_DIR / "outputs" / "outputs_loftr_SB013_segmented"
PREPARED_DIR = DL_ROOT / "Imagenes_a_escala" / "SB013"
RAW_HSI_HDR = ROOT / "Datos" / "SB013" / "SB013" / "SB013_001" / "SB013_nrm.hdr"
RAW_HE_PATH = ROOT / "Datos" / "SB013" / "SB013" / "WSI" / "Paraffin.jpg"


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def robust_normalize(channel: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    channel = channel.astype(np.float32)
    lo = float(np.percentile(channel, p_low))
    hi = float(np.percentile(channel, p_high))
    if hi <= lo:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - lo) / (hi - lo), 0, 1)


def load_raw_hsi_pseudo_rgb() -> tuple[np.ndarray, dict]:
    image = open_image(str(RAW_HSI_HDR))
    wavelengths = np.asarray([float(value) for value in image.metadata["wavelength"]], dtype=np.float32)
    targets = {"r": 650.0, "g": 550.0, "b": 450.0}
    indices = {name: int(np.argmin(np.abs(wavelengths - target))) for name, target in targets.items()}
    selected = np.asarray(image.read_bands([indices["r"], indices["g"], indices["b"]]), dtype=np.float32)
    rgb = np.stack([robust_normalize(selected[:, :, index]) for index in range(3)], axis=-1)
    info = {
        "indices": indices,
        "wavelengths_nm": {name: float(wavelengths[index]) for name, index in indices.items()},
    }
    return (rgb * 255).astype(np.uint8), info


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: int = 8) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty mask.")
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + padding)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + padding)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def resize_and_pad(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    max_side: int = 768,
    multiple: int = 8,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    new_width = max(multiple, int(round(width * scale)))
    new_height = max(multiple, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    resized_mask = None
    if mask is not None:
        resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0

    padded_width = int(np.ceil(new_width / multiple) * multiple)
    padded_height = int(np.ceil(new_height / multiple) * multiple)
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded[:new_height, :new_width] = resized
    padded_mask = None
    if resized_mask is not None:
        padded_mask = np.zeros((padded_height, padded_width), dtype=bool)
        padded_mask[:new_height, :new_width] = resized_mask

    info = {
        "original_shape": [height, width],
        "resized_shape": [new_height, new_width],
        "padded_shape": [padded_height, padded_width],
        "scale": float(scale),
    }
    return padded, padded_mask, info


def gray_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return torch.from_numpy(gray)[None, None].to(device)


def run_loftr(matcher: KF.LoFTR, moving: np.ndarray, fixed: np.ndarray, device: torch.device) -> dict:
    with torch.no_grad():
        correspondences = matcher(
            {
                "image0": gray_tensor(moving, device),
                "image1": gray_tensor(fixed, device),
            }
        )
    return {
        "points_moving": correspondences["keypoints0"].detach().cpu().numpy(),
        "points_fixed": correspondences["keypoints1"].detach().cpu().numpy(),
        "confidence": correspondences["confidence"].detach().cpu().numpy(),
    }


def estimate_transforms(points_moving: np.ndarray, points_fixed: np.ndarray) -> dict:
    affine = None
    affine_inliers = None
    homography = None
    homography_inliers = None
    if len(points_moving) >= 3:
        affine, affine_inliers = cv2.estimateAffinePartial2D(
            points_moving,
            points_fixed,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.995,
        )
    if len(points_moving) >= 4:
        homography, homography_inliers = cv2.findHomography(
            points_moving,
            points_fixed,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.995,
        )
    return {
        "affine": affine,
        "affine_inliers": affine_inliers,
        "homography": homography,
        "homography_inliers": homography_inliers,
    }


def inlier_count(inliers: np.ndarray | None) -> int:
    return int(inliers.sum()) if inliers is not None else 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def warp_with_transform(
    moving_image: np.ndarray,
    moving_mask: np.ndarray | None,
    fixed_shape: tuple[int, int],
    transform: np.ndarray | None,
    kind: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if transform is None:
        return None, None
    height, width = fixed_shape
    if kind == "affine":
        warped_image = cv2.warpAffine(moving_image, transform, (width, height), flags=cv2.INTER_LINEAR)
        warped_mask = None if moving_mask is None else cv2.warpAffine(
            moving_mask.astype(np.uint8) * 255, transform, (width, height), flags=cv2.INTER_NEAREST
        ) > 127
    else:
        warped_image = cv2.warpPerspective(moving_image, transform, (width, height), flags=cv2.INTER_LINEAR)
        warped_mask = None if moving_mask is None else cv2.warpPerspective(
            moving_mask.astype(np.uint8) * 255, transform, (width, height), flags=cv2.INTER_NEAREST
        ) > 127
    return warped_image, warped_mask


def draw_matches(
    moving: np.ndarray,
    fixed: np.ndarray,
    points_moving: np.ndarray,
    points_fixed: np.ndarray,
    confidence: np.ndarray,
    inliers: np.ndarray | None = None,
    max_draw: int = 80,
) -> np.ndarray:
    if len(points_moving) == 0:
        indices = np.array([], dtype=int)
    else:
        valid = np.arange(len(points_moving))
        if inliers is not None:
            inlier_indices = np.flatnonzero(inliers.ravel() > 0)
            if len(inlier_indices):
                valid = inlier_indices
        order = valid[np.argsort(confidence[valid])[::-1]]
        indices = order[:max_draw]

    h0, w0 = moving.shape[:2]
    h1, w1 = fixed.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = moving
    canvas[:h1, w0 : w0 + w1] = fixed
    rng = np.random.default_rng(13)
    for index in indices:
        color = tuple(int(value) for value in rng.integers(60, 255, size=3))
        x0, y0 = points_moving[index]
        x1, y1 = points_fixed[index]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + w0, int(round(y1)))
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.line(canvas, p0, p1, color, 1)
    return canvas


def contour_overlay(fixed: np.ndarray, warped: np.ndarray, fixed_mask: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
    canvas = cv2.addWeighted(fixed, 0.60, warped, 0.40, 0)
    fixed_contours, _ = cv2.findContours(fixed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_contours, _ = cv2.findContours(warped_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, fixed_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(canvas, moving_contours, -1, (255, 0, 255), 2)
    return canvas


def prepare_cases() -> tuple[list[dict], dict]:
    raw_hsi, raw_hsi_info = load_raw_hsi_pseudo_rgb()
    raw_he = read_rgb(RAW_HE_PATH)

    he = read_rgb(PREPARED_DIR / "SB013_h&e.png")
    hsi = read_rgb(PREPARED_DIR / "SB013_hsi.png")
    he_mask = read_mask(PREPARED_DIR / "SB013_he_mask.png")
    hsi_mask = read_mask(PREPARED_DIR / "SB013_hsi_mask.png")

    raw_hsi_prepared, _, raw_hsi_resize = resize_and_pad(raw_hsi, max_side=768)
    raw_he_prepared, _, raw_he_resize = resize_and_pad(raw_he, max_side=768)

    hsi_unmasked, hsi_unmasked_mask, hsi_unmasked_resize = resize_and_pad(hsi, hsi_mask, max_side=768)
    he_unmasked, he_unmasked_mask, he_unmasked_resize = resize_and_pad(he, he_mask, max_side=768)

    hsi_segmented = apply_mask(hsi, hsi_mask)
    he_segmented = apply_mask(he, he_mask)
    hsi_crop, hsi_crop_mask, hsi_bbox = crop_to_mask(hsi_segmented, hsi_mask)
    he_crop, he_crop_mask, he_bbox = crop_to_mask(he_segmented, he_mask)
    hsi_segmented_prepared, hsi_segmented_mask, hsi_segmented_resize = resize_and_pad(hsi_crop, hsi_crop_mask, max_side=768)
    he_segmented_prepared, he_segmented_mask, he_segmented_resize = resize_and_pad(he_crop, he_crop_mask, max_side=768)

    cases = [
        {
            "case": "original_full",
            "label": "Original notebook inputs",
            "moving": raw_hsi_prepared,
            "fixed": raw_he_prepared,
            "moving_mask": None,
            "fixed_mask": None,
            "preparation": {"moving_resize": raw_hsi_resize, "fixed_resize": raw_he_resize},
        },
        {
            "case": "scaled_unmasked",
            "label": "Scale-matched images before explicit mask crop",
            "moving": hsi_unmasked,
            "fixed": he_unmasked,
            "moving_mask": hsi_unmasked_mask,
            "fixed_mask": he_unmasked_mask,
            "preparation": {"moving_resize": hsi_unmasked_resize, "fixed_resize": he_unmasked_resize},
        },
        {
            "case": "segmented_cropped",
            "label": "Segmented specimen only",
            "moving": hsi_segmented_prepared,
            "fixed": he_segmented_prepared,
            "moving_mask": hsi_segmented_mask,
            "fixed_mask": he_segmented_mask,
            "preparation": {
                "moving_bbox": hsi_bbox,
                "fixed_bbox": he_bbox,
                "moving_resize": hsi_segmented_resize,
                "fixed_resize": he_segmented_resize,
            },
        },
    ]
    return cases, raw_hsi_info


def save_case_inputs(case: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(case["moving"])
    axes[0].set_title("Moving HSI")
    axes[0].axis("off")
    axes[1].imshow(case["fixed"])
    axes[1].set_title("Fixed H&E")
    axes[1].axis("off")
    fig.suptitle(case["label"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{case['case']}_inputs.png", dpi=180)
    plt.close(fig)


def process_case(case: dict, matcher: KF.LoFTR, device: torch.device) -> dict:
    save_case_inputs(case)
    matches = run_loftr(matcher, case["moving"], case["fixed"], device)
    transforms = estimate_transforms(matches["points_moving"], matches["points_fixed"])

    affine_warped, affine_mask = warp_with_transform(
        case["moving"], case["moving_mask"], case["fixed"].shape[:2], transforms["affine"], "affine"
    )
    homography_warped, homography_mask = warp_with_transform(
        case["moving"], case["moving_mask"], case["fixed"].shape[:2], transforms["homography"], "homography"
    )

    affine_dice = None
    homography_dice = None
    if case["fixed_mask"] is not None and affine_mask is not None:
        affine_dice = dice(affine_mask, case["fixed_mask"])
    if case["fixed_mask"] is not None and homography_mask is not None:
        homography_dice = dice(homography_mask, case["fixed_mask"])

    preferred_inliers = transforms["homography_inliers"] if transforms["homography_inliers"] is not None else transforms["affine_inliers"]
    match_image = draw_matches(
        case["moving"],
        case["fixed"],
        matches["points_moving"],
        matches["points_fixed"],
        matches["confidence"],
        preferred_inliers,
    )
    Image.fromarray(match_image).save(OUTPUT_DIR / f"{case['case']}_loftr_matches.png")

    if affine_warped is not None and affine_mask is not None:
        affine_overlay = contour_overlay(case["fixed"], affine_warped, case["fixed_mask"], affine_mask)
        Image.fromarray(affine_overlay).save(OUTPUT_DIR / f"{case['case']}_affine_overlay.png")
    if homography_warped is not None and homography_mask is not None:
        homography_overlay = contour_overlay(case["fixed"], homography_warped, case["fixed_mask"], homography_mask)
        Image.fromarray(homography_overlay).save(OUTPUT_DIR / f"{case['case']}_homography_overlay.png")

    num_matches = len(matches["points_moving"])
    affine_inliers = inlier_count(transforms["affine_inliers"])
    homography_inliers = inlier_count(transforms["homography_inliers"])
    summary = {
        "case": case["case"],
        "label": case["label"],
        "num_matches": num_matches,
        "mean_confidence": float(np.mean(matches["confidence"])) if num_matches else 0.0,
        "median_confidence": float(np.median(matches["confidence"])) if num_matches else 0.0,
        "high_confidence_matches_ge_0_5": int(np.sum(matches["confidence"] >= 0.5)),
        "affine_available": transforms["affine"] is not None,
        "affine_inliers": affine_inliers,
        "affine_inlier_ratio": affine_inliers / num_matches if num_matches else 0.0,
        "affine_mask_dice": affine_dice,
        "homography_available": transforms["homography"] is not None,
        "homography_inliers": homography_inliers,
        "homography_inlier_ratio": homography_inliers / num_matches if num_matches else 0.0,
        "homography_mask_dice": homography_dice,
        "moving_shape": list(case["moving"].shape),
        "fixed_shape": list(case["fixed"].shape),
        "preparation": case["preparation"],
        "affine_matrix": transforms["affine"].tolist() if transforms["affine"] is not None else None,
        "homography_matrix": transforms["homography"].tolist() if transforms["homography"] is not None else None,
    }
    return summary


def create_summary_figure(summaries: list[dict]) -> None:
    labels = [summary["case"].replace("_", "\n") for summary in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(x, [summary["num_matches"] for summary in summaries], color=["#777777", "#4477aa", "#228833"])
    axes[0].set_title("LoFTR matches")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - 0.18, [summary["affine_inlier_ratio"] for summary in summaries], 0.36, label="Affine")
    axes[1].bar(x + 0.18, [summary["homography_inlier_ratio"] for summary in summaries], 0.36, label="Homography")
    axes[1].set_title("RANSAC inlier ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    valid = [summary for summary in summaries if summary["affine_mask_dice"] is not None]
    valid_x = np.arange(len(valid))
    axes[2].bar(valid_x - 0.18, [summary["affine_mask_dice"] for summary in valid], 0.36, label="Affine")
    axes[2].bar(valid_x + 0.18, [summary["homography_mask_dice"] for summary in valid], 0.36, label="Homography")
    axes[2].set_title("Mask Dice after LoFTR transform")
    axes[2].set_xticks(valid_x)
    axes[2].set_xticklabels([summary["case"].replace("_", "\n") for summary in valid])
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loftr_SB013_comparison_summary.png", dpi=180)
    plt.close(fig)


def write_csv(summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        rows.append(
            {
                key: summary[key]
                for key in [
                    "case",
                    "label",
                    "num_matches",
                    "mean_confidence",
                    "median_confidence",
                    "high_confidence_matches_ge_0_5",
                    "affine_inliers",
                    "affine_inlier_ratio",
                    "affine_mask_dice",
                    "homography_inliers",
                    "homography_inlier_ratio",
                    "homography_mask_dice",
                ]
            }
        )
    with open(OUTPUT_DIR / "loftr_SB013_comparison_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()
    cases, raw_hsi_info = prepare_cases()
    summaries = []
    for case in cases:
        print(f"=== {case['case']} ===")
        summary = process_case(case, matcher, device)
        summaries.append(summary)
        print(json.dumps({key: summary[key] for key in summary if "matrix" not in key and key != "preparation"}, indent=2))

    payload = {
        "subject": "SB013",
        "method": "LoFTR outdoor + RANSAC affine/homography",
        "direction": "HSI moving -> H&E fixed",
        "device": str(device),
        "raw_hsi_pseudo_rgb": raw_hsi_info,
        "summaries": summaries,
        "note": "Raw full-image case has no directly comparable mask Dice. Segmented cases use prepared masks.",
    }
    with open(OUTPUT_DIR / "loftr_SB013_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(summaries)
    create_summary_figure(summaries)
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
