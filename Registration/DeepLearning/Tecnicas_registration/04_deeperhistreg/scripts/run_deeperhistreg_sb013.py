from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import deeperhistreg


ROOT = Path.cwd()
DL_ROOT = ROOT / "Registration" / "DeepLearning"
SUBJECT = "SB013"
INPUT_DIR = DL_ROOT / "Imagenes_a_escala" / SUBJECT
METHOD_DIR = DL_ROOT / "Tecnicas_registration" / "04_deeperhistreg"
OUTPUT_DIR = METHOD_DIR / "outputs" / f"outputs_deeperhistreg_{SUBJECT}"
PREP_DIR = OUTPUT_DIR / "prepared_inputs"
TEMP_DIR = OUTPUT_DIR / "temporary"


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def masked_rgb(image: np.ndarray, mask: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out


def mask_from_black_background(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 8
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return keep_largest_component(mask > 0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def contour_overlay(fixed_rgb: np.ndarray, moving_rgb: np.ndarray, fixed_mask: np.ndarray, moving_mask: np.ndarray) -> np.ndarray:
    canvas = fixed_rgb.copy()
    canvas = cv2.addWeighted(canvas, 0.65, moving_rgb, 0.35, 0)
    fixed_contours, _ = cv2.findContours(fixed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_contours, _ = cv2.findContours(moving_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, fixed_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(canvas, moving_contours, -1, (255, 0, 255), 2)
    return canvas


def prepare_inputs() -> tuple[Path, Path]:
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    he = read_rgb(INPUT_DIR / f"{SUBJECT}_h&e.png")
    hsi = read_rgb(INPUT_DIR / f"{SUBJECT}_hsi.png")
    he_mask = keep_largest_component(read_mask(INPUT_DIR / f"{SUBJECT}_he_mask.png"))
    hsi_mask = keep_largest_component(read_mask(INPUT_DIR / f"{SUBJECT}_hsi_mask.png"))

    he_prepared = masked_rgb(he, he_mask, dilate_px=2)
    hsi_prepared = masked_rgb(hsi, hsi_mask, dilate_px=2)

    source_path = PREP_DIR / f"{SUBJECT}_source_he_masked.png"
    target_path = PREP_DIR / f"{SUBJECT}_target_hsi_masked.png"
    Image.fromarray(he_prepared).save(source_path)
    Image.fromarray(hsi_prepared).save(target_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(he_prepared)
    axes[0].set_title("Source H&E masked")
    axes[0].axis("off")
    axes[1].imshow(hsi_prepared)
    axes[1].set_title("Target HSI masked")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{SUBJECT}_prepared_inputs.png", dpi=160)
    plt.close(fig)
    return source_path, target_path


def build_params() -> dict:
    params = deeperhistreg.configs.default_nonrigid_fast()
    params["device"] = "cpu"
    params["echo"] = True
    params["save_final_displacement_field"] = False

    params["loading_params"]["loader"] = "pil"
    params["loading_params"]["pad_value"] = 0.0
    params["loading_params"]["source_resample_ratio"] = 1.0
    params["loading_params"]["target_resample_ratio"] = 1.0

    params["saving_params"]["saver"] = "pil"
    params["saving_params"]["final_saver"] = "pil"
    params["saving_params"]["save_params"] = "pil"
    params["saving_params"]["extension"] = ".png"
    params["saving_params"]["final_extension"] = ".png"

    params["preprocessing_params"]["initial_resolution"] = 512
    params["preprocessing_params"]["normalization"] = True
    params["preprocessing_params"]["convert_to_gray"] = True
    params["preprocessing_params"]["clahe"] = True
    params["preprocessing_params"]["flip_intensity"] = False

    params["run_initial_registration"] = True
    params["initial_registration_params"] = {
        "save_results": True,
        "initial_registration_function": "multi_feature",
        "registration_size": 512,
        "registration_sizes": [220, 320, 420, 512],
        "transform_type": "rigid",
        "keypoint_threshold": 0.005,
        "match_threshold": 0.3,
        "sinkhorn_iterations": 50,
        "show": False,
        "angle_step": 90,
        "cuda": False,
        "device": "cpu",
        "echo": True,
        "run_sift_ransac": True,
        "run_superpoint_superglue": False,
        "run_superpoint_ransac": False,
    }

    params["run_nonrigid_registration"] = True
    params["nonrigid_registration_params"]["device"] = "cpu"
    params["nonrigid_registration_params"]["echo"] = True
    params["nonrigid_registration_params"]["registration_size"] = 512
    params["nonrigid_registration_params"]["num_levels"] = 4
    params["nonrigid_registration_params"]["used_levels"] = 4
    params["nonrigid_registration_params"]["iterations"] = [30, 30, 40, 50]
    params["nonrigid_registration_params"]["learning_rates"] = [0.004, 0.003, 0.002, 0.001]
    params["nonrigid_registration_params"]["alphas"] = [2.0, 2.0, 2.2, 2.5]
    return params


def find_warped_source() -> Path:
    matches = sorted(OUTPUT_DIR.glob("warped_source*"))
    if not matches:
        matches = sorted((TEMP_DIR / f"{SUBJECT}_deeperhistreg" / "Nonrigid_Registration").glob("warped_source*"))
    if not matches:
        matches = sorted((TEMP_DIR / f"{SUBJECT}_deeperhistreg" / "Initial_Registration").glob("warped_source*"))
    if not matches:
        raise FileNotFoundError(f"No warped_source found in {OUTPUT_DIR}")
    copied = OUTPUT_DIR / f"{SUBJECT}_warped_source_deeperhistreg.png"
    shutil.copy(matches[0], copied)
    return copied


def summarize_result(warped_path: Path) -> dict:
    fixed = read_rgb(INPUT_DIR / f"{SUBJECT}_hsi.png")
    fixed_mask = keep_largest_component(read_mask(INPUT_DIR / f"{SUBJECT}_hsi_mask.png"))
    warped = read_rgb(warped_path)
    warped_mask = mask_from_black_background(warped)

    if warped.shape[:2] != fixed.shape[:2]:
        warped = cv2.resize(warped, (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.resize(warped_mask.astype(np.uint8), (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    score = dice(warped_mask, fixed_mask)
    overlay = contour_overlay(fixed, warped, fixed_mask, warped_mask)
    overlay_path = OUTPUT_DIR / f"{SUBJECT}_deeperhistreg_overlay_contours.png"
    Image.fromarray(overlay).save(overlay_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(fixed)
    axes[0].set_title("Fixed HSI")
    axes[0].axis("off")
    axes[1].imshow(warped)
    axes[1].set_title("Warped H&E by DeeperHistReg")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Contours | Dice mask={score:.3f}")
    axes[2].axis("off")
    fig.tight_layout()
    comparison_path = OUTPUT_DIR / f"{SUBJECT}_deeperhistreg_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    summary = {
        "subject": SUBJECT,
        "method": "DeeperHistReg exploratory",
        "source": str(INPUT_DIR / f"{SUBJECT}_h&e.png"),
        "target": str(INPUT_DIR / f"{SUBJECT}_hsi.png"),
        "warped_source": str(warped_path),
        "overlay_contours": str(overlay_path),
        "comparison": str(comparison_path),
        "dice_mask_from_black_background": score,
        "note": "Dice is only a rough sanity metric because the warped H&E mask is re-estimated from black background.",
    }
    with open(OUTPUT_DIR / f"{SUBJECT}_deeperhistreg_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path, target_path = prepare_inputs()
    params = build_params()

    with open(OUTPUT_DIR / f"{SUBJECT}_deeperhistreg_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    try:
        deeperhistreg.run_registration(
            source_path=str(source_path),
            target_path=str(target_path),
            output_path=str(OUTPUT_DIR),
            case_name=f"{SUBJECT}_deeperhistreg",
            temporary_path=str(TEMP_DIR),
            registration_parameters=params,
            save_displacement_field=False,
            copy_target=True,
            delete_temporary_results=False,
        )
    except FileNotFoundError:
        print("DeeperHistReg did not create Results_Final; using Nonrigid_Registration temporary output.")

    warped_path = find_warped_source()
    summary = summarize_result(warped_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
