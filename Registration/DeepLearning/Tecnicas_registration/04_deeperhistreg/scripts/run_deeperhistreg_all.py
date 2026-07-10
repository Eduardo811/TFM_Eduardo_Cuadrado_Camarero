from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import deeperhistreg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path.cwd()
DL_ROOT = ROOT / "Registration" / "DeepLearning"
METHOD_DIR = DL_ROOT / "Tecnicas_registration" / "04_deeperhistreg"
INPUT_ROOT = DL_ROOT / "Imagenes_a_escala"
BASELINE_DIR = DL_ROOT / "Tecnicas_registration" / "00_baseline_clasico" / "outputs" / "outputs_baseline_clasico_final"
VOX_REG_DIR = (
    DL_ROOT
    / "Tecnicas_registration"
    / "01_voxelmorph"
    / "outputs"
    / "outputs_voxelmorph_regularization_grid"
)
OUTPUT_ROOT = METHOD_DIR / "outputs" / "outputs_deeperhistreg_all"
SUBJECTS = ["SB012", "SB013", "SB017", "SB018", "SB019", "SB020"]


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    canvas = cv2.addWeighted(fixed_rgb, 0.65, moving_rgb, 0.35, 0)
    fixed_contours, _ = cv2.findContours(fixed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_contours, _ = cv2.findContours(moving_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, fixed_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(canvas, moving_contours, -1, (255, 0, 255), 2)
    return canvas


def prepare_inputs(subject: str, output_dir: Path) -> tuple[Path, Path]:
    input_dir = INPUT_ROOT / subject
    prep_dir = output_dir / "prepared_inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)

    he = read_rgb(input_dir / f"{subject}_h&e.png")
    hsi = read_rgb(input_dir / f"{subject}_hsi.png")
    he_mask = keep_largest_component(read_mask(input_dir / f"{subject}_he_mask.png"))
    hsi_mask = keep_largest_component(read_mask(input_dir / f"{subject}_hsi_mask.png"))

    he_prepared = masked_rgb(he, he_mask, dilate_px=2)
    hsi_prepared = masked_rgb(hsi, hsi_mask, dilate_px=2)

    source_path = prep_dir / f"{subject}_source_he_masked.png"
    target_path = prep_dir / f"{subject}_target_hsi_masked.png"
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
    fig.savefig(output_dir / f"{subject}_prepared_inputs.png", dpi=160)
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


def find_warped_source(subject: str, output_dir: Path, temp_dir: Path) -> Path:
    matches = sorted(output_dir.glob("warped_source*"))
    if not matches:
        matches = sorted((temp_dir / f"{subject}_deeperhistreg" / "Nonrigid_Registration").glob("warped_source*"))
    if not matches:
        matches = sorted((temp_dir / f"{subject}_deeperhistreg" / "Initial_Registration").glob("warped_source*"))
    if not matches:
        raise FileNotFoundError(f"No warped_source found for {subject} in {output_dir}")
    copied = output_dir / f"{subject}_warped_source_deeperhistreg.png"
    shutil.copy(matches[0], copied)
    return copied


def summarize_result(subject: str, output_dir: Path, warped_path: Path) -> dict:
    input_dir = INPUT_ROOT / subject
    fixed = read_rgb(input_dir / f"{subject}_hsi.png")
    fixed_mask = keep_largest_component(read_mask(input_dir / f"{subject}_hsi_mask.png"))
    warped = read_rgb(warped_path)
    warped_mask = mask_from_black_background(warped)

    if warped.shape[:2] != fixed.shape[:2]:
        warped = cv2.resize(warped, (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.resize(warped_mask.astype(np.uint8), (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    score = dice(warped_mask, fixed_mask)
    overlay = contour_overlay(fixed, warped, fixed_mask, warped_mask)
    overlay_path = output_dir / f"{subject}_deeperhistreg_overlay_contours.png"
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
    comparison_path = output_dir / f"{subject}_deeperhistreg_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    baseline = load_json(BASELINE_DIR / subject / f"{subject}_baseline_summary.json")
    baseline_dice = float(baseline["dice_final"])

    summary = {
        "subject": subject,
        "method": "DeeperHistReg exploratory",
        "source": str(input_dir / f"{subject}_h&e.png"),
        "target": str(input_dir / f"{subject}_hsi.png"),
        "warped_source": str(warped_path),
        "overlay_contours": str(overlay_path),
        "comparison": str(comparison_path),
        "baseline_dice": baseline_dice,
        "deeperhistreg_dice_mask_from_black_background": score,
        "deeperhistreg_minus_baseline": score - baseline_dice,
        "note": "Dice is only a rough sanity metric because the warped H&E mask is re-estimated from black background.",
    }
    with open(output_dir / f"{subject}_deeperhistreg_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_subject(subject: str, force: bool = False) -> dict:
    output_dir = OUTPUT_ROOT / subject
    temp_dir = output_dir / "temporary"
    summary_path = output_dir / f"{subject}_deeperhistreg_summary.json"
    if summary_path.exists() and not force:
        print(f"loading existing result: {summary_path}")
        return load_json(summary_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path, target_path = prepare_inputs(subject, output_dir)
    params = build_params()
    with open(output_dir / f"{subject}_deeperhistreg_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    try:
        deeperhistreg.run_registration(
            source_path=str(source_path),
            target_path=str(target_path),
            output_path=str(output_dir),
            case_name=f"{subject}_deeperhistreg",
            temporary_path=str(temp_dir),
            registration_parameters=params,
            save_displacement_field=False,
            copy_target=True,
            delete_temporary_results=False,
        )
    except FileNotFoundError:
        print(f"{subject}: DeeperHistReg did not create Results_Final; using temporary output.")

    warped_path = find_warped_source(subject, output_dir, temp_dir)
    return summarize_result(subject, output_dir, warped_path)


def load_voxelmorph_best_rows() -> dict[str, dict]:
    csv_path = VOX_REG_DIR / "regularized_voxelmorph_summary.csv"
    if not csv_path.exists():
        return {}
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["config_tag"] == "reg_s050_d012":
                rows[row["subject"]] = row
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_overlay_grid(rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(rows), 1, figsize=(7, 4.8 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        image = Image.open(row["overlay_contours"])
        ax.imshow(image)
        ax.set_title(
            f"{row['subject']} | DeeperHistReg Dice={row['deeperhistreg_dice_mask_from_black_background']:.3f} "
            f"| baseline={row['baseline_dice']:.3f}"
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "deeperhistreg_overlay_grid.png", dpi=180)
    plt.close(fig)


def create_method_comparison(rows: list[dict]) -> None:
    vox_rows = load_voxelmorph_best_rows()
    subjects = [r["subject"] for r in rows]
    x = np.arange(len(subjects))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [r["baseline_dice"] for r in rows], width, label="Classical baseline")
    ax.bar(x, [float(vox_rows.get(s, {}).get("voxelmorph_dice", np.nan)) for s in subjects], width, label="VoxelMorph reg_s050_d012")
    ax.bar(x + width, [r["deeperhistreg_dice_mask_from_black_background"] for r in rows], width, label="DeeperHistReg")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mask Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Method comparison using mask Dice")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "method_dice_comparison.png", dpi=180)
    plt.close(fig)


def create_report(rows: list[dict]) -> None:
    vox_rows = load_voxelmorph_best_rows()
    mean_deeper = float(np.mean([r["deeperhistreg_dice_mask_from_black_background"] for r in rows]))
    mean_baseline = float(np.mean([r["baseline_dice"] for r in rows]))
    lines = [
        "# DeeperHistReg exploratory evaluation",
        "",
        "This experiment runs DeeperHistReg on the six prepared HSI-H&E pairs. It should be interpreted as an external feature/non-rigid registration baseline, not as a method designed specifically for HSI-H&E.",
        "",
        "Important caveat: the DeeperHistReg Dice is approximate because the warped H&E mask is re-estimated from the black background of the warped source image.",
        "",
        "## Summary",
        "",
        f"- Mean classical baseline Dice: {mean_baseline:.3f}",
        f"- Mean DeeperHistReg Dice: {mean_deeper:.3f}",
        f"- Mean DeeperHistReg gain vs baseline: {mean_deeper - mean_baseline:+.3f}",
        "",
        "## Case-level comparison",
        "",
        "| Subject | Classical baseline | VoxelMorph reg_s050_d012 | DeeperHistReg | DeeperHistReg - baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        vox = vox_rows.get(row["subject"], {})
        vox_dice = float(vox["voxelmorph_dice"]) if vox else float("nan")
        lines.append(
            f"| {row['subject']} | {row['baseline_dice']:.3f} | {vox_dice:.3f} | "
            f"{row['deeperhistreg_dice_mask_from_black_background']:.3f} | {row['deeperhistreg_minus_baseline']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- DeeperHistReg provides a useful external comparison, but it is not directly competitive with the current classical baseline on these HSI-H&E pairs.",
            "- The method is designed for histology-to-histology registration; HSI pseudo-RGB and H&E have a much larger modality gap.",
            "- The result supports keeping DeeperHistReg as an exploratory baseline/future-work direction rather than as the main registration pipeline.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for subject in SUBJECTS:
        print(f"=== {subject} ===")
        summaries.append(run_subject(subject))

    rows = [
        {
            "subject": s["subject"],
            "baseline_dice": s["baseline_dice"],
            "deeperhistreg_dice_mask_from_black_background": s["deeperhistreg_dice_mask_from_black_background"],
            "deeperhistreg_minus_baseline": s["deeperhistreg_minus_baseline"],
            "warped_source": s["warped_source"],
            "overlay_contours": s["overlay_contours"],
            "comparison": s["comparison"],
        }
        for s in summaries
    ]
    write_csv(rows, OUTPUT_ROOT / "deeperhistreg_all_summary.csv")
    with open(OUTPUT_ROOT / "deeperhistreg_all_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    create_overlay_grid(rows)
    create_method_comparison(rows)
    create_report(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
