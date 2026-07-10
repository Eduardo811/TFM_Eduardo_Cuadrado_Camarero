from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path.cwd()
METHOD_DIR = ROOT / "Registration" / "DeepLearning" / "Tecnicas_registration" / "01_voxelmorph"
OUTPUTS_DIR = METHOD_DIR / "outputs"
GRID_DIR = OUTPUTS_DIR / "outputs_voxelmorph_regularization_grid"
QUALITY_DIR = OUTPUTS_DIR / "outputs_voxelmorph_quality_review"
FINAL_DIR = OUTPUTS_DIR / "outputs_voxelmorph_final_block"
SELECTED_CONFIG = "reg_s050_d012"
SUBJECTS = ["SB012", "SB013", "SB017", "SB018", "SB019", "SB020"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def status_label(status: str) -> str:
    return {
        "acceptable_exploratory": "acceptable exploratory",
        "needs_review": "needs visual review",
        "reject_as_final": "reject as final",
        "needs_regularized_rerun": "needs regularized rerun",
    }.get(status, status)


def create_method_barplot(rows: list[dict]) -> None:
    subjects = [row["subject"] for row in rows]
    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, [as_float(row, "initial_dice") for row in rows], width, label="Affine input")
    ax.bar(x, [as_float(row, "baseline_dice") for row in rows], width, label="Classical baseline")
    ax.bar(x + width, [as_float(row, "voxelmorph_dice") for row in rows], width, label="VoxelMorph selected")

    for idx, row in enumerate(rows):
        gain = as_float(row, "voxelmorph_minus_baseline")
        ax.text(idx + width, as_float(row, "voxelmorph_dice") + 0.015, f"{gain:+.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mask Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_title("Selected VoxelMorph-style registration vs baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "voxelmorph_selected_dice_comparison.png", dpi=180)
    plt.close(fig)


def create_quality_plot(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "acceptable_exploratory": "#2ca02c",
        "needs_review": "#ff7f0e",
        "reject_as_final": "#d62728",
    }

    for row in rows:
        status = row["quality_status"]
        ax.scatter(
            as_float(row, "flow_p95_px"),
            as_float(row, "voxelmorph_minus_baseline"),
            s=95,
            color=colors.get(status, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.5,
            label=status_label(status),
        )
        ax.text(as_float(row, "flow_p95_px") + 0.8, as_float(row, "voxelmorph_minus_baseline"), row["subject"], fontsize=9, va="center")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("P95 displacement (px)")
    ax.set_ylabel("Dice gain vs classical baseline")
    ax.set_title("Overlap gain vs deformation magnitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "voxelmorph_selected_quality_scatter.png", dpi=180)
    plt.close(fig)


def create_summary_montage(rows: list[dict]) -> None:
    thumbs = []
    for row in rows:
        subject = row["subject"]
        src = OUTPUTS_DIR / f"outputs_voxelmorph_instance_{SELECTED_CONFIG}_{subject}" / f"{subject}_voxelmorph_instance_summary.png"
        if not src.exists():
            continue
        image = Image.open(src).convert("RGB")
        image.thumbnail((720, 390))
        canvas = Image.new("RGB", (760, 445), "white")
        canvas.paste(image, ((760 - image.width) // 2, 35))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{subject} | Dice {float(row['baseline_dice']):.3f} -> {float(row['voxelmorph_dice']):.3f} "
            f"({float(row['voxelmorph_minus_baseline']):+.3f}) | {status_label(row['quality_status'])}"
        )
        draw.text((18, 12), title, fill="black")
        thumbs.append(canvas)

    if not thumbs:
        return

    cols = 2
    rows_n = int(np.ceil(len(thumbs) / cols))
    montage = Image.new("RGB", (cols * 760, rows_n * 445), "white")
    for idx, thumb in enumerate(thumbs):
        montage.paste(thumb, ((idx % cols) * 760, (idx // cols) * 445))
    montage.save(FINAL_DIR / "voxelmorph_selected_summary_montage.png")


def build_final_rows() -> list[dict]:
    rows = [row for row in read_csv(GRID_DIR / "regularized_voxelmorph_summary.csv") if row["config_tag"] == SELECTED_CONFIG]
    rows = sorted(rows, key=lambda row: SUBJECTS.index(row["subject"]))
    final_rows = []
    for row in rows:
        final_rows.append(
            {
                "subject": row["subject"],
                "selected_config": row["config_tag"],
                "initial_stage": row["initial_stage"],
                "initial_dice": f"{float(row['initial_dice']):.6f}",
                "classical_baseline_dice": f"{float(row['baseline_dice']):.6f}",
                "voxelmorph_dice": f"{float(row['voxelmorph_dice']):.6f}",
                "voxelmorph_minus_baseline": f"{float(row['voxelmorph_minus_baseline']):+.6f}",
                "flow_p95_px": f"{float(row['flow_p95_px']):.3f}",
                "negative_jacobian_fraction": f"{float(row['jacobian_det_negative_fraction']):.6f}",
                "quality_status": row["quality_status"],
                "interpretation": status_label(row["quality_status"]),
            }
        )
    return final_rows


def create_readme(rows: list[dict], config_rows: list[dict]) -> None:
    selected = next(row for row in config_rows if row["config_tag"] == SELECTED_CONFIG)
    mean_baseline = np.mean([float(row["classical_baseline_dice"]) for row in rows])
    mean_vm = np.mean([float(row["voxelmorph_dice"]) for row in rows])
    acceptable = sum(row["quality_status"] == "acceptable_exploratory" for row in rows)
    review = sum(row["quality_status"] == "needs_review" for row in rows)
    reject = sum(row["quality_status"] == "reject_as_final" for row in rows)

    lines = [
        "# VoxelMorph final deep-learning block",
        "",
        "This folder closes the VoxelMorph-style experiments as the main deep-learning registration block for the TFM.",
        "",
        "## What was tested",
        "",
        "A 2D VoxelMorph-style CNN was used as an instance-specific deformable registration method. The model was not trained as a universal model across subjects because only six specimen pairs are available. Instead, one small network is optimized for each H&E-HSI pair.",
        "",
        "The network receives signed distance maps from the H&E and HSI masks and predicts a dense deformation field. The loss combines mask-distance similarity, mask overlap and deformation smoothness.",
        "",
        "## Selected configuration",
        "",
        f"- Selected config: `{SELECTED_CONFIG}`",
        "- Smooth weight: 0.50",
        "- Max displacement norm: 0.12",
        f"- Mean classical baseline Dice: {mean_baseline:.3f}",
        f"- Mean selected VoxelMorph Dice: {mean_vm:.3f}",
        f"- Mean gain vs classical baseline: {mean_vm - mean_baseline:+.3f}",
        f"- Mean P95 displacement: {float(selected['mean_flow_p95_px']):.1f} px",
        f"- Mean negative Jacobian fraction: {float(selected['mean_negative_jacobian_fraction']):.3f}",
        f"- Quality status count: {acceptable} acceptable, {review} needs review, {reject} reject as final",
        "",
        "This configuration was selected because it provides a better compromise than the first aggressive VoxelMorph run: it preserves a positive mean Dice gain while reducing displacement magnitude and deformation-risk indicators.",
        "",
        "## Case-level summary",
        "",
        "| Subject | Baseline Dice | VoxelMorph Dice | Gain | P95 flow px | Neg. Jacobian frac. | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subject']} | {float(row['classical_baseline_dice']):.3f} | {float(row['voxelmorph_dice']):.3f} | "
            f"{float(row['voxelmorph_minus_baseline']):+.3f} | {float(row['flow_p95_px']):.1f} | "
            f"{float(row['negative_jacobian_fraction']):.3f} | {status_label(row['quality_status'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The VoxelMorph-style approach is the most relevant deep-learning experiment developed in this project. It shows that a neural deformation model can improve overlap in several cases, especially when used as an instance-specific refinement after the classical alignment.",
            "",
            "However, the result should not be presented as a fully validated final pipeline. Some cases still require visual review, and SB017 remains problematic because the gain in Dice is associated with a deformation that is too aggressive to accept as final.",
            "",
            "The defensible conclusion is: VoxelMorph-style registration is promising for HSI-H&E refinement, but with the current dataset size it should be treated as a proof of concept rather than a generalizable trained model.",
            "",
            "## Files in this folder",
            "",
            "- `voxelmorph_final_selected_summary.csv`: clean selected case-level table.",
            "- `voxelmorph_final_selected_summary.json`: same information in JSON format.",
            "- `voxelmorph_selected_dice_comparison.png`: Dice comparison against the affine input and classical baseline.",
            "- `voxelmorph_selected_quality_scatter.png`: Dice gain vs deformation magnitude.",
            "- `voxelmorph_selected_summary_montage.png`: visual summary of the selected configuration.",
            "- `regularization_tradeoff.png`: trade-off across all tested regularization settings.",
            "- `regularization_overlay_grid.png`: visual grid from the regularization experiment.",
            "- `selected_subject_outputs/`: copied per-subject overlays, warped H&E images and masks for the selected configuration.",
        ]
    )
    (FINAL_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def copy_selected_outputs(rows: list[dict]) -> None:
    for row in rows:
        subject = row["subject"]
        src_dir = OUTPUTS_DIR / f"outputs_voxelmorph_instance_{SELECTED_CONFIG}_{subject}"
        dst_dir = FINAL_DIR / "selected_subject_outputs" / subject
        copy_if_exists(src_dir / f"{subject}_he_voxelmorph_instance_to_hsi.png", dst_dir / f"{subject}_he_voxelmorph_selected_to_hsi.png")
        copy_if_exists(src_dir / f"{subject}_he_mask_voxelmorph_instance_to_hsi.png", dst_dir / f"{subject}_he_mask_voxelmorph_selected_to_hsi.png")
        copy_if_exists(src_dir / f"{subject}_overlay_contours_voxelmorph_instance.png", dst_dir / f"{subject}_overlay_contours_voxelmorph_selected.png")
        copy_if_exists(src_dir / f"{subject}_voxelmorph_instance_summary.png", dst_dir / f"{subject}_voxelmorph_selected_summary.png")
        copy_if_exists(src_dir / f"{subject}_voxelmorph_instance_summary.json", dst_dir / f"{subject}_voxelmorph_selected_summary.json")


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = build_final_rows()
    config_rows = read_csv(GRID_DIR / "regularized_voxelmorph_config_summary.csv")

    write_csv(final_rows, FINAL_DIR / "voxelmorph_final_selected_summary.csv")
    with open(FINAL_DIR / "voxelmorph_final_selected_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2)

    copy_if_exists(GRID_DIR / "regularization_tradeoff.png", FINAL_DIR / "regularization_tradeoff.png")
    copy_if_exists(GRID_DIR / "regularization_overlay_grid.png", FINAL_DIR / "regularization_overlay_grid.png")
    copy_if_exists(QUALITY_DIR / "voxelmorph_quality_review_scatter.png", FINAL_DIR / "initial_aggressive_quality_review_scatter.png")

    raw_rows = [row for row in read_csv(GRID_DIR / "regularized_voxelmorph_summary.csv") if row["config_tag"] == SELECTED_CONFIG]
    raw_rows = sorted(raw_rows, key=lambda row: SUBJECTS.index(row["subject"]))
    create_method_barplot(raw_rows)
    create_quality_plot(raw_rows)
    create_summary_montage(raw_rows)
    copy_selected_outputs(raw_rows)
    create_readme(final_rows, config_rows)

    print(f"Created final VoxelMorph block at: {FINAL_DIR}")


if __name__ == "__main__":
    main()
