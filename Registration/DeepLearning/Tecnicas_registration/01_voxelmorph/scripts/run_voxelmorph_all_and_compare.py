from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_voxelmorph_instance import METHOD_DIR, ROOT, run


SUBJECTS = ["SB012", "SB013", "SB017", "SB018", "SB019", "SB020"]
BASELINE_FINAL_DIR = (
    ROOT
    / "Registration"
    / "DeepLearning"
    / "Tecnicas_registration"
    / "00_baseline_clasico"
    / "outputs"
    / "outputs_baseline_clasico_final"
)
OUTPUT_DIR = METHOD_DIR / "outputs" / "outputs_voxelmorph_instance_todos"


def as_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_row(subject: str, vox_summary: dict) -> dict:
    baseline_path = BASELINE_FINAL_DIR / subject / f"{subject}_baseline_summary.json"
    baseline = load_json(baseline_path)
    baseline_dice = as_float(baseline.get("dice_final"))
    row = {
        "subject": subject,
        "initial_stage": vox_summary["init_stage"],
        "initial_dice": vox_summary["initial_dice"],
        "voxelmorph_dice": vox_summary["final_dice"],
        "baseline_final_method": baseline.get("final_method"),
        "baseline_final_note": baseline.get("final_note"),
        "baseline_visual_review_flag": baseline.get("visual_review_flag"),
        "baseline_dice": baseline_dice,
        "voxelmorph_minus_initial": vox_summary["final_dice"] - vox_summary["initial_dice"],
        "voxelmorph_minus_baseline": vox_summary["final_dice"] - baseline_dice,
        "flow_mean_px": vox_summary.get("flow_mean_px"),
        "flow_p95_px": vox_summary.get("flow_p95_px"),
        "flow_max_px": vox_summary.get("flow_max_px"),
        "flow_smoothness_px": vox_summary.get("flow_smoothness_px"),
        "jacobian_det_min": vox_summary.get("jacobian_det_min"),
        "jacobian_det_p01": vox_summary.get("jacobian_det_p01"),
        "jacobian_det_negative_fraction": vox_summary.get("jacobian_det_negative_fraction"),
    }
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_dice_plot(rows: list[dict]) -> None:
    subjects = [r["subject"] for r in rows]
    x = np.arange(len(subjects))
    width = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [r["initial_dice"] for r in rows], width, label="Inicial afin/rigido")
    ax.bar(x, [r["baseline_dice"] for r in rows], width, label="Baseline clasico final")
    ax.bar(x + width, [r["voxelmorph_dice"] for r in rows], width, label="VoxelMorph-style")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Dice mascara")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "voxelmorph_vs_baseline_dice.png", dpi=180)
    plt.close(fig)


def create_overlay_grid(rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 4.8 * len(rows)))
    for i, row in enumerate(rows):
        subject = row["subject"]
        baseline_overlay = BASELINE_FINAL_DIR / subject / f"{subject}_baseline_overlay_contours.png"
        vox_overlay = METHOD_DIR / "outputs" / f"outputs_voxelmorph_instance_{subject}" / f"{subject}_overlay_contours_voxelmorph_instance.png"
        axes[i, 0].imshow(Image.open(baseline_overlay))
        axes[i, 0].set_title(f"{subject} baseline | Dice={row['baseline_dice']:.3f}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(Image.open(vox_overlay))
        axes[i, 1].set_title(f"{subject} VoxelMorph-style | Dice={row['voxelmorph_dice']:.3f}")
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "voxelmorph_vs_baseline_overlay_grid.png", dpi=180)
    plt.close(fig)


def create_markdown_report(rows: list[dict]) -> None:
    lines = [
        "# VoxelMorph-style instance-specific vs baseline clasico",
        "",
        "Este experimento entrena una red 2D tipo VoxelMorph por sujeto. No es un modelo generalizable; es un ajuste instance-specific para comprobar si una CNN puede estimar una deformacion densa util para cada pareja H&E-HSI.",
        "",
        "| Sujeto | Dice inicial | Dice baseline | Dice VoxelMorph | VM - baseline | p95 flow px | max flow px | Jacobian negativo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subject']} | {row['initial_dice']:.3f} | {row['baseline_dice']:.3f} | "
            f"{row['voxelmorph_dice']:.3f} | {row['voxelmorph_minus_baseline']:+.3f} | "
            f"{row['flow_p95_px']:.2f} | {row['flow_max_px']:.2f} | {row['jacobian_det_negative_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretacion",
            "",
            "- `Dice inicial` es el solape antes de la CNN, normalmente tras el registro afin/rigido del baseline.",
            "- `Dice baseline` es el resultado final aceptado del pipeline clasico.",
            "- `Dice VoxelMorph` es el resultado de la red instance-specific.",
            "- Las metricas de flujo sirven como control: si Dice sube pero el campo es enorme o plegado, el resultado no debe aceptarse automaticamente.",
        ]
    )
    (OUTPUT_DIR / "README_voxelmorph_vs_baseline.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = {}
    for subject in SUBJECTS:
        print(f"=== {subject} ===")
        summary = run(subject=subject, iterations=250, size=256, lr=2e-3, smooth_weight=0.08, dice_weight=0.45)
        summaries[subject] = summary
        rows.append(collect_row(subject, summary))

    write_csv(rows, OUTPUT_DIR / "voxelmorph_vs_baseline_summary.csv")
    with open(OUTPUT_DIR / "voxelmorph_vs_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump({"subjects": SUBJECTS, "rows": rows, "summaries": summaries}, f, indent=2)
    create_dice_plot(rows)
    create_overlay_grid(rows)
    create_markdown_report(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
