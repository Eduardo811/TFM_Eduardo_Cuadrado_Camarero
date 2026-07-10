from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from run_transmorph_instance import METHOD_DIR, ROOT, run


SUBJECTS = ["SB012", "SB013", "SB017", "SB018", "SB019", "SB020"]
CONFIGS = [
    {
        "tag": "tm_s040_d012",
        "label": "smooth=0.40, max_disp=0.12",
        "smooth_weight": 0.40,
        "max_disp_norm": 0.12,
    },
    {
        "tag": "tm_s060_d010",
        "label": "smooth=0.60, max_disp=0.10",
        "smooth_weight": 0.60,
        "max_disp_norm": 0.10,
    },
    {
        "tag": "tm_s100_d008",
        "label": "smooth=1.00, max_disp=0.08",
        "smooth_weight": 1.00,
        "max_disp_norm": 0.08,
    },
]

BASELINE_FINAL_DIR = (
    ROOT
    / "Registration"
    / "DeepLearning"
    / "Tecnicas_registration"
    / "00_baseline_clasico"
    / "outputs"
    / "outputs_baseline_clasico_final"
)
VOXELMORPH_FINAL_CSV = (
    ROOT
    / "Registration"
    / "DeepLearning"
    / "Tecnicas_registration"
    / "01_voxelmorph"
    / "outputs"
    / "outputs_voxelmorph_final_block"
    / "voxelmorph_final_selected_summary.csv"
)
GRID_DIR = METHOD_DIR / "outputs" / "outputs_transmorph_grid"
FINAL_DIR = METHOD_DIR / "outputs" / "outputs_transmorph_final_block"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_voxelmorph_selected() -> dict[str, dict]:
    if not VOXELMORPH_FINAL_CSV.exists():
        return {}
    return {row["subject"]: row for row in read_csv(VOXELMORPH_FINAL_CSV)}


def summary_path(subject: str, tag: str) -> Path:
    return METHOD_DIR / "outputs" / f"outputs_transmorph_instance_{tag}_{subject}" / f"{subject}_transmorph_instance_summary.json"


def quality_status(row: dict) -> str:
    gain = float(row["transmorph_minus_baseline"])
    p95 = float(row["flow_p95_px"])
    neg = float(row["jacobian_det_negative_fraction"])
    if neg > 0.10 or p95 > 70:
        return "reject_as_final"
    if neg > 0.05 or p95 > 50:
        return "needs_review"
    if gain < -0.002:
        return "not_better_than_baseline"
    return "acceptable_exploratory"


def status_label(status: str) -> str:
    return {
        "acceptable_exploratory": "acceptable exploratory",
        "needs_review": "needs visual review",
        "reject_as_final": "reject as final",
        "not_better_than_baseline": "not better than baseline",
    }.get(status, status)


def collect_row(subject: str, config: dict, summary: dict, vox_rows: dict[str, dict]) -> dict:
    baseline = load_json(BASELINE_FINAL_DIR / subject / f"{subject}_baseline_summary.json")
    baseline_dice = float(baseline["dice_final"])
    vox = vox_rows.get(subject, {})
    row = {
        "config_tag": config["tag"],
        "config_label": config["label"],
        "smooth_weight": config["smooth_weight"],
        "max_disp_norm": config["max_disp_norm"],
        "subject": subject,
        "initial_stage": summary["init_stage"],
        "input_size": summary["input_size"],
        "iterations": summary["iterations"],
        "transformer_depth": summary["transformer_depth"],
        "transformer_heads": summary["transformer_heads"],
        "initial_dice": summary["initial_dice"],
        "baseline_dice": baseline_dice,
        "voxelmorph_selected_dice": float(vox["voxelmorph_dice"]) if vox else np.nan,
        "transmorph_dice": summary["final_dice"],
        "transmorph_minus_baseline": summary["final_dice"] - baseline_dice,
        "transmorph_minus_voxelmorph": summary["final_dice"] - float(vox["voxelmorph_dice"]) if vox else np.nan,
        "flow_mean_px": summary["flow_mean_px"],
        "flow_p95_px": summary["flow_p95_px"],
        "flow_max_px": summary["flow_max_px"],
        "flow_smoothness_px": summary["flow_smoothness_px"],
        "jacobian_det_min": summary["jacobian_det_min"],
        "jacobian_det_p01": summary["jacobian_det_p01"],
        "jacobian_det_negative_fraction": summary["jacobian_det_negative_fraction"],
    }
    row["quality_status"] = quality_status(row)
    return row


def summarize_by_config(rows: list[dict]) -> list[dict]:
    out = []
    for config in CONFIGS:
        cfg_rows = [row for row in rows if row["config_tag"] == config["tag"]]
        statuses = [row["quality_status"] for row in cfg_rows]
        out.append(
            {
                "config_tag": config["tag"],
                "config_label": config["label"],
                "mean_transmorph_dice": float(np.mean([float(row["transmorph_dice"]) for row in cfg_rows])),
                "mean_gain_vs_baseline": float(np.mean([float(row["transmorph_minus_baseline"]) for row in cfg_rows])),
                "mean_gain_vs_voxelmorph": float(np.nanmean([float(row["transmorph_minus_voxelmorph"]) for row in cfg_rows])),
                "mean_flow_p95_px": float(np.mean([float(row["flow_p95_px"]) for row in cfg_rows])),
                "mean_negative_jacobian_fraction": float(np.mean([float(row["jacobian_det_negative_fraction"]) for row in cfg_rows])),
                "acceptable_count": statuses.count("acceptable_exploratory"),
                "needs_review_count": statuses.count("needs_review"),
                "reject_count": statuses.count("reject_as_final"),
                "not_better_count": statuses.count("not_better_than_baseline"),
            }
        )
    return out


def select_best_config(config_summary: list[dict]) -> str:
    best = min(
        config_summary,
        key=lambda row: (
            row["reject_count"],
            row["needs_review_count"],
            row["not_better_count"],
            -row["acceptable_count"],
            -row["mean_gain_vs_baseline"],
            row["mean_negative_jacobian_fraction"],
        ),
    )
    return best["config_tag"]


def create_tradeoff_plot(config_summary: list[dict]) -> None:
    labels = [row["config_tag"].replace("_", "\n") for row in config_summary]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - 0.18, [row["mean_gain_vs_baseline"] for row in config_summary], 0.36, color="#4477aa", label="Mean Dice gain vs baseline")
    ax1.bar(x + 0.18, [row["mean_gain_vs_voxelmorph"] for row in config_summary], 0.36, color="#66aa55", label="Mean Dice gain vs VoxelMorph")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Mean Dice gain")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["mean_negative_jacobian_fraction"] for row in config_summary], color="#cc4444", marker="o", label="Negative Jacobian")
    ax2.set_ylabel("Mean negative Jacobian fraction")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.set_title("TransMorph-style regularization trade-off")
    fig.tight_layout()
    fig.savefig(GRID_DIR / "transmorph_regularization_tradeoff.png", dpi=180)
    plt.close(fig)


def create_overlay_grid(rows: list[dict]) -> None:
    tags = [config["tag"] for config in CONFIGS]
    fig, axes = plt.subplots(len(SUBJECTS), len(tags), figsize=(5.2 * len(tags), 4.4 * len(SUBJECTS)))
    for i, subject in enumerate(SUBJECTS):
        for j, tag in enumerate(tags):
            row = next(r for r in rows if r["subject"] == subject and r["config_tag"] == tag)
            overlay = METHOD_DIR / "outputs" / f"outputs_transmorph_instance_{tag}_{subject}" / f"{subject}_overlay_contours_transmorph_instance.png"
            axes[i, j].imshow(Image.open(overlay))
            axes[i, j].set_title(
                f"{subject} | Dice={float(row['transmorph_dice']):.3f}\n"
                f"gain={float(row['transmorph_minus_baseline']):+.3f}, negJ={float(row['jacobian_det_negative_fraction']):.3f}"
            )
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(GRID_DIR / "transmorph_regularization_overlay_grid.png", dpi=180)
    plt.close(fig)


def create_method_comparison(selected_rows: list[dict]) -> None:
    subjects = [row["subject"] for row in selected_rows]
    x = np.arange(len(subjects))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, [float(row["baseline_dice"]) for row in selected_rows], width, label="Classical baseline")
    ax.bar(x, [float(row["voxelmorph_selected_dice"]) for row in selected_rows], width, label="VoxelMorph selected")
    ax.bar(x + width, [float(row["transmorph_dice"]) for row in selected_rows], width, label="TransMorph-style selected")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Mask Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Selected deep-learning registration comparison")
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "transmorph_selected_method_comparison.png", dpi=180)
    plt.close(fig)


def create_quality_scatter(selected_rows: list[dict]) -> None:
    colors = {
        "acceptable_exploratory": "#2ca02c",
        "needs_review": "#ff7f0e",
        "reject_as_final": "#d62728",
        "not_better_than_baseline": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in selected_rows:
        status = row["quality_status"]
        ax.scatter(
            float(row["flow_p95_px"]),
            float(row["transmorph_minus_baseline"]),
            s=95,
            color=colors.get(status, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.5,
            label=status_label(status),
        )
        ax.text(float(row["flow_p95_px"]) + 0.8, float(row["transmorph_minus_baseline"]), row["subject"], fontsize=9, va="center")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("P95 displacement (px)")
    ax.set_ylabel("Dice gain vs classical baseline")
    ax.set_title("Selected TransMorph-style quality review")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "transmorph_selected_quality_scatter.png", dpi=180)
    plt.close(fig)


def create_selected_montage(selected_rows: list[dict], selected_tag: str) -> None:
    thumbs = []
    for row in selected_rows:
        subject = row["subject"]
        src = METHOD_DIR / "outputs" / f"outputs_transmorph_instance_{selected_tag}_{subject}" / f"{subject}_transmorph_instance_summary.png"
        if not src.exists():
            continue
        image = Image.open(src).convert("RGB")
        image.thumbnail((720, 390))
        canvas = Image.new("RGB", (760, 445), "white")
        canvas.paste(image, ((760 - image.width) // 2, 35))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{subject} | Dice {float(row['baseline_dice']):.3f} -> {float(row['transmorph_dice']):.3f} "
            f"({float(row['transmorph_minus_baseline']):+.3f}) | {status_label(row['quality_status'])}"
        )
        draw.text((18, 12), title, fill="black")
        thumbs.append(canvas)
    cols = 2
    rows_n = int(np.ceil(len(thumbs) / cols))
    montage = Image.new("RGB", (cols * 760, rows_n * 445), "white")
    for idx, thumb in enumerate(thumbs):
        montage.paste(thumb, ((idx % cols) * 760, (idx // cols) * 445))
    montage.save(FINAL_DIR / "transmorph_selected_summary_montage.png")


def copy_selected_outputs(selected_rows: list[dict], selected_tag: str) -> None:
    for row in selected_rows:
        subject = row["subject"]
        src_dir = METHOD_DIR / "outputs" / f"outputs_transmorph_instance_{selected_tag}_{subject}"
        dst_dir = FINAL_DIR / "selected_subject_outputs" / subject
        dst_dir.mkdir(parents=True, exist_ok=True)
        mappings = [
            (f"{subject}_he_transmorph_instance_to_hsi.png", f"{subject}_he_transmorph_selected_to_hsi.png"),
            (f"{subject}_he_mask_transmorph_instance_to_hsi.png", f"{subject}_he_mask_transmorph_selected_to_hsi.png"),
            (f"{subject}_overlay_contours_transmorph_instance.png", f"{subject}_overlay_contours_transmorph_selected.png"),
            (f"{subject}_transmorph_instance_summary.png", f"{subject}_transmorph_selected_summary.png"),
            (f"{subject}_transmorph_instance_summary.json", f"{subject}_transmorph_selected_summary.json"),
        ]
        for src_name, dst_name in mappings:
            src = src_dir / src_name
            if src.exists():
                shutil.copy2(src, dst_dir / dst_name)


def write_reports(rows: list[dict], config_summary: list[dict], selected_tag: str) -> None:
    selected_rows = [row for row in rows if row["config_tag"] == selected_tag]
    selected_summary = next(row for row in config_summary if row["config_tag"] == selected_tag)

    grid_lines = [
        "# TransMorph-style grid",
        "",
        "This folder contains the regularization grid for the 2D TransMorph-style exploratory experiment.",
        "",
        "| Config | Mean Dice | Gain vs baseline | Gain vs VoxelMorph | Mean p95 flow | Mean neg. Jacobian | Accept | Review | Reject | Not better |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in config_summary:
        grid_lines.append(
            f"| {row['config_tag']} | {row['mean_transmorph_dice']:.3f} | {row['mean_gain_vs_baseline']:+.3f} | "
            f"{row['mean_gain_vs_voxelmorph']:+.3f} | {row['mean_flow_p95_px']:.1f} | "
            f"{row['mean_negative_jacobian_fraction']:.3f} | {row['acceptable_count']} | {row['needs_review_count']} | "
            f"{row['reject_count']} | {row['not_better_count']} |"
        )
    grid_lines.extend(
        [
            "",
            f"Selected automatic compromise: `{selected_tag}`.",
        ]
    )
    (GRID_DIR / "README.md").write_text("\n".join(grid_lines), encoding="utf-8")

    final_lines = [
        "# TransMorph-style final block",
        "",
        "This folder summarizes the exploratory TransMorph-style registration experiment for the H&E -> HSI task.",
        "",
        "## Why this is TransMorph-style, not the official TransMorph model",
        "",
        "The original TransMorph paper proposes a hybrid Transformer-ConvNet architecture for unsupervised volumetric medical image registration. Our dataset is much smaller and 2D: six paired H&E-HSI specimens. For this reason, the experiment here adapts the idea to a lightweight 2D instance-specific setting rather than training the original 3D model.",
        "",
        "Reference: Chen et al., `TransMorph: Transformer for unsupervised medical image registration`, Medical Image Analysis, 82, 102615, 2022. DOI: 10.1016/j.media.2022.102615.",
        "",
        "## What was tested",
        "",
        "A small 2D hybrid CNN-Transformer network receives signed distance maps from the H&E and HSI masks. A convolutional encoder extracts local structure, a Transformer bottleneck models longer-range spatial relationships, and a convolutional decoder predicts a dense deformation field.",
        "",
        "As with the VoxelMorph-style experiment, the model is optimized separately for each subject. It should therefore be interpreted as an instance-specific deep registration proof of concept, not as a universal trained model.",
        "",
        "## Selected configuration",
        "",
        f"- Selected config: `{selected_tag}`",
        f"- Mean TransMorph-style Dice: {selected_summary['mean_transmorph_dice']:.3f}",
        f"- Mean gain vs classical baseline: {selected_summary['mean_gain_vs_baseline']:+.3f}",
        f"- Mean gain vs selected VoxelMorph: {selected_summary['mean_gain_vs_voxelmorph']:+.3f}",
        f"- Mean p95 displacement: {selected_summary['mean_flow_p95_px']:.1f} px",
        f"- Mean negative Jacobian fraction: {selected_summary['mean_negative_jacobian_fraction']:.3f}",
        f"- Quality status count: {selected_summary['acceptable_count']} acceptable, {selected_summary['needs_review_count']} review, {selected_summary['reject_count']} reject, {selected_summary['not_better_count']} not better",
        "",
        "## Case-level selected results",
        "",
        "| Subject | Baseline Dice | VoxelMorph selected | TransMorph-style | Gain vs baseline | Gain vs VoxelMorph | P95 flow | Neg. Jacobian | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected_rows:
        final_lines.append(
            f"| {row['subject']} | {float(row['baseline_dice']):.3f} | {float(row['voxelmorph_selected_dice']):.3f} | "
            f"{float(row['transmorph_dice']):.3f} | {float(row['transmorph_minus_baseline']):+.3f} | "
            f"{float(row['transmorph_minus_voxelmorph']):+.3f} | {float(row['flow_p95_px']):.1f} | "
            f"{float(row['jacobian_det_negative_fraction']):.3f} | {status_label(row['quality_status'])} |"
        )
    final_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment gives a second deep-learning family after VoxelMorph: a transformer-based registration variant. It is useful to show that the project explored both CNN-style and transformer-style deformable registration.",
            "",
            "The main limitation is the same as before: there are only six image pairs, so the model cannot be presented as a trained general solution. The result is best framed as an exploratory instance-specific experiment.",
            "",
            "If the selected TransMorph-style configuration does not outperform VoxelMorph, the correct conclusion is not that the experiment failed, but that VoxelMorph remains the stronger deep-learning block for this dataset.",
            "",
            "## Files in this folder",
            "",
            "- `transmorph_final_selected_summary.csv`: clean case-level selected table.",
            "- `transmorph_final_selected_summary.json`: same information in JSON format.",
            "- `transmorph_selected_method_comparison.png`: baseline vs VoxelMorph vs TransMorph-style.",
            "- `transmorph_selected_quality_scatter.png`: Dice gain vs deformation magnitude.",
            "- `transmorph_selected_summary_montage.png`: visual summary of selected TransMorph-style outputs.",
            "- `selected_subject_outputs/`: copied outputs for the selected configuration.",
        ]
    )
    (FINAL_DIR / "README.md").write_text("\n".join(final_lines), encoding="utf-8")


def create_final_csv_json(rows: list[dict], selected_tag: str) -> list[dict]:
    selected_rows = [row for row in rows if row["config_tag"] == selected_tag]
    clean_rows = []
    for row in selected_rows:
        clean_rows.append(
            {
                "subject": row["subject"],
                "selected_config": row["config_tag"],
                "initial_stage": row["initial_stage"],
                "initial_dice": f"{float(row['initial_dice']):.6f}",
                "classical_baseline_dice": f"{float(row['baseline_dice']):.6f}",
                "voxelmorph_selected_dice": f"{float(row['voxelmorph_selected_dice']):.6f}",
                "transmorph_dice": f"{float(row['transmorph_dice']):.6f}",
                "transmorph_minus_baseline": f"{float(row['transmorph_minus_baseline']):+.6f}",
                "transmorph_minus_voxelmorph": f"{float(row['transmorph_minus_voxelmorph']):+.6f}",
                "flow_p95_px": f"{float(row['flow_p95_px']):.3f}",
                "negative_jacobian_fraction": f"{float(row['jacobian_det_negative_fraction']):.6f}",
                "quality_status": row["quality_status"],
                "interpretation": status_label(row["quality_status"]),
            }
        )
    write_csv(clean_rows, FINAL_DIR / "transmorph_final_selected_summary.csv")
    with open(FINAL_DIR / "transmorph_final_selected_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_rows, f, indent=2)
    return selected_rows


def main() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    vox_rows = load_voxelmorph_selected()
    rows = []
    summaries = {}

    for config in CONFIGS:
        for subject in SUBJECTS:
            print(f"=== {config['tag']} | {subject} ===")
            path = summary_path(subject, config["tag"])
            if path.exists():
                print(f"loading existing result: {path}")
                summary = load_json(path)
            else:
                summary = run(
                    subject=subject,
                    iterations=140,
                    size=192,
                    lr=1.5e-3,
                    smooth_weight=config["smooth_weight"],
                    dice_weight=0.45,
                    max_disp_norm=config["max_disp_norm"],
                    transformer_depth=2,
                    transformer_heads=4,
                    output_tag=config["tag"],
                )
            row = collect_row(subject, config, summary, vox_rows)
            rows.append(row)
            summaries[f"{config['tag']}::{subject}"] = summary

    config_summary = summarize_by_config(rows)
    selected_tag = select_best_config(config_summary)
    selected_rows = create_final_csv_json(rows, selected_tag)

    write_csv(rows, GRID_DIR / "transmorph_grid_summary.csv")
    write_csv(config_summary, GRID_DIR / "transmorph_grid_config_summary.csv")
    with open(GRID_DIR / "transmorph_grid_summary.json", "w", encoding="utf-8") as f:
        json.dump({"configs": CONFIGS, "subjects": SUBJECTS, "rows": rows, "config_summary": config_summary, "selected_tag": selected_tag, "summaries": summaries}, f, indent=2)

    create_tradeoff_plot(config_summary)
    create_overlay_grid(rows)
    create_method_comparison(selected_rows)
    create_quality_scatter(selected_rows)
    create_selected_montage(selected_rows, selected_tag)
    copy_selected_outputs(selected_rows, selected_tag)
    write_reports(rows, config_summary, selected_tag)

    print(json.dumps({"selected_tag": selected_tag, "config_summary": config_summary}, indent=2))


if __name__ == "__main__":
    main()
