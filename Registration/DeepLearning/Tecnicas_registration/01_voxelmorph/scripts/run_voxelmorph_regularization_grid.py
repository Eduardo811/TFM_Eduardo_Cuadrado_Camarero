from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_voxelmorph_instance import METHOD_DIR, ROOT, run


SUBJECTS = ["SB012", "SB013", "SB017", "SB018", "SB019", "SB020"]
CONFIGS = [
    {
        "tag": "reg_s015_d018",
        "label": "smooth=0.15, max_disp=0.18",
        "smooth_weight": 0.15,
        "max_disp_norm": 0.18,
    },
    {
        "tag": "reg_s030_d015",
        "label": "smooth=0.30, max_disp=0.15",
        "smooth_weight": 0.30,
        "max_disp_norm": 0.15,
    },
    {
        "tag": "reg_s050_d012",
        "label": "smooth=0.50, max_disp=0.12",
        "smooth_weight": 0.50,
        "max_disp_norm": 0.12,
    },
    {
        "tag": "reg_s080_d010",
        "label": "smooth=0.80, max_disp=0.10",
        "smooth_weight": 0.80,
        "max_disp_norm": 0.10,
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
CURRENT_SUMMARY_CSV = METHOD_DIR / "outputs" / "outputs_voxelmorph_instance_todos" / "voxelmorph_vs_baseline_summary.csv"
OUTPUT_DIR = METHOD_DIR / "outputs" / "outputs_voxelmorph_regularization_grid"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_current_rows() -> list[dict]:
    with open(CURRENT_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["config_tag"] = "current_s008_d022"
        row["config_label"] = "current smooth=0.08, max_disp=0.22"
        row["smooth_weight"] = 0.08
        row["max_disp_norm"] = 0.22
        for key in [
            "initial_dice",
            "baseline_dice",
            "voxelmorph_dice",
            "voxelmorph_minus_baseline",
            "flow_mean_px",
            "flow_p95_px",
            "flow_max_px",
            "flow_smoothness_px",
            "jacobian_det_min",
            "jacobian_det_p01",
            "jacobian_det_negative_fraction",
        ]:
            row[key] = float(row[key])
    return rows


def collect_row(subject: str, config: dict, summary: dict) -> dict:
    baseline_path = BASELINE_FINAL_DIR / subject / f"{subject}_baseline_summary.json"
    baseline = load_json(baseline_path)
    baseline_dice = float(baseline.get("dice_final"))
    return {
        "config_tag": config["tag"],
        "config_label": config["label"],
        "smooth_weight": config["smooth_weight"],
        "max_disp_norm": config["max_disp_norm"],
        "subject": subject,
        "initial_stage": summary["init_stage"],
        "initial_dice": summary["initial_dice"],
        "baseline_dice": baseline_dice,
        "voxelmorph_dice": summary["final_dice"],
        "voxelmorph_minus_baseline": summary["final_dice"] - baseline_dice,
        "flow_mean_px": summary.get("flow_mean_px"),
        "flow_p95_px": summary.get("flow_p95_px"),
        "flow_max_px": summary.get("flow_max_px"),
        "flow_smoothness_px": summary.get("flow_smoothness_px"),
        "jacobian_det_min": summary.get("jacobian_det_min"),
        "jacobian_det_p01": summary.get("jacobian_det_p01"),
        "jacobian_det_negative_fraction": summary.get("jacobian_det_negative_fraction"),
    }


def existing_summary_path(subject: str, tag: str) -> Path:
    return (
        METHOD_DIR
        / "outputs"
        / f"outputs_voxelmorph_instance_{tag}_{subject}"
        / f"{subject}_voxelmorph_instance_summary.json"
    )


def quality_status(row: dict) -> str:
    neg = row["jacobian_det_negative_fraction"]
    p95 = row["flow_p95_px"]
    gain = row["voxelmorph_minus_baseline"]
    if neg > 0.10 or p95 > 80:
        return "reject_as_final"
    if neg > 0.05 or p95 > 60:
        return "needs_review"
    if gain < -0.002:
        return "not_better_than_baseline"
    return "acceptable_exploratory"


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_config(rows: list[dict]) -> list[dict]:
    summary = []
    for tag in sorted({r["config_tag"] for r in rows}):
        cfg_rows = [r for r in rows if r["config_tag"] == tag]
        statuses = [r["quality_status"] for r in cfg_rows]
        summary.append(
            {
                "config_tag": tag,
                "config_label": cfg_rows[0]["config_label"],
                "mean_dice": float(np.mean([r["voxelmorph_dice"] for r in cfg_rows])),
                "mean_gain_vs_baseline": float(np.mean([r["voxelmorph_minus_baseline"] for r in cfg_rows])),
                "mean_flow_p95_px": float(np.mean([r["flow_p95_px"] for r in cfg_rows])),
                "mean_negative_jacobian_fraction": float(np.mean([r["jacobian_det_negative_fraction"] for r in cfg_rows])),
                "acceptable_count": statuses.count("acceptable_exploratory"),
                "needs_review_count": statuses.count("needs_review"),
                "reject_count": statuses.count("reject_as_final"),
                "not_better_count": statuses.count("not_better_than_baseline"),
            }
        )
    return summary


def create_tradeoff_plot(config_summary: list[dict]) -> None:
    labels = [s["config_tag"].replace("_", "\n") for s in config_summary]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x - 0.18, [s["mean_gain_vs_baseline"] for s in config_summary], 0.36, label="Mean Dice gain", color="#2f7ebc")
    ax1.set_ylabel("Mean Dice gain vs baseline")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, [s["mean_negative_jacobian_fraction"] for s in config_summary], marker="o", color="#c94040", label="Mean negative Jacobian")
    ax2.set_ylabel("Mean negative Jacobian fraction")
    ax2.axhline(0.10, color="#c94040", linestyle="--", linewidth=1.2, alpha=0.7)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    ax1.set_title("VoxelMorph regularization trade-off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regularization_tradeoff.png", dpi=180)
    plt.close(fig)


def create_case_grid(rows: list[dict]) -> None:
    tags = [c["tag"] for c in CONFIGS]
    fig, axes = plt.subplots(len(SUBJECTS), len(tags), figsize=(5.2 * len(tags), 4.4 * len(SUBJECTS)))
    for i, subject in enumerate(SUBJECTS):
        for j, tag in enumerate(tags):
            row = next(r for r in rows if r["subject"] == subject and r["config_tag"] == tag)
            overlay = METHOD_DIR / "outputs" / f"outputs_voxelmorph_instance_{tag}_{subject}" / f"{subject}_overlay_contours_voxelmorph_instance.png"
            axes[i, j].imshow(Image.open(overlay))
            axes[i, j].set_title(
                f"{subject} | Dice={row['voxelmorph_dice']:.3f}\n"
                f"negJ={row['jacobian_det_negative_fraction']:.3f}, p95={row['flow_p95_px']:.1f}"
            )
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regularization_overlay_grid.png", dpi=180)
    plt.close(fig)


def create_report(rows: list[dict], config_summary: list[dict]) -> None:
    lines = [
        "# VoxelMorph regularization grid",
        "",
        "This experiment reruns the instance-specific VoxelMorph-style method with stronger smoothness and lower maximum displacement. The goal is not to maximize Dice alone, but to reduce deformation folding and obtain a more plausible registration.",
        "",
        "## Config-level summary",
        "",
        "| Config | Mean Dice | Mean gain | Mean p95 flow | Mean negative Jacobian | Acceptable | Review | Reject | Not better |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in config_summary:
        lines.append(
            f"| {s['config_tag']} | {s['mean_dice']:.3f} | {s['mean_gain_vs_baseline']:+.3f} | "
            f"{s['mean_flow_p95_px']:.1f} | {s['mean_negative_jacobian_fraction']:.3f} | "
            f"{s['acceptable_count']} | {s['needs_review_count']} | {s['reject_count']} | {s['not_better_count']} |"
        )

    lines.extend(
        [
            "",
            "## Case-level results",
            "",
            "| Config | Subject | Baseline Dice | VM Dice | Gain | p95 flow | Negative Jacobian | Status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['config_tag']} | {row['subject']} | {row['baseline_dice']:.3f} | "
            f"{row['voxelmorph_dice']:.3f} | {row['voxelmorph_minus_baseline']:+.3f} | "
            f"{row['flow_p95_px']:.1f} | {row['jacobian_det_negative_fraction']:.3f} | {row['quality_status']} |"
        )

    best = min(
        config_summary,
        key=lambda s: (
            s["reject_count"],
            s["needs_review_count"],
            -s["acceptable_count"],
            -s["mean_gain_vs_baseline"],
        ),
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best current trade-off by the automatic criteria: `{best['config_tag']}`.",
            "- This should still be checked visually, because mask Dice and Jacobian statistics do not guarantee correct histological correspondence.",
            "- If all regularized configurations lose too much Dice, the defensible conclusion is that VoxelMorph is useful as a proof of concept but not yet reliable as the final pipeline with the current data.",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = {}
    current_rows = load_current_rows()
    for row in current_rows:
        row["quality_status"] = quality_status(row)

    for config in CONFIGS:
        for subject in SUBJECTS:
            print(f"=== {config['tag']} | {subject} ===")
            summary_path = existing_summary_path(subject, config["tag"])
            if summary_path.exists():
                print(f"loading existing result: {summary_path}")
                summary = load_json(summary_path)
            else:
                summary = run(
                    subject=subject,
                    iterations=250,
                    size=256,
                    lr=2e-3,
                    smooth_weight=config["smooth_weight"],
                    dice_weight=0.45,
                    max_disp_norm=config["max_disp_norm"],
                    output_tag=config["tag"],
                )
            row = collect_row(subject, config, summary)
            row["quality_status"] = quality_status(row)
            rows.append(row)
            summaries[f"{config['tag']}::{subject}"] = summary

    all_rows = current_rows + rows
    write_csv(rows, OUTPUT_DIR / "regularized_voxelmorph_summary.csv")
    write_csv(all_rows, OUTPUT_DIR / "regularized_voxelmorph_with_current_summary.csv")
    config_summary = summarize_by_config(all_rows)
    write_csv(config_summary, OUTPUT_DIR / "regularized_voxelmorph_config_summary.csv")
    with open(OUTPUT_DIR / "regularized_voxelmorph_summary.json", "w", encoding="utf-8") as f:
        json.dump({"configs": CONFIGS, "subjects": SUBJECTS, "rows": rows, "config_summary": config_summary, "summaries": summaries}, f, indent=2)
    create_tradeoff_plot(config_summary)
    create_case_grid(rows)
    create_report(all_rows, config_summary)
    print(json.dumps(config_summary, indent=2))


if __name__ == "__main__":
    main()
