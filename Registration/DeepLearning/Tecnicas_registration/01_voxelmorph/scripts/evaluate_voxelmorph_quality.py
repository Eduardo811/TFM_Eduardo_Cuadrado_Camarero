from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
METHOD_DIR = ROOT / "Registration" / "DeepLearning" / "Tecnicas_registration" / "01_voxelmorph"
INPUT_CSV = METHOD_DIR / "outputs" / "outputs_voxelmorph_instance_todos" / "voxelmorph_vs_baseline_summary.csv"
OUTPUT_DIR = METHOD_DIR / "outputs" / "outputs_voxelmorph_quality_review"


def load_rows() -> list[dict]:
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    numeric_fields = [
        "initial_dice",
        "voxelmorph_dice",
        "baseline_dice",
        "voxelmorph_minus_initial",
        "voxelmorph_minus_baseline",
        "flow_mean_px",
        "flow_p95_px",
        "flow_max_px",
        "flow_smoothness_px",
        "jacobian_det_min",
        "jacobian_det_p01",
        "jacobian_det_negative_fraction",
    ]
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def classify(row: dict) -> tuple[str, str]:
    neg = row["jacobian_det_negative_fraction"]
    p95 = row["flow_p95_px"]
    gain = row["voxelmorph_minus_baseline"]

    if neg > 0.10 or p95 > 80:
        return (
            "reject_as_final",
            "High deformation risk: excessive folding and/or very large displacement. Use only as proof of concept.",
        )
    if neg > 0.05 or p95 > 60:
        return (
            "needs_regularized_rerun",
            "Promising overlap, but deformation quality is borderline. Re-run with stronger smoothness constraints.",
        )
    if gain < 0:
        return (
            "not_better_than_baseline",
            "The classical baseline is already sufficient for this case; VoxelMorph does not improve it.",
        )
    return (
        "acceptable_exploratory",
        "Good exploratory result: improved or preserved Dice with relatively controlled deformation.",
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def color_for_status(status: str) -> tuple[int, int, int]:
    if status == "acceptable_exploratory":
        return (45, 150, 75)
    if status == "needs_regularized_rerun":
        return (220, 145, 25)
    if status == "not_better_than_baseline":
        return (120, 120, 120)
    return (205, 70, 70)


def draw_scatter(rows: list[dict], out_path: Path) -> None:
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = font(40, True)
    axis_font = font(25)
    tick_font = font(22)
    label_font = font(24, True)
    small_font = font(20)

    left, right, top, bottom = 130, 80, 155, 145
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = -0.02, 0.36
    y_min, y_max = 0.0, 0.25

    d.text((width // 2, 32), "VoxelMorph-style quality review", font=title_font, fill=(20, 20, 20), anchor="ma")
    d.text((width // 2, 82), "Overlap gain vs deformation risk", font=axis_font, fill=(60, 60, 60), anchor="ma")

    for val in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        y = top + plot_h - (val - y_min) / (y_max - y_min) * plot_h
        d.line((left, y, width - right, y), fill=(225, 225, 225), width=2)
        d.text((left - 15, y), f"{val:.2f}", font=tick_font, fill=(40, 40, 40), anchor="rm")
    for val in [0.00, 0.10, 0.20, 0.30]:
        x = left + (val - x_min) / (x_max - x_min) * plot_w
        d.line((x, top, x, height - bottom), fill=(235, 235, 235), width=2)
        d.text((x, height - bottom + 25), f"{val:.2f}", font=tick_font, fill=(40, 40, 40), anchor="ma")

    # Risk threshold used in the review.
    y_thr = top + plot_h - (0.10 - y_min) / (y_max - y_min) * plot_h
    d.line((left, y_thr, width - right, y_thr), fill=(205, 70, 70), width=4)
    d.text((width - right - 10, y_thr - 10), "10% negative Jacobian threshold", font=small_font, fill=(160, 45, 45), anchor="rd")

    d.line((left, top, left, height - bottom), fill=(30, 30, 30), width=3)
    d.line((left, height - bottom, width - right, height - bottom), fill=(30, 30, 30), width=3)
    d.text((width // 2, height - 34), "Dice improvement over classical baseline", font=axis_font, fill=(30, 30, 30), anchor="ma")
    d.text((left, top - 35), "Negative Jacobian fraction", font=axis_font, fill=(30, 30, 30), anchor="lm")

    for row in rows:
        x_val = row["voxelmorph_minus_baseline"]
        y_val = row["jacobian_det_negative_fraction"]
        x = left + (x_val - x_min) / (x_max - x_min) * plot_w
        y = top + plot_h - (y_val - y_min) / (y_max - y_min) * plot_h
        color = color_for_status(row["quality_status"])
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=color, outline=(20, 20, 20), width=2)
        d.text((x + 22, y - 5), row["subject"], font=label_font, fill=(25, 25, 25), anchor="lm")

    legend = [
        ("acceptable_exploratory", "acceptable exploratory"),
        ("needs_regularized_rerun", "needs regularized rerun"),
        ("reject_as_final", "reject as final"),
        ("not_better_than_baseline", "not better than baseline"),
    ]
    lx, ly = left, height - 82
    for idx, (status, label) in enumerate(legend):
        x = lx + idx * 330
        d.rectangle((x, ly, x + 28, ly + 22), fill=color_for_status(status))
        d.text((x + 38, ly + 11), label, font=small_font, fill=(35, 35, 35), anchor="lm")

    img.save(out_path, quality=95)


def write_outputs(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reviewed = []
    for row in rows:
        status, reason = classify(row)
        row = dict(row)
        row["quality_status"] = status
        row["quality_reason"] = reason
        reviewed.append(row)

    csv_path = OUTPUT_DIR / "voxelmorph_quality_review.csv"
    fieldnames = [
        "subject",
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
        "quality_status",
        "quality_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in reviewed:
            writer.writerow({k: row[k] for k in fieldnames})

    with open(OUTPUT_DIR / "voxelmorph_quality_review.json", "w", encoding="utf-8") as f:
        json.dump(reviewed, f, indent=2)

    mean_baseline = sum(r["baseline_dice"] for r in reviewed) / len(reviewed)
    mean_vm = sum(r["voxelmorph_dice"] for r in reviewed) / len(reviewed)
    mean_neg = sum(r["jacobian_det_negative_fraction"] for r in reviewed) / len(reviewed)

    md_lines = [
        "# VoxelMorph-style quality review",
        "",
        "This review closes the first instance-specific VoxelMorph-style experiment by considering both mask overlap and deformation plausibility.",
        "",
        "## Summary",
        "",
        f"- Mean classical baseline Dice: {mean_baseline:.3f}",
        f"- Mean VoxelMorph-style Dice: {mean_vm:.3f}",
        f"- Mean Dice gain over baseline: {mean_vm - mean_baseline:+.3f}",
        f"- Mean negative-Jacobian fraction: {mean_neg:.3f}",
        "",
        "## Case-level decision",
        "",
        "| Subject | Baseline Dice | VoxelMorph Dice | Gain | P95 flow (px) | Negative Jacobian | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in reviewed:
        md_lines.append(
            f"| {row['subject']} | {row['baseline_dice']:.3f} | {row['voxelmorph_dice']:.3f} | "
            f"{row['voxelmorph_minus_baseline']:+.3f} | {row['flow_p95_px']:.1f} | "
            f"{row['jacobian_det_negative_fraction']:.3f} | {row['quality_status']} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The experiment is successful as a proof of concept: the neural instance-specific refinement can increase mask overlap after the classical initialization.",
            "- It is not yet acceptable as a final registration method: several cases have high displacement and a large fraction of negative Jacobians, indicating folding or non-plausible local deformation.",
            "- The next technical step should be a regularized VoxelMorph rerun, increasing smoothness/topology constraints and comparing the trade-off between Dice and deformation plausibility.",
            "",
            "## Recommended use in the TFM",
            "",
            "Present the current VoxelMorph-style experiment as an exploratory deep-learning registration result, not as the definitive HSI-H&E registration pipeline.",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(md_lines), encoding="utf-8")
    draw_scatter(reviewed, OUTPUT_DIR / "voxelmorph_quality_review_scatter.png")


def main() -> None:
    rows = load_rows()
    write_outputs(rows)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
