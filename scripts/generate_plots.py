from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import RESULTS_DIR


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_summary_plot(rows, output_path: Path) -> None:
    width, height = 1200, 700
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.rectangle((40, 40, width - 40, height - 40), outline=(30, 30, 30), width=2)
    draw.text((70, 50), "Ablation Summary", fill=(0, 0, 0), font=title_font)

    labels = [row["config_id"] for row in rows]
    values = [float(row.get("selected_evidence", 0)) for row in rows]
    max_value = max(values) if values else 1
    chart_x = 90
    chart_y = 120
    chart_w = 1000
    chart_h = 480
    bar_w = chart_w / max(len(values), 1) - 30

    for idx, value in enumerate(values):
        x0 = chart_x + idx * (bar_w + 18)
        x1 = x0 + bar_w
        y0 = chart_y + chart_h - int((value / max_value) * (chart_h - 20))
        y1 = chart_y + chart_h
        draw.rectangle((x0, y0, x1, y1), fill=(60, 120, 216))
        draw.text((x0, y1 + 10), labels[idx], fill=(0, 0, 0), font=font)
        draw.text((x0, max(0, y0 - 25)), str(int(value)), fill=(0, 0, 0), font=font)

    image.save(output_path)


def main() -> None:
    summary_csv = RESULTS_DIR / "experiment_summary.csv"
    output_json = RESULTS_DIR / "experiment_summary.json"
    plot_path = RESULTS_DIR / "plots" / "ablation_summary.png"
    if not summary_csv.exists():
        raise FileNotFoundError("No experiment summary CSV found. Run scripts/run_experiments.py first.")

    rows = load_rows(summary_csv)
    save_summary_plot(rows, plot_path)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, indent=2)

    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
