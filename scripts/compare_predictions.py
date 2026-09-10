"""复用已保存的逐帧预测比较净收益，不重新运行模型。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.probe import evaluate_predictions, paired_location_changes


def read_predictions(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["visibility_raw"] = int(row["visibility_raw"])
        for key in ("x_raw", "y_raw", "pred_x", "pred_y", "presence_probability"):
            row[key] = float(row[key]) if row[key] else None
    return rows, np.array([[r["pred_x"], r["pred_y"]] for r in rows]), np.array([r["presence_probability"] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows_a, xy_a, prob_a = read_predictions(args.baseline)
    rows_b, xy_b, prob_b = read_predictions(args.challenger)
    keys = ("game", "clip", "original_frame_id", "visibility_raw", "x_raw", "y_raw")
    if [tuple(r[k] for k in keys) for r in rows_a] != [tuple(r[k] for k in keys) for r in rows_b]:
        raise ValueError("预测的目标集合、顺序或标签不同；不能静默取交集作主要比较")
    result = dict(baseline=str(args.baseline), challenger=str(args.challenger),
                  baseline_metrics=evaluate_predictions(rows_a, xy_a, prob_a),
                  challenger_metrics=evaluate_predictions(rows_b, xy_b, prob_b),
                  paired=paired_location_changes(rows_a, xy_a, xy_b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["paired"]["all"], indent=2))


if __name__ == "__main__":
    main()
