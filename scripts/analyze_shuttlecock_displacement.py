"""Shuttlecock 开发训练集的标签中心位移与方形搜索几何覆盖。"""
import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.tennis import center_pairs, grid_targets


DELTAS = (1, 2, 4)


def split_for(source, match):
    if source == "Amateur" and 1 <= match <= 3:
        return "train"
    if source == "Professional" and 1 <= match <= 20:
        return "train"
    if source == "Professional" and 21 <= match <= 23:
        return "val"
    return None


def fps_label(value):
    fps = float(Fraction(value))
    return f"{fps:g}"


def pair_stats(rows):
    result = {}
    for delta in DELTAS:
        pairs = list(center_pairs(rows, delta))
        if pairs:
            displacement = np.array([[b["x_raw"] - a["x_raw"], b["y_raw"] - a["y_raw"]]
                                     for a, b in pairs])
            l2 = np.linalg.norm(displacement, axis=1)
            linf = np.abs(displacement).max(axis=1)
            grid_linf = []
            for a, b in pairs:
                hw = (a["height"], a["width"])
                cells = grid_targets([[a["x_raw"], a["y_raw"]], [b["x_raw"], b["y_raw"]]],
                                     [True, True], (36, 64), hw)
                ay, ax = divmod(int(cells[0]), 64)
                by, bx = divmod(int(cells[1]), 64)
                grid_linf.append(max(abs(by - ay), abs(bx - ax)))
            quantiles = [float(x) for x in np.quantile(l2, [.5, .9, .99, 1])]
            pixel_coverage = {str(r): float(np.mean(linf <= r)) for r in (20, 40, 80, 160)}
            grid_coverage = {str(r): float(np.mean(np.asarray(grid_linf) <= r)) for r in (2, 4, 8)}
        else:
            quantiles = [None] * 4
            pixel_coverage = {str(r): None for r in (20, 40, 80, 160)}
            grid_coverage = {str(r): None for r in (2, 4, 8)}
        result[str(delta)] = {
            "pairs": len(pairs),
            "l2_p50_p90_p99_max": quantiles,
            "square_radius_original_px_coverage": pixel_coverage,
            "native_36x64_square_radius_coverage": grid_coverage,
        }
    return result


def analyze(data_root, output):
    data_root, output = Path(data_root), Path(output)
    records = []
    with (data_root / "verification.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parts = Path(row["video"]).parts
            source, match = parts[0], int(parts[1][5:])
            split = split_for(source, match)
            if split is None:
                continue
            record = dict(row, split=split, source=source, match=match,
                          rally=Path(row["video"]).stem, fps_group=fps_label(row["avg_frame_rate"]))
            for key in ("width", "height", "nb_frames", "csv_rows", "first_frame", "last_frame"):
                record[key] = int(record[key])
            records.append(record)
    records.sort(key=lambda r: (r["split"] != "train", r["source"], r["match"], r["rally"]))

    rows, invalid = [], []
    for record in records:
        if record["split"] != "train":
            continue
        csv_path = data_root / "original/TrackNetV2" / record["csv"]
        with csv_path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                frame, visibility = int(raw["Frame"]), int(raw["Visibility"])
                x, y = float(raw["X"]), float(raw["Y"])
                valid_xy = (math.isfinite(x) and math.isfinite(y)
                            and 0 <= x < record["width"] and 0 <= y < record["height"])
                identity = {"source": record["source"], "match": record["match"],
                            "rally": record["rally"], "Frame": frame}
                if visibility not in (0, 1) or (visibility == 1 and not valid_xy):
                    invalid.append(dict(identity, Visibility=visibility, X=x, Y=y,
                                        reason="bad_visibility_or_visible_coordinate"))
                    continue
                rows.append(dict(identity, game=f'{record["source"]}/match{record["match"]}',
                                 clip=record["rally"], original_frame_id=frame,
                                 visibility=visibility, x_raw=x, y_raw=y,
                                 label_state="located" if visibility == 1 else "not_visible",
                                 width=record["width"], height=record["height"],
                                 fps_group=record["fps_group"]))

    groups = {"overall": rows,
              "Professional": [r for r in rows if r["source"] == "Professional"],
              "Amateur": [r for r in rows if r["source"] == "Amateur"]}
    for fps in sorted({r["fps_group"] for r in rows}, key=float):
        groups[f"fps_{fps}"] = [r for r in rows if r["fps_group"] == fps]

    summary = {
        "scope": "training labels only: Professional match1-20 and Amateur match1-3; validation metadata only; Test excluded",
        "units": "original pixels and native 36x64 cells at each rally metadata width/height",
        "pair_rule": "same source/match/rally and exact original Frame difference; both endpoints Visibility=1 with finite in-frame coordinates",
        "limit": "geometric search coverage, not correspondence or detection success rate",
        "counts": {
            "train_rallies": sum(r["split"] == "train" for r in records),
            "val_rallies": sum(r["split"] == "val" for r in records),
            "train_csv_rows": len(rows) + len(invalid),
            "located_rows": sum(r["label_state"] == "located" for r in rows),
            "not_visible_rows": sum(r["label_state"] == "not_visible" for r in rows),
            "invalid_rows": len(invalid),
            "unlabeled_video_tail_frames": sum(max(0, r["nb_frames"] - r["last_frame"] - 1)
                                                  for r in records if r["split"] == "train"),
        },
        "train": {name: pair_stats(group) for name, group in groups.items()},
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_fields = ("split", "source", "match", "rally", "video", "csv", "width", "height",
                       "avg_frame_rate", "fps_group", "nb_frames", "duration", "csv_rows",
                       "first_frame", "last_frame", "contiguous", "bad_visibility",
                       "invisible_nonzero_xy")
    with (output / "development_rallies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    if invalid:
        with (output / "invalid_labels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, invalid[0].keys())
            writer.writeheader()
            writer.writerows(invalid)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/shuttlecock")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.data_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
