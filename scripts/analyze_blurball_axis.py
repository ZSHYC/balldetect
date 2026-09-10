"""BlurBall 训练侧拖影轴与真实帧间中点位移轴诊断。"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.tennis import center_pairs


def axis_error(theta, half_length, dx, dy):
    if (not all(math.isfinite(v) for v in (theta, half_length, dx, dy))
            or half_length <= 0 or math.hypot(dx, dy) == 0):
        return None
    direction = math.degrees(math.atan2(dy, dx))
    return abs((theta - direction + 90) % 180 - 90)


def split_for(match):
    if 0 <= match <= 17:
        return "train"
    if 18 <= match <= 21:
        return "val"
    return None


def quantiles(values, probabilities):
    return [float(x) for x in np.quantile(values, probabilities)] if values else [None] * len(probabilities)


def pair_summary(pairs):
    result = {}
    for delta in (1, 2):
        selected = [p for p in pairs if p["delta"] == delta]
        angles = [p["angle_error_deg"] for p in selected if p["angle_error_deg"] is not None]
        result[str(delta)] = {
            "position_pairs": len(selected),
            "axis_pairs": len(angles),
            "displacement_l2_p50_p90_p99_max": quantiles(
                [p["displacement_l2"] for p in selected], [.5, .9, .99, 1]),
            "angle_error_median_p90": quantiles(angles, [.5, .9]),
            "within_deg": {str(a): float(np.mean(np.asarray(angles) <= a)) if angles else None
                           for a in (15, 30, 45)},
            "l_zero_pairs": sum(p["l_valid"] and p["l"] == 0 for p in selected),
            "zero_displacement_pairs": sum(p["displacement_l2"] == 0 for p in selected),
            "invalid_blur_pairs": sum(not p["blur_valid"] for p in selected),
            "axis_unavailable_reasons": {
                reason: sum(p["axis_unavailable_reason"] == reason for p in selected)
                for reason in ("invalid_blur", "l_zero", "zero_displacement")
            },
        }
    return result


def build_groups(pairs):
    groups = {"overall": pairs}
    bins = {
        "half_length_l0": lambda p: p["l_valid"] and p["l"] == 0,
        "half_length_0_2": lambda p: p["l_valid"] and 0 < p["l"] <= 2,
        "half_length_2_5": lambda p: p["l_valid"] and 2 < p["l"] <= 5,
        "half_length_5_10": lambda p: p["l_valid"] and 5 < p["l"] <= 10,
        "half_length_gt10": lambda p: p["l_valid"] and p["l"] > 10,
    }
    groups.update({name: [p for p in pairs if condition(p)] for name, condition in bins.items()})
    for match in sorted({p["match"] for p in pairs}):
        groups[f"match_{match}"] = [p for p in pairs if p["match"] == match]
    for width, height in sorted({(p["width"], p["height"]) for p in pairs}):
        resolution = [p for p in pairs if (p["width"], p["height"]) == (width, height)]
        groups[f"resolution_{width}x{height}"] = resolution
        groups.update({f"resolution_{width}x{height}__{name}":
                       [p for p in resolution if condition(p)]
                       for name, condition in bins.items()})
    return groups


def refresh_summary_from_pairs(summary_path, pairs_path):
    summary_path, pairs_path = Path(summary_path), Path(pairs_path)
    with pairs_path.open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    for pair in pairs:
        for key in ("delta", "previous_frame", "Frame", "width", "height"):
            pair[key] = int(pair[key])
        for key in ("theta", "l", "dx", "dy", "displacement_l2"):
            pair[key] = float(pair[key])
        pair["angle_error_deg"] = (float(pair["angle_error_deg"])
                                   if pair["angle_error_deg"] else None)
        pair["blur_valid"] = pair["axis_unavailable_reason"] != "invalid_blur"
        pair["l_valid"] = math.isfinite(pair["l"]) and pair["l"] >= 0
    summary = json.loads(summary_path.read_text())
    summary["groups"] = {name: pair_summary(group) for name, group in build_groups(pairs).items()}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def analyze(data_root, output):
    data_root, output = Path(data_root), Path(output)
    metadata_path = data_root / "official_metadata/video_label_validation.tsv"
    manifest = []
    with metadata_path.open(newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            match = int(Path(raw["video"]).parts[0])
            split = split_for(match)
            if split is None:
                continue
            row = dict(raw, split=split, match=f"{match:02d}",
                       rally=Path(raw["video"]).stem,
                       label_csv=f"labels/all_csv_annotations/{match:02d}_csv_{Path(raw['video']).stem}.csv")
            row["width"], row["height"] = int(row["width"]), int(row["height"])
            manifest.append(row)
    manifest.sort(key=lambda r: (r["split"] != "train", r["match"], r["rally"]))

    rows, invalid_centers, invalid_blur = [], [], []
    for video in manifest:
        if video["split"] != "train":
            continue
        with (data_root / video["label_csv"]).open(newline="") as handle:
            for raw in csv.DictReader(handle):
                frame, visibility = int(raw["Frame"]), int(raw["Visibility"])
                x, y, theta, length = (float(raw[k]) for k in ("X", "Y", "theta", "l"))
                center_valid = (visibility == 1 and math.isfinite(x) and math.isfinite(y)
                                and 0 <= x < video["width"] and 0 <= y < video["height"])
                blur_valid = math.isfinite(theta) and math.isfinite(length) and length >= 0
                identity = {"match": video["match"], "rally": video["rally"], "Frame": frame}
                if visibility not in (0, 1) or (visibility == 1 and not center_valid):
                    invalid_centers.append(dict(identity, Visibility=visibility, X=x, Y=y,
                                                reason="invalid_visibility_or_visible_center"))
                if not blur_valid:
                    invalid_blur.append(dict(identity, theta=theta, l=length,
                                             reason="nonfinite_theta_or_negative_nonfinite_l"))
                rows.append(dict(identity, game=video["match"], clip=video["rally"],
                                 original_frame_id=frame, visibility=visibility, x_raw=x, y_raw=y,
                                 theta=theta, l=length, center_valid=center_valid,
                                 blur_valid=blur_valid, l_valid=math.isfinite(length) and length >= 0,
                                 label_state=("located" if center_valid else
                                              "not_visible" if visibility == 0 else "invalid"),
                                 width=video["width"], height=video["height"],
                                 avg_frame_rate=video["avg_frame_rate"]))

    pairs = []
    for delta in (1, 2):
        for previous, current in center_pairs(rows, delta):
            dx, dy = current["x_raw"] - previous["x_raw"], current["y_raw"] - previous["y_raw"]
            displacement = math.hypot(dx, dy)
            error = axis_error(current["theta"], current["l"], dx, dy) if current["blur_valid"] else None
            if not current["blur_valid"]:
                reason = "invalid_blur"
            elif current["l"] == 0:
                reason = "l_zero"
            elif displacement == 0:
                reason = "zero_displacement"
            else:
                reason = ""
            pairs.append({"match": current["match"], "rally": current["rally"], "delta": delta,
                          "previous_frame": previous["Frame"], "Frame": current["Frame"],
                          "theta": current["theta"], "l": current["l"], "dx": dx, "dy": dy,
                          "displacement_l2": displacement,
                          "displacement_angle_deg": (math.degrees(math.atan2(dy, dx))
                                                       if displacement else None),
                          "angle_error_deg": error, "axis_unavailable_reason": reason,
                          "width": current["width"], "height": current["height"],
                          "avg_frame_rate": current["avg_frame_rate"],
                          "blur_valid": current["blur_valid"], "l_valid": current["l_valid"]})

    groups = build_groups(pairs)

    summary = {
        "scope": "match00-17 center CSV train only; match18-21 metadata only; match22-25 excluded",
        "geometry": "current theta axis versus p_t-p_(t-delta), modulo 180 degrees",
        "limit": "axis agreement is conditional association, not signed velocity or motion-label correctness",
        "counts": {
            "train_rallies": sum(r["split"] == "train" for r in manifest),
            "val_rallies": sum(r["split"] == "val" for r in manifest),
            "train_rows": len(rows), "located_rows": sum(r["center_valid"] for r in rows),
            "not_visible_rows": sum(r["visibility"] == 0 for r in rows),
            "invalid_center_rows": len(invalid_centers), "invalid_blur_rows": len(invalid_blur),
            "train_rallies_by_avg_frame_rate": dict(Counter(
                r["avg_frame_rate"] for r in manifest if r["split"] == "train")),
        },
        "groups": {name: pair_summary(group) for name, group in groups.items()},
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_fields = ("split", "match", "rally", "video", "label_csv", "status", "codec",
                       "width", "height", "r_frame_rate", "avg_frame_rate", "ffprobe_frames",
                       "duration", "mid_rows", "mid_min", "mid_max", "mid_contiguous")
    with (output / "development_rallies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    pair_fields = ("match", "rally", "delta", "previous_frame", "Frame", "theta", "l", "dx", "dy",
                   "displacement_l2", "displacement_angle_deg", "angle_error_deg",
                   "axis_unavailable_reason", "width", "height", "avg_frame_rate")
    with (output / "pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, pair_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    for name, records in (("invalid_centers.csv", invalid_centers),
                          ("invalid_blur.csv", invalid_blur)):
        if records:
            with (output / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, records[0].keys())
                writer.writeheader()
                writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/blurball")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.data_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
