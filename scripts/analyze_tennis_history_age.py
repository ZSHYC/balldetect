"""共同逐帧目标上的历史年龄、可见性与固定近R2/远R4几何支持。"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.tennis import grid_targets

GAPS = (1, 2, 4, 8)
SUPPORT_GROUPS = ("no_visible_history", "visible_history_covered", "visible_history_uncovered")


def build_windows(frames, targets):
    lookup = {(r["game"], r["clip"], int(r["original_frame_id"])): i
              for i, r in enumerate(frames)}
    windows = {s: {} for s in GAPS}
    for i in targets:
        row = frames[i]
        frame = int(row["original_frame_id"])
        for s in GAPS:
            indices = [lookup.get((row["game"], row["clip"], frame - age)) for age in (2*s, s, 0)]
            if all(index is not None for index in indices):
                windows[s][i] = indices
    common = [i for i in targets if all(i in windows[s] for s in GAPS)]
    return windows, common


def describe_window(frames, cells, window, gap):
    current = frames[window[-1]]
    record = {"split": "val" if current["game"] == "game7" else "train",
              "game": current["game"], "clip": current["clip"], "s": gap,
              "current_original_frame_id": current["original_frame_id"],
              "current_visibility": current["visibility_raw"],
              "current_label_state": current["label_state"]}
    visible, covered = False, False
    for slot, index, age, radius in (("near", window[1], gap, 2), ("far", window[0], 2*gap, 4)):
        previous = frames[index]
        located = current["label_state"] == previous["label_state"] == "located"
        both_vc1 = located and current["visibility_raw"] == previous["visibility_raw"] == 1
        displacement, distance = None, None
        if located:
            displacement = math.hypot(current["x_raw"] - previous["x_raw"],
                                      current["y_raw"] - previous["y_raw"])
            a, b = int(cells[window[-1]]), int(cells[index])
            distance = max(abs(a // 64 - b // 64), abs(a % 64 - b % 64))
        within = distance <= radius if located else None
        record.update({f"{slot}_original_frame_id": previous["original_frame_id"],
                       f"{slot}_age_frames": age, f"{slot}_radius": radius,
                       f"{slot}_visibility": previous["visibility_raw"],
                       f"{slot}_label_state": previous["label_state"],
                       f"{slot}_both_located": located, f"{slot}_both_vc1": both_vc1,
                       f"{slot}_displacement_px": displacement,
                       f"{slot}_native_chebyshev": distance, f"{slot}_covered": within})
        visible |= both_vc1
        covered |= both_vc1 and within
    record["vc1_support"] = ("current_not_vc1" if current["visibility_raw"] != 1 else
                              "no_visible_history" if not visible else
                              "visible_history_covered" if covered else "visible_history_uncovered")
    return record


def distribution(values):
    if not values:
        return {key: None for key in ("p50", "p90", "p99", "max")}
    return dict(zip(("p50", "p90", "p99", "max"),
                    map(float, np.quantile(values, (.5, .9, .99, 1.)))))


def summarize(records):
    current_vc1 = sum(r["current_visibility"] == 1 for r in records)
    counts = {group: sum(r["vc1_support"] == group for r in records) for group in SUPPORT_GROUPS}
    assert sum(counts.values()) == current_vc1
    visible = counts["visible_history_covered"] + counts["visible_history_uncovered"]
    result = {"targets": len(records), "current_vc1": current_vc1,
              "vc1_support_counts": counts,
              "fractions_of_current_vc1": {k: v / current_vc1 if current_vc1 else None
                                            for k, v in counts.items()},
              "coverage_given_visible_history": counts["visible_history_covered"] / visible if visible else None,
              "both_histories_vc1_when_current_vc1": sum(r["near_both_vc1"] and r["far_both_vc1"]
                                                         for r in records)}
    for slot, radius in (("near", 2), ("far", 4)):
        data = {"age_frames": records[0][f"{slot}_age_frames"] if records else None,
                "radius_native_cells": radius,
                "visibility_transitions": dict(sorted(Counter(
                    f'{r["current_visibility"]}->{r[f"{slot}_visibility"]}' for r in records).items()))}
        for group, condition in (("all_located", "both_located"), ("both_vc1", "both_vc1")):
            pairs = [r for r in records if r[f"{slot}_{condition}"]]
            within = sum(r[f"{slot}_covered"] for r in pairs)
            data[group] = {"n": len(pairs), "within_old_radius": within,
                           "outside_old_radius": len(pairs) - within,
                           "coverage": within / len(pairs) if pairs else None,
                           "l2_original_px": distribution([r[f"{slot}_displacement_px"] for r in pairs]),
                           "native_chebyshev": distribution([r[f"{slot}_native_chebyshev"] for r in pairs])}
        result[slot] = data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=Path("data/cache/tennis/rgb_512x288_all_h2/metadata.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "summary.json").exists() or (args.output / "windows.csv").exists():
        raise ValueError("输出已存在；保留已有实验记录")
    started = time.perf_counter()
    metadata = json.loads(args.metadata.read_text())
    frames = metadata["frames"]
    expected = {"target_step": 1, "history": 2, "input_hw": [288, 512]}
    if any(metadata["config"].get(k) != v for k, v in expected.items()):
        raise ValueError("需要原全量step1/history2的512×288缓存元数据")
    if (metadata["config"]["games"] != list(range(1, 8))
            or set(r["game"] for r in frames) != {f"game{i}" for i in range(1, 8)}):
        raise ValueError("本诊断只支持games1–7的完整开发缓存")
    if metadata["excluded_label_rows"] != 0:
        raise ValueError("元数据曾按标签删源帧；需先从完整manifest恢复真实历史可用性")
    if {(r["width"], r["height"]) for r in frames} != {(1280, 720)}:
        raise ValueError("需要既定1280×720原始坐标系")
    targets = [w[-1] for w in metadata["windows"]]
    windows, common = build_windows(frames, targets)
    if [windows[1].get(i) for i in targets] != metadata["windows"]:
        raise ValueError("s=1自然窗口未恢复原合法目标及真实时序")
    cells = grid_targets([[r["x_raw"], r["y_raw"]] for r in frames],
                         [r["label_state"] == "located" for r in frames], (36, 64))
    records = [describe_window(frames, cells, windows[s][i], s) for s in GAPS for i in common]
    result = {"metadata": str(args.metadata), "gaps": GAPS, "native_grid_hw": [36, 64],
              "near_far_radius": [2, 4], "original_hw": [720, 1280],
              "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                         text=True).strip(),
              "meaning": "fixed original target cadence; only history age changes; GT support, not correspondence, receptive field, rho, output FPS or physical speed",
              "visibility_transition_direction": "current -> history", "splits": {}}
    for split in ("train", "val"):
        selected = [i for i in targets if (frames[i]["game"] == "game7") == (split == "val")]
        common_split = [i for i in common if (frames[i]["game"] == "game7") == (split == "val")]
        clips = sorted({frames[i]["game"] + "/" + frames[i]["clip"] for i in selected})
        natural_counts = {str(s): sum(i in windows[s] for i in selected) for s in GAPS}
        data = {"original_targets": len(selected), "natural_targets_by_gap": natural_counts,
                "common_targets": len(common_split), "removed_from_original": len(selected) - len(common_split),
                "cohort_by_clip": {}, "by_gap": {}}
        for clip in clips:
            old = [i for i in selected if frames[i]["game"] + "/" + frames[i]["clip"] == clip]
            data["cohort_by_clip"][clip] = {
                "original": len(old), "natural_by_gap": {str(s): sum(i in windows[s] for i in old) for s in GAPS},
                "common": sum(frames[i]["game"] + "/" + frames[i]["clip"] == clip for i in common_split)}
        for s in GAPS:
            group = [r for r in records if r["split"] == split and r["s"] == s]
            data["by_gap"][str(s)] = summarize(group)
            if split == "val":
                data["by_gap"][str(s)]["by_clip"] = {
                    clip: summarize([r for r in group if r["game"] + "/" + r["clip"] == clip]) for clip in clips}
        result["splits"][split] = data
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "windows.csv").open("w", newline="") as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["timing_scope"] = "metadata read, exact-window construction, geometry, aggregation and CSV write; no RGB/model access"
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for split, data in result["splits"].items():
        print(json.dumps({"split": split, "natural": data["natural_targets_by_gap"],
                          "common_targets": data["common_targets"],
                          "by_gap": {s: {k: v[k] for k in ("current_vc1", "vc1_support_counts")}
                                     for s, v in data["by_gap"].items()}}), flush=True)


if __name__ == "__main__":
    main()
