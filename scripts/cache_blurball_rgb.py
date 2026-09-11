"""将 BlurBall development rallies 解码为原生三帧 RGB 缓存。"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_HW = (288, 512)
DEVELOPMENT_MATCHES = tuple(range(22))
TIME_BASE_RE = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
FRAME_RE = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b")


def _read_manifest(manifest):
    with Path(manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("BlurBall manifest 为空")
    seen = set()
    for row in rows:
        match, rally = int(row["match"]), row["rally"]
        identity = (match, rally)
        if match not in DEVELOPMENT_MATCHES:
            raise ValueError(f"manifest 包含禁止进入 development 缓存的 match{match:02d}")
        expected_split = "train" if match <= 17 else "val"
        if row["split"] != expected_split:
            raise ValueError(f"match{match:02d}/{rally} split 应为 {expected_split}")
        if identity in seen:
            raise ValueError(f"manifest 重复 rally: match{match:02d}/{rally}")
        seen.add(identity)
    return rows


def prepare_cache_source(data_root, manifest, selected_rallies=None):
    data_root, manifest = Path(data_root), Path(manifest)
    rows = _read_manifest(manifest)
    requested = ({(int(match), str(rally)) for match, rally in selected_rallies}
                 if selected_rallies is not None else None)
    selected = [row for row in rows
                if requested is None or (int(row["match"]), row["rally"]) in requested]
    found = {(int(row["match"]), row["rally"]) for row in selected}
    if requested is not None and found != requested:
        missing = sorted(requested - found)
        raise ValueError(f"manifest 中没有所选 rallies: {missing}")

    selected.sort(key=lambda row: (int(row["match"]), row["rally"]))
    frames, windows = [], []
    train_targets = val_targets = boundary_excluded = 0
    for rally_row in selected:
        match = int(rally_row["match"])
        rally = rally_row["rally"]
        width, height = int(rally_row["width"]), int(rally_row["height"])
        expected_rows = int(rally_row["mid_rows"])
        if (rally_row["status"] != "OK" or rally_row["mid_contiguous"].lower() != "true"
                or int(rally_row["mid_min"]) != 0
                or int(rally_row["mid_max"]) != expected_rows - 1):
            raise ValueError(f"match{match:02d}/{rally} manifest 未确认连续中心标签")
        video_path = Path("raw") / rally_row["video"]
        label_path = Path(rally_row["label_csv"])
        if not (data_root / video_path).is_file():
            raise FileNotFoundError(data_root / video_path)
        if not (data_root / label_path).is_file():
            raise FileNotFoundError(data_root / label_path)

        with (data_root / label_path).open(newline="") as handle:
            labels = list(csv.DictReader(handle))
        if len(labels) != expected_rows:
            raise ValueError(
                f"match{match:02d}/{rally} 中心标签 {len(labels)} 行 != manifest {expected_rows}")
        start = len(frames)
        for expected_frame, raw in enumerate(labels):
            frame = int(raw["Frame"])
            visibility = int(raw["Visibility"])
            x, y = float(raw["X"]), float(raw["Y"])
            theta, half_length = float(raw["theta"]), float(raw["l"])
            if frame != expected_frame:
                raise ValueError(
                    f"match{match:02d}/{rally} Frame={frame}，预期 {expected_frame}")
            if visibility not in (0, 1):
                raise ValueError(
                    f"match{match:02d}/{rally}/Frame{frame} Visibility={visibility}")
            if (visibility == 1 and
                    (not math.isfinite(x) or not math.isfinite(y)
                     or not 0 <= x < width or not 0 <= y < height)):
                raise ValueError(
                    f"match{match:02d}/{rally}/Frame{frame} 可见中心非法: ({x}, {y})")
            frames.append({
                "game": f"match{match:02d}", "clip": rally,
                "match": f"{match:02d}", "rally": rally,
                "original_frame_id": frame, "split": rally_row["split"],
                "visibility_raw": visibility, "x_raw": x, "y_raw": y,
                "theta_raw": theta, "l_raw": half_length,
                "width": width, "height": height,
                "label_state": "located" if visibility == 1 else "not_visible",
                "video_path": str(video_path), "label_path": str(label_path),
            })
        rally_windows = [[index - 2, index - 1, index]
                         for index in range(start + 2, len(frames))]
        windows.extend(rally_windows)
        if rally_row["split"] == "train":
            train_targets += len(rally_windows)
        else:
            val_targets += len(rally_windows)
        boundary_excluded += min(2, len(labels))

    selected_ids = [[f"{int(row['match']):02d}", row["rally"]] for row in selected]
    config = {
        "dataset": "blurball", "data_root": str(data_root.resolve()),
        "manifest": str(manifest.resolve()), "videos_root": str((data_root / "raw").resolve()),
        "labels_root": str((data_root / "labels/all_csv_annotations").resolve()),
        "matches": sorted({int(row["match"]) for row in selected}),
        "rallies": selected_ids, "input_hw": list(INPUT_HW), "history": 2,
        "target_step": 1, "resize": "FFmpeg bilinear RGB",
        "annotation": "streak midpoint",
    }
    return {"config": config, "rallies": selected, "frames": frames, "windows": windows,
            "train_targets": train_targets, "val_targets": val_targets,
            "boundary_excluded_targets": boundary_excluded}


def _read_exact(stream, size):
    parts, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _decode_rally(ffmpeg, video, expected_frames, rgb, offset, log):
    log.seek(0, 2)
    log_start = log.tell()
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-copyts", "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"scale={INPUT_HW[1]}:{INPUT_HW[0]}:flags=bilinear,showinfo=checksum=0",
        "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
    frame_bytes = INPUT_HW[0] * INPUT_HW[1] * 3
    decoded = 0
    try:
        for decoded in range(expected_frames):
            payload = _read_exact(process.stdout, frame_bytes)
            if len(payload) != frame_bytes:
                raise ValueError(f"FFmpeg 仅解码 {decoded}/{expected_frames} 帧: {video}")
            pixels = np.frombuffer(payload, dtype=np.uint8).reshape(*INPUT_HW, 3)
            rgb[offset + decoded] = pixels.transpose(2, 0, 1)
        if process.stdout.read(1):
            raise ValueError(f"FFmpeg 解码帧数超过标签 {expected_frames}: {video}")
    finally:
        process.stdout.close()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    log.flush()
    log.seek(log_start)
    segment = log.read().decode("utf-8", errors="replace")
    bases = TIME_BASE_RE.findall(segment)
    if len(set(bases)) != 1:
        raise ValueError(f"无法唯一解析 FFmpeg showinfo time_base: {video}")
    time_base = [int(value) for value in bases[0]]
    pts_rows = [(int(n), int(pts)) for n, pts in FRAME_RE.findall(segment)]
    if [n for n, _ in pts_rows] != list(range(expected_frames)):
        raise ValueError(f"FFmpeg showinfo 帧序号与标签不一致: {video}")
    return time_base, [pts for _, pts in pts_rows]


def cache_blurball_rgb(data_root, manifest, output, selected_rallies=None, ffmpeg="ffmpeg"):
    data_root, manifest, output = Path(data_root), Path(manifest), Path(output)
    source = prepare_cache_source(data_root, manifest, selected_rallies)
    meta_path = output / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["config"] != source["config"]:
            raise ValueError("已有 BlurBall RGB 缓存配置不同；请使用新目录")
        rgb = np.load(output / "rgb.npy", mmap_mode="r")
        if list(rgb.shape) != metadata["shape"] or rgb.dtype != np.uint8:
            raise ValueError("已有 BlurBall RGB 数组与 metadata 不一致")
        return metadata
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录含未完成或未知产物；请使用空目录")

    output.mkdir(parents=True, exist_ok=True)
    rgb = np.lib.format.open_memmap(
        output / "rgb.npy", mode="w+", dtype=np.uint8,
        shape=(len(source["frames"]), 3, *INPUT_HW))
    started = time.perf_counter()
    offset = 0
    with (output / "decode.log").open("a+b") as log:
        for rally in source["rallies"]:
            count = int(rally["mid_rows"])
            time_base, pts = _decode_rally(
                ffmpeg, data_root / "raw" / rally["video"], count, rgb, offset, log)
            for local_frame, point in enumerate(pts):
                row = source["frames"][offset + local_frame]
                row["pts"] = point
                row["time_base"] = time_base
                row["pts_seconds"] = point * time_base[0] / time_base[1]
            offset += count
            print(f"decoded match{int(rally['match']):02d}/{rally['rally']}: {count} frames", flush=True)
    rgb.flush()
    elapsed = time.perf_counter() - started
    metadata = {
        "config": source["config"], "frames": source["frames"],
        "windows": source["windows"], "shape": list(rgb.shape), "dtype": "uint8",
        "train_targets": source["train_targets"], "val_targets": source["val_targets"],
        "boundary_excluded_targets": source["boundary_excluded_targets"],
        "elapsed_seconds": elapsed,
        "timing_scope": "FFmpeg decode + bilinear resize + RGB mmap write/flush; excludes manifest/label parsing and metadata write",
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/blurball")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "outputs/blurball/development_axis/development_rallies.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = cache_blurball_rgb(args.data_root, args.manifest, args.output)
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in ("frames", "windows")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
