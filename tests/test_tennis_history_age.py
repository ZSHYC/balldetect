"""历史年龄诊断：真实身份、固定cohort及不随s膨胀的旧半径。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_tennis_history_age import build_windows, describe_window, summarize
from ballmotion.tennis import grid_targets


def test_history_age():
    frames = []
    for clip in ("Clip1", "Clip2"):
        for frame in range(19):
            if clip == "Clip2" and frame == 8:
                continue
            vc = 0 if clip == "Clip1" and frame in (0, 1, 9) else 1
            if clip == "Clip1" and frame == 18:
                vc = 2
            frames.append({"game": "game1", "clip": clip, "original_frame_id": f"{frame:04d}",
                           "visibility_raw": vc, "label_state": "absent" if vc == 0 else "located",
                           "x_raw": 60 if frame == 8 else 100 if frame == 10 else 0,
                           "y_raw": 0})
    lookup = {(r["clip"], int(r["original_frame_id"])): i for i, r in enumerate(frames)}
    targets = [lookup[("Clip1", f)] for f in (2, 16, 17, 18)] + [lookup[("Clip2", 16)]]
    windows, common = build_windows(frames, targets)
    assert [len(windows[s]) for s in (1, 2, 4, 8)] == [5, 4, 3, 3]
    assert common == targets[1:4]
    assert [int(frames[i]["original_frame_id"]) for i in windows[8][common[0]]] == [0, 8, 16]
    cells = grid_targets([[r["x_raw"], r["y_raw"]] for r in frames],
                         [r["label_state"] == "located" for r in frames], (36, 64))
    records = [describe_window(frames, cells, windows[8][i], 8) for i in common]
    assert records[0]["near_native_chebyshev"] == 3
    assert records[0]["near_radius"] == 2 and records[0]["far_radius"] == 4
    assert records[0]["vc1_support"] == "visible_history_uncovered"
    assert records[1]["vc1_support"] == "no_visible_history"
    assert records[2]["vc1_support"] == "current_not_vc1"
    stats = summarize(records)
    assert stats["current_vc1"] == 2
    assert stats["vc1_support_counts"] == {"no_visible_history": 1,
                                           "visible_history_covered": 0,
                                           "visible_history_uncovered": 1}
    assert stats["near"]["all_located"]["n"] == 2
    assert stats["near"]["both_vc1"]["n"] == 1
    assert stats["far"]["all_located"]["n"] == 1
    assert stats["far"]["both_vc1"]["n"] == 0
    assert stats["near"]["visibility_transitions"]["1->0"] == 1
    assert summarize([])["current_vc1"] == 0


if __name__ == "__main__":
    test_history_age()
    print("history age check passed")
