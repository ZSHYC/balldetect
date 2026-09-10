"""时序特征缓存只存真实唯一帧，并严格复用兼容特征。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cache_tennis_features import FEATURE_CONFIG_KEYS, open_reuse_source, select_cache_rows


def row(game, clip, frame, state="located"):
    return dict(game=game, clip=clip, original_frame_id=str(frame), label_state=state,
                image_path=f"Dataset/{game}/{clip}/{frame:04}.jpg")


class TemporalCacheTest(unittest.TestCase):
    def test_targets_are_limited_by_split_and_frames_are_unique(self):
        rows = [row(game, "Clip1", frame, "unknown" if frame == 1 else "located")
                for game in ("game1", "game7") for frame in range(11)]
        frames, windows, skipped = select_cache_rows(rows, target_step=2, history=2, max_frames=4)
        self.assertEqual(windows.shape, (4, 3))
        self.assertEqual([frames[w[-1]]["game"] for w in windows], ["game1", "game1", "game7", "game7"])
        self.assertIn("unknown", [frames[i]["label_state"] for i in windows[0][:-1]])
        self.assertEqual(len(frames), len({tuple(r[k] for k in ("game", "clip", "original_frame_id"))
                                          for r in frames}))
        for window in windows:
            selected = [frames[i] for i in window]
            self.assertEqual([int(r["original_frame_id"]) for r in selected],
                             list(range(int(selected[-1]["original_frame_id"]) - 2,
                                        int(selected[-1]["original_frame_id"]) + 1)))
            self.assertEqual(len({(r["game"], r["clip"]) for r in selected}), 1)
        self.assertEqual(skipped, 2)

    def test_reuse_ignores_sampling_but_requires_feature_conditions_and_stage(self):
        config = {key: key for key in FEATURE_CONFIG_KEYS}
        config["storage_dtype"] = "float16"
        old_config = {**config, "frame_step": 8, "max_frames": 0, "saved_stages": [1]}
        old_rows = [row("game1", "Clip1", 0), row("game1", "Clip1", 8)]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            values = np.arange(8, dtype=np.float16).reshape(2, 1, 2, 2)
            np.save(cache / "stage1.npy", values)
            metadata = dict(config=old_config, frames=old_rows,
                            shapes=[[2, 1, 4, 4], [2, 1, 2, 2], [2, 1, 1, 1], [2, 1, 1, 1]],
                            fp16_conversion=[])
            (cache / "metadata.json").write_text(json.dumps(metadata))
            _, arrays, lookup = open_reuse_source(cache, config, [1])
            np.testing.assert_array_equal(arrays[1], values)
            self.assertEqual(lookup[("game1", "Clip1", 8, old_rows[1]["image_path"])], 1)
            _, arrays, lookup = open_reuse_source(cache, config, [0, 1])
            self.assertEqual((arrays, lookup), ({}, {}))
            incompatible = dict(config)
            incompatible["input_hw"] = [576, 1024]
            with self.assertRaisesRegex(ValueError, "input_hw"):
                open_reuse_source(cache, incompatible, [1])


if __name__ == "__main__":
    unittest.main()
