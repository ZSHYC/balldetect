import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts/cache_tennis_full_rgb.py"
spec = importlib.util.spec_from_file_location("tennis_full_rgb", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TennisFullRgbTest(unittest.TestCase):
    def test_real_windows_reuse_and_unique_decode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tennis"
            fields = ("game", "clip", "original_frame_id", "image_path", "label_path",
                      "label_record_count", "visibility_raw", "x_raw", "y_raw",
                      "trajectory_status_raw", "width", "height", "wasb_split",
                      "tracknetv4_game_split")
            rows = []
            for game, clip, count in (("game1", "Clip1", 4), ("game1", "Clip2", 3),
                                      ("game7", "Clip3", 3)):
                for frame in range(count):
                    path = f"images/{game}_{clip}_{frame}.png"
                    visibility = 0 if (clip, frame) == ("Clip1", 1) else 1
                    rows.append((game, clip, f"{frame:04d}", path, "Label.csv", 1, visibility,
                                 0 if not visibility else 10 + frame, 0 if not visibility else 20,
                                 0, 1280, 720, "train", "test"))
                    if (clip, frame) != ("Clip1", 0):
                        image = data / path
                        image.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (9, 5), (frame + 1, 20, 30)).save(image)
            data.mkdir(exist_ok=True)
            with (data / "frames.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)

            reuse = root / "reuse"
            reuse.mkdir()
            reused_frame = dict(zip(fields, rows[0]))
            reused_pixels = np.lib.format.open_memmap(reuse / "rgb.npy", mode="w+", dtype=np.uint8,
                                                       shape=(1, 3, 288, 512))
            reused_pixels[:] = np.asarray([200, 100, 50], dtype=np.uint8)[:, None, None]
            reused_pixels.flush()
            source_config = {"data_root": str(data.resolve()), "input_hw": [288, 512],
                             "resize": "PIL bilinear RGB"}
            (reuse / "metadata.json").write_text(json.dumps({
                "source_config": source_config, "frames": [reused_frame],
                "shape": [1, 3, 288, 512], "dtype": "uint8",
            }))

            output = root / "full"
            metadata = module.cache_full_rgb(data, output, reuse, workers=0, batch_size=2)
            rgb = np.load(output / "rgb.npy")

            self.assertEqual(metadata["shape"], [10, 3, 288, 512])
            self.assertEqual(metadata["reused_frames"], 1)
            self.assertEqual(metadata["decoded_frames"], 9)
            self.assertEqual(rgb[0, :, 0, 0].tolist(), [200, 100, 50])
            self.assertEqual(rgb[3, :, 0, 0].tolist(), [4, 20, 30])
            self.assertEqual(metadata["input_hw"], [288, 512])
            self.assertEqual(metadata["config"]["input_hw"], [288, 512])
            targets = [(metadata["frames"][w[-1]]["clip"],
                        int(metadata["frames"][w[-1]]["original_frame_id"]))
                       for w in metadata["windows"]]
            self.assertEqual(targets, [("Clip1", 2), ("Clip1", 3), ("Clip2", 2), ("Clip3", 2)])
            first_window = metadata["windows"][0]
            self.assertEqual([int(metadata["frames"][i]["visibility_raw"]) for i in first_window],
                             [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
