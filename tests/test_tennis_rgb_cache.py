import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts/cache_tennis_rgb.py"
spec = importlib.util.spec_from_file_location("tennis_rgb_cache", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TennisRgbCacheTest(unittest.TestCase):
    def test_unique_rgb_frames_preserve_source_order_and_windows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            colors = [(250, 10, 20), (5, 240, 30), (7, 40, 230), (90, 80, 70)]
            frames = []
            for index, color in enumerate(colors):
                name = f"frame{index}.png"
                Image.new("RGB", (9, 5), color).save(images / name)
                frames.append({"image_path": name, "original_frame_id": index})
            windows = [[0, 1, 2], [1, 2, 3]]
            source = root / "source"
            source.mkdir()
            config = {"data_root": str(images), "input_hw": [4, 6], "resize": "PIL bilinear RGB"}
            (source / "metadata.json").write_text(json.dumps({
                "config": config, "frames": frames, "windows": windows,
            }))

            output = root / "rgb"
            metadata = module.cache_rgb(source, output, workers=0, batch_size=2)
            rgb = np.load(output / "rgb.npy")

            self.assertEqual(rgb.shape, (4, 3, 4, 6))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertEqual(rgb[:, :, 0, 0].tolist(), [list(c) for c in colors])
            self.assertEqual(metadata["frames"], frames)
            self.assertEqual(metadata["windows"], windows)
            self.assertEqual(set(metadata), {"source_cache", "source_config", "frames", "windows",
                                             "shape", "input_hw", "dtype", "elapsed_seconds"})
            self.assertEqual(metadata["source_cache"], str(source.resolve()))
            for path in images.iterdir():
                path.unlink()
            self.assertEqual(module.cache_rgb(source, output, workers=0), metadata)
            with self.assertRaises(ValueError):
                module.cache_rgb(root / "different-source", output, workers=0)


if __name__ == "__main__":
    unittest.main()
