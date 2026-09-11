import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/cache_blurball_rgb.py"


def write_labels(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Frame", "Visibility", "X", "Y", "theta", "l"))
        writer.writerows(rows)


def write_manifest(path, records):
    fields = ("split", "match", "rally", "video", "label_csv", "status", "codec",
              "width", "height", "r_frame_rate", "avg_frame_rate", "ffprobe_frames",
              "duration", "mid_rows", "mid_min", "mid_max", "mid_contiguous")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(records)


class BlurBallCacheTest(unittest.TestCase):
    def test_source_rows_preserve_rallies_and_not_visible_frames(self):
        self.assertTrue(SCRIPT.exists(), "BlurBall cache entry point is missing")
        spec = importlib.util.spec_from_file_location("blurball_cache", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with TemporaryDirectory() as directory:
            root = Path(directory) / "blurball"
            manifest = Path(directory) / "development_rallies.csv"
            records = [
                {"split": "train", "match": "00", "rally": "001",
                 "video": "00/rallies_videos/001.mp4",
                 "label_csv": "labels/all_csv_annotations/00_csv_001.csv", "status": "OK",
                 "codec": "h264", "width": 100, "height": 50, "r_frame_rate": "25/1",
                 "avg_frame_rate": "25/1", "ffprobe_frames": 4, "duration": 0.16,
                 "mid_rows": 4, "mid_min": 0, "mid_max": 3, "mid_contiguous": "True"},
                {"split": "val", "match": "18", "rally": "007",
                 "video": "18/rallies_videos/007.mp4",
                 "label_csv": "labels/all_csv_annotations/18_csv_007.csv", "status": "OK",
                 "codec": "h264", "width": 80, "height": 40, "r_frame_rate": "30/1",
                 "avg_frame_rate": "30/1", "ffprobe_frames": 3, "duration": 0.1,
                 "mid_rows": 3, "mid_min": 0, "mid_max": 2, "mid_contiguous": "True"},
            ]
            write_manifest(manifest, records)
            for record in records:
                video = root / "raw" / record["video"]
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()
            write_labels(root / records[0]["label_csv"], [
                (0, 1, 10.5, 11.5, 12.25, 3.5),
                (1, 0, -1, -1, 91.0, 7.0),
                (2, 0, -1, -1, 37.0, 4.25),
                (3, 1, 20.5, 21.5, 73.0, 9.0),
            ])
            write_labels(root / records[1]["label_csv"], [
                (0, 0, -1, -1, 3.0, 1.0),
                (1, 1, 30.0, 20.0, 4.0, 2.0),
                (2, 1, 31.0, 21.0, 5.0, 2.5),
            ])

            source = module.prepare_cache_source(root, manifest)

            self.assertEqual(source["windows"], [[0, 1, 2], [1, 2, 3], [4, 5, 6]])
            self.assertEqual(source["train_targets"], 2)
            self.assertEqual(source["val_targets"], 1)
            self.assertEqual(source["boundary_excluded_targets"], 4)
            self.assertEqual([row["original_frame_id"] for row in source["frames"]],
                             [0, 1, 2, 3, 0, 1, 2])
            self.assertEqual(source["frames"][2]["label_state"], "not_visible")
            self.assertEqual(source["frames"][2]["visibility_raw"], 0)
            self.assertEqual(source["frames"][2]["theta_raw"], 37.0)
            self.assertEqual(source["frames"][2]["l_raw"], 4.25)
            self.assertEqual(source["frames"][3]["x_raw"], 20.5)
            self.assertEqual(source["frames"][3]["y_raw"], 21.5)
            self.assertEqual(source["frames"][4]["game"], "match18")
            self.assertEqual(source["frames"][4]["clip"], "007")

            forbidden = Path(directory) / "with_test.csv"
            write_manifest(forbidden, records + [{**records[1], "split": "test", "match": "22"}])
            with self.assertRaisesRegex(ValueError, "match22"):
                module.prepare_cache_source(root, forbidden)


if __name__ == "__main__":
    unittest.main()
