import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/analyze_blurball_axis.py"
spec = importlib.util.spec_from_file_location("blurball_axis", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_labels(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Frame", "Visibility", "X", "Y", "theta", "l"))
        writer.writerows(rows)


class BlurBallAxisTest(unittest.TestCase):
    def test_axis_geometry_and_development_read_boundary(self):
        self.assertEqual(module.axis_error(0, 2, -5, 0), 0)
        self.assertEqual(module.axis_error(90, 2, 0, 5), 0)
        self.assertEqual(module.axis_error(90, 2, 5, 0), 90)
        self.assertIsNone(module.axis_error(0, 0, 5, 0))
        self.assertIsNone(module.axis_error(0, 2, 0, 0))

        with TemporaryDirectory() as directory:
            root = Path(directory) / "blurball"
            metadata = root / "official_metadata/video_label_validation.tsv"
            metadata.parent.mkdir(parents=True)
            fields = ("video", "status", "codec", "width", "height", "r_frame_rate",
                      "avg_frame_rate", "ffprobe_frames", "duration", "mid_rows", "mid_min",
                      "mid_max", "mid_contiguous", "end_rows", "end_min", "end_max",
                      "end_contiguous")
            records = [
                ("00/rallies_videos/001.mp4", "OK", "h264", 100, 50, "25/1", "25/1", 4, 1, 4, 0, 3, True, 4, 0, 3, True),
                ("00/rallies_videos/002.mp4", "OK", "h264", 100, 50, "25/1", "25/1", 5, 1, 4, 0, 4, False, 4, 0, 4, False),
                ("00/rallies_videos/003.mp4", "OK", "h264", 200, 100, "25/1", "25/1", 2, 1, 2, 0, 1, True, 2, 0, 1, True),
                ("18/rallies_videos/001.mp4", "OK", "h264", 100, 50, "30/1", "30/1", 2, 1, 2, 0, 1, True, 2, 0, 1, True),
                ("22/rallies_videos/001.mp4", "OK", "h264", 100, 50, "30/1", "30/1", 2, 1, 2, 0, 1, True, 2, 0, 1, True),
            ]
            with metadata.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerows(records)
            labels = root / "labels/all_csv_annotations"
            write_labels(labels / "00_csv_001.csv", [
                (0, 1, 10, 10, 0, 2), (1, 1, 5, 10, 0, 2),
                (2, 1, 5, 15, 90, 2), (3, 1, 10, 15, 90, 2),
            ])
            write_labels(labels / "00_csv_002.csv", [
                (0, 1, 1, 1, 0, 0), (2, 1, 2, 1, 0, 2),
                (3, 1, 2, 1, 0, 2), (4, 1, 3, 1, 0, 0),
            ])
            write_labels(labels / "00_csv_003.csv", [
                (0, 1, 4, 4, 0, 2), (1, 1, 5, 4, "nan", 2),
            ])

            summary = module.analyze(root, Path(directory) / "out")

            overall = summary["groups"]["overall"]["1"]
            self.assertEqual(overall["position_pairs"], 6)
            self.assertEqual(overall["axis_pairs"], 3)
            self.assertEqual(overall["angle_error_median_p90"], [0.0, 72.0])
            self.assertEqual(overall["within_deg"]["15"], 2 / 3)
            self.assertEqual(overall["l_zero_pairs"], 1)
            self.assertEqual(overall["zero_displacement_pairs"], 1)
            self.assertEqual(overall["invalid_blur_pairs"], 1)
            self.assertEqual(overall["axis_unavailable_reasons"], {
                "invalid_blur": 1, "l_zero": 1, "zero_displacement": 1,
            })
            groups = summary["groups"]
            self.assertEqual(groups["resolution_100x50__half_length_0_2"]["1"]["position_pairs"], 4)
            self.assertEqual(groups["resolution_200x100__half_length_0_2"]["1"]["position_pairs"], 1)
            self.assertEqual(summary["counts"]["train_rallies"], 3)
            self.assertEqual(summary["counts"]["val_rallies"], 1)
            with (Path(directory) / "out/pairs.csv").open() as handle:
                pairs = list(csv.DictReader(handle))
            bad = [r for r in pairs if r["rally"] == "003" and r["delta"] == "1"]
            self.assertEqual(len(bad), 1)
            self.assertEqual(bad[0]["angle_error_deg"], "")
            refreshed = module.refresh_summary_from_pairs(
                Path(directory) / "out/summary.json", Path(directory) / "out/pairs.csv")
            self.assertEqual(refreshed, summary)


if __name__ == "__main__":
    unittest.main()
