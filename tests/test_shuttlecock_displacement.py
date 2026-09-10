import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/analyze_shuttlecock_displacement.py"
spec = importlib.util.spec_from_file_location("shuttle_motion", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Frame", "Visibility", "X", "Y"))
        writer.writerows(rows)


class ShuttlecockDisplacementTest(unittest.TestCase):
    def test_real_frame_pairs_and_read_boundary(self):
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "shuttlecock"
            verification = root / "verification.tsv"
            verification.parent.mkdir(parents=True)
            fields = ("video", "csv", "width", "height", "avg_frame_rate", "nb_frames",
                      "duration", "csv_rows", "first_frame", "last_frame", "contiguous",
                      "bad_visibility", "invisible_nonzero_xy")
            records = [
                ("Professional/match1/video/a.mp4", "Professional/match1/csv/a_ball.csv", 100, 50, "30/1", 8, 1, 6, 0, 6, 0, 0, 0),
                ("Amateur/match1/video/b.mp4", "Amateur/match1/csv/b_ball.csv", 100, 50, "25/1", 2, 1, 2, 0, 1, 1, 0, 0),
                ("Professional/match21/video/val.mp4", "Professional/match21/csv/missing.csv", 100, 50, "30/1", 3, 1, 3, 0, 2, 1, 0, 0),
                ("Test/match1/video/test.mp4", "Test/match1/csv/missing.csv", 100, 50, "30/1", 3, 1, 3, 0, 2, 1, 0, 0),
            ]
            with verification.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerows(records)

            write_csv(root / "original/TrackNetV2/Professional/match1/csv/a_ball.csv", [
                (0, 1, 10, 10), (1, 0, 0, 0), (2, 1, 20, 10),
                (4, 1, 30, 10), (5, 1, 100, 10), (6, 1, 40, 10),
            ])
            write_csv(root / "original/TrackNetV2/Amateur/match1/csv/b_ball.csv", [
                (0, 1, 1, 1), (1, 1, 4, 5),
            ])

            summary = module.analyze(root, temporary / "out")

            self.assertEqual(summary["counts"]["train_rallies"], 2)
            self.assertEqual(summary["counts"]["val_rallies"], 1)
            self.assertEqual(summary["counts"]["unlabeled_video_tail_frames"], 1)
            self.assertEqual(summary["train"]["overall"]["1"]["pairs"], 1)
            self.assertEqual(summary["train"]["overall"]["1"]["l2_p50_p90_p99_max"], [5.0] * 4)
            self.assertEqual(summary["train"]["overall"]["2"]["pairs"], 3)
            with (temporary / "out/invalid_labels.csv").open() as handle:
                invalid = list(csv.DictReader(handle))
            self.assertEqual([(r["rally"], r["Frame"]) for r in invalid], [("a", "5")])
            with (temporary / "out/development_rallies.csv").open() as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual([r["split"] for r in manifest], ["train", "train", "val"])


if __name__ == "__main__":
    unittest.main()
