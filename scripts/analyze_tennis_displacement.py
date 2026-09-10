"""真实标签中心位移与方形搜索覆盖；不由中心标签虚构球尺寸或 rho。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.tennis import read_frames, center_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, excluded = read_frames(ROOT / "data/tracknet_tennis", set(range(1, 8)))
    results = {"coordinate_units": "original 1280x720 pixels", "excluded": len(excluded),
               "interpretation": "label-center displacement includes camera motion and annotation conventions; not optical-flow GT"}
    for split, games in (("train", set(range(1, 7))), ("val", {7})):
        subset = [r for r in rows if int(r["game"][4:]) in games]
        results[split] = {}
        for delta in (1, 2, 4):
            pairs = list(center_pairs(subset, delta))
            displacement = np.array([[b["x_raw"] - a["x_raw"], b["y_raw"] - a["y_raw"]] for a, b in pairs])
            l2 = np.linalg.norm(displacement, axis=1)
            linf = np.abs(displacement).max(axis=1)
            easy = np.array([a["visibility_raw"] == b["visibility_raw"] == 1 for a, b in pairs])
            results[split][str(delta)] = {"pairs": len(pairs), "target_frames": len(subset),
                "easy_both_endpoints": int(easy.sum()),
                "l2_p50_p90_p99_max": [float(x) for x in np.quantile(l2, [.5, .9, .99, 1])],
                "easy_l2_p50_p90_p99_max": [float(x) for x in np.quantile(l2[easy], [.5, .9, .99, 1])],
                "square_radius_original_px_coverage": {str(r): float(np.mean(linf <= r)) for r in (20, 40, 80, 160)},
                "limit": "双端合法位置标签条件下的几何覆盖，不包括缺标/无球；不等于对应识别成功率"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
