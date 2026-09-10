"""固定预测原生块四子格 oracle 的几何边界。"""
import unittest

import numpy as np

from scripts.analyze_tennis_subcell import fixed_block_subcell_oracle


class SubcellDiagnosticTest(unittest.TestCase):
    def test_fixed_block_oracle_cases_and_absence(self):
        predicted = np.array([
            [4.5, 4.5],    # 原预测已命中，oracle 保持同一子格。
            [4.5, 4.5],    # 原预测未命中，同一原生块内另一子格可命中。
            [4.5, 4.5],    # GT 属于相邻原生块，但当前块仍有 8px 内候选。
            [4.5, 4.5],    # 当前块四子格都不可达。
            [24.5, 34.5],  # absence 行坐标不得改变。
        ])
        target = np.array([
            [5.0, 5.0],
            [14.5, 14.5],
            [19.5, 4.5],
            [40.0, 40.0],
            [np.nan, np.nan],
        ])
        located = np.array([True, True, True, True, False])

        oracle, pred_block, target_block = fixed_block_subcell_oracle(
            predicted, target, located)

        np.testing.assert_allclose(oracle[:4], [
            [4.5, 4.5], [14.5, 14.5], [14.5, 4.5], [14.5, 14.5]
        ])
        np.testing.assert_array_equal(oracle[4], predicted[4])
        np.testing.assert_array_equal(pred_block, [0, 0, 0, 0, 65])
        np.testing.assert_array_equal(target_block, [0, 0, 1, 130, -1])

        actual_error = np.linalg.norm(predicted[:4] - target[:4], axis=1)
        oracle_error = np.linalg.norm(oracle[:4] - target[:4], axis=1)
        np.testing.assert_array_equal(actual_error <= 8, [True, False, False, False])
        np.testing.assert_array_equal(oracle_error <= 8, [True, True, True, False])
        self.assertTrue(np.all(oracle_error <= actual_error))


if __name__ == "__main__":
    unittest.main()
