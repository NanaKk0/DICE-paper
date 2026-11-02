import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_reanalysis import bool_column, norm_metrics, read_stored_force_labels, safe_path, tail_means
from reproduce import Archive, assert_close
from plots import same_window_tma_references


class MetricsTests(unittest.TestCase):
    def test_thermal_comparison_matches_method_and_temperature_window(self):
        refs = pd.DataFrame([
            dict(composition=c, method=m, T_low_K=303.15, T_high_K=t, linear_CTE_1e6_per_K=v)
            for c, m, t, v in [("8YSZ", "HT-XRD", 1073.15, 10.51),
                                ("10YSZ", "HT-XRD", 1073.15, 10.56),
                                ("8YSZ", "TMA", 1173.15, 10.31),
                                ("10YSZ", "TMA", 1173.15, 10.38)]])
        selected = same_window_tma_references(refs, ["8YSZ", "10YSZ"])
        self.assertEqual(selected.method.tolist(), ["TMA", "TMA"])
        np.testing.assert_allclose(selected.linear_CTE_1e6_per_K, [10.31, 10.38])

    def test_thermal_comparison_cannot_silently_use_other_window(self):
        refs = pd.DataFrame([dict(composition="8YSZ", method="TMA", T_low_K=303.15,
                                 T_high_K=1073.15, linear_CTE_1e6_per_K=10.51)])
        with self.assertRaises(ValueError):
            same_window_tma_references(refs, ["8YSZ"])

    def test_component_rmse_uses_three_components(self):
        self.assertAlmostEqual(norm_metrics([3, 4])["force_component_RMSE_eV_A"], np.sqrt(25 / 6))

    def test_p99_uses_linear_interpolation(self):
        self.assertAlmostEqual(norm_metrics([0, 10])["vector_P99_eV_A"], 9.9)

    def test_tail_count_ceil_not_round(self):
        np.testing.assert_allclose(tail_means([1, 2, 3, 4], [0.01, 26, 100]), [4, 3.5, 2.5])

    def test_no_negative_norms(self):
        with self.assertRaises(ValueError):
            norm_metrics([-1, 2])

    def test_tail_fraction_domain(self):
        for q in [0, -1, 101]:
            with self.assertRaises(ValueError):
                tail_means([1], [q])

    def test_membership_values_not_truthy_strings(self):
        np.testing.assert_array_equal(bool_column(pd.Series(["True", "False"])), [True, False])
        with self.assertRaises(ValueError):
            bool_column(pd.Series(["yes"]))

    def test_archive_path_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "DATA_REGISTRY.json").write_text(json.dumps({"schema": 1}))
            archive = Archive(root)
            with self.assertRaises(ValueError):
                archive.path("../outside")
            with self.assertRaises(ValueError):
                safe_path(root, root.parent)

    def test_assertion_rejects_bad_science(self):
        with self.assertRaises(ValueError):
            assert_close(1, 2, "test")

    def test_stored_label_properties_define_column_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.extxyz"
            path.write_text('1\nLattice="1 0 0 0 1 0 0 0 1" '
                            'Properties=species:S:1:forces:R:3:pos:R:3\nO 1 2 3 4 5 6\n')
            frames = read_stored_force_labels(path)
            self.assertEqual(frames[0]["species"], ["O"])
            np.testing.assert_array_equal(frames[0]["forces"], [[1, 2, 3]])
            np.testing.assert_array_equal(frames[0]["pos"], [[4, 5, 6]])

    def test_stored_label_truncation_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.extxyz"
            path.write_text('2\nLattice="1 0 0 0 1 0 0 0 1" '
                            'Properties=species:S:1:pos:R:3:forces:R:3\nO 0 0 0 1 2 3\n')
            with self.assertRaises(ValueError):
                read_stored_force_labels(path)


if __name__ == "__main__":
    unittest.main()
