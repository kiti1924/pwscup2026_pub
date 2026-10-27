import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pwscup_public.cli import calibrate_mia, integrate_aia, write_new
from pwscup_public.kernels import priority, reassign, transform

ROOT = Path(__file__).resolve().parents[1]


class PublicOperations(unittest.TestCase):
    def test_event_jitter_preserves_static_labels_and_censoring(self):
        base = pd.DataFrame(dict(age=[50, 40, 60, 70], time=[.001, 9.999, 10., 10.],
                                 onset=[1, 0, 0, 0], death=[0, 1, 0, 0]))
        out = transform(base, 1.)
        pd.testing.assert_frame_equal(base.drop(columns="time"), out.drop(columns="time"))
        np.testing.assert_array_equal(out.time.iloc[2:], base.time.iloc[2:])
        self.assertTrue(out.time.iloc[:2].between(.001, 9.999).all())
        pd.testing.assert_frame_equal(out, transform(base, 1.))
        pd.testing.assert_frame_equal(base, transform(base, 0.))

    def test_calibration_protects_rare_excluded_and_score_multiset(self):
        baseline = np.array([[.1], [.2], [.3], [.4], [.5]])
        rare = np.array([1, 0, 0, 0, 0], dtype=bool)
        excluded = np.array([0, 0, 0, 0, 1], dtype=bool)
        out, reports = reassign(baseline, -baseline, rare, excluded)
        np.testing.assert_array_equal(out[[0, 4]], baseline[[0, 4]])
        np.testing.assert_array_equal(np.sort(out, axis=0), np.sort(baseline, axis=0))
        self.assertEqual(out[1:4, 0].tolist(), [.4, .3, .2])
        self.assertIsNone(reports[0]["fallback"])

    def test_tie_split_causes_target_fallback(self):
        baseline = np.array([[.1], [.1], [.2]])
        out, reports = reassign(baseline, np.array([[0.], [2.], [1.]]),
                                np.array([1, 0, 0], bool), np.zeros(3, bool))
        np.testing.assert_array_equal(out, baseline)
        self.assertEqual(reports[0]["fallback"], "baseline_tie_would_split")

    def test_calibration_alignment_and_shuffle_invariance(self):
        baseline = pd.DataFrame({"record_id": [3, 5, 7, 9, 11], "0": [.1, .2, .3, .4, .5]})
        evidence = pd.DataFrame({"record_id": [3, 5, 7, 9, 11], "0": [5., 4., 3., 2., 1.]})
        masks = pd.DataFrame({"record_id": [3, 5, 7, 9, 11], "rare": [1, 0, 0, 0, 0],
                              "excluded": [0, 0, 0, 0, 1], "home": [0, 0, 1, 0, 0]})
        out, _ = calibrate_mia(baseline, evidence, masks)
        shuffled, _ = calibrate_mia(baseline.sample(frac=1, random_state=4),
                                    evidence.sample(frac=1, random_state=5), masks.iloc[::-1])
        pd.testing.assert_frame_equal(out.set_index("record_id").sort_index(),
                                      shuffled.set_index("record_id").sort_index())
        with self.assertRaises(ValueError):
            calibrate_mia(baseline, evidence.iloc[:-1], masks)

    def test_whole_target_selection_preserves_strings(self):
        a = pd.DataFrame({"target_k": ["1", "1", "2", "2"],
                          "challenge_row_id": ["10", "11", "20", "21"],
                          "time_hat": ["0.1000", "9.000", "7.25", "8.0"]})
        b = a.copy()
        b["time_hat"] = ["1.1234567891234567", "2.0", "3.000", "4.00"]
        out = integrate_aia({"A": a, "B": b}, {"1": "A", "2": "B"}, 2)
        self.assertEqual(out.time_hat.tolist(), ["0.1000", "9.000", "3.000", "4.00"])
        with self.assertRaises(ValueError):
            integrate_aia({"A": a}, {"1": "A", "2": "B"}, 2)
        with self.assertRaises(ValueError):
            integrate_aia({"A": pd.concat([a, a.iloc[:1]])}, {"1": "A"}, 2)

    def test_kernels_match_archived_function_ast(self):
        kernel = ast.parse((ROOT/"pwscup_public/kernels.py").read_text())
        functions = {n.name: n for n in kernel.body if isinstance(n, ast.FunctionDef)}
        for path, names in [("research/utility_slack.py", ["transform"]),
                            ("research/rare_fixed_calibration.py", ["priority", "reassign"])]:
            source = ast.parse((ROOT/path).read_text())
            original = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
            for name in names:
                self.assertEqual(ast.dump(functions[name]), ast.dump(original[name]))

    def test_output_refuses_to_overwrite_an_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.csv"
            write_new(pd.DataFrame({"value": [1]}), path)
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(pd.DataFrame({"value": [2]}), path)
            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
