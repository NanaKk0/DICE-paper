import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import Archive, main_table


class ReaderEntryTests(unittest.TestCase):
    def test_current_main_panels_resolve_without_historical_filename_guessing(self):
        with (ROOT / "MAIN_PANEL_MAP.csv").open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for name in row["input_files"].split(" ; "):
                self.assertTrue((ROOT / name).is_file(), name)
        self.assertEqual([r["panel"] for r in rows if r["figure"]=="Fig. 2"], list("abcd"))
        self.assertEqual([r["panel"] for r in rows if r["figure"]=="Fig. 3"], list("abcdefg"))

    def test_all_paper_objects_have_correct_commands(self):
        with (ROOT / "FIGURE_TABLE_MAP.csv").open() as f:
            rows = {r["object"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(rows), 35)
        self.assertIn(" reexport ", rows["Fig. 1"]["command"])
        self.assertIn(" main_table ", rows["Table 1"]["command"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            main_table(Archive(ROOT / "source-data"), output)
            with (output / "MainTable1.csv").open() as f:
                values = list(csv.DictReader(f))
            self.assertEqual([r["target_DFT_records"] for r in values], ["140", "195", "494"])
            self.assertEqual([r["student_geometries"] for r in values], ["2580", "4272", "6041"])

    def test_first_frame_records_are_model_outputs_and_full_vectors(self):
        reference = json.loads((ROOT / "FIRST_FRAME_REFERENCES.json").read_text())
        cards = json.loads((ROOT / "MODEL_CARDS.json").read_text())
        hashes = {m["sha256"] for m in cards["models"]}
        self.assertEqual(len(reference["outputs"]), 6)
        for row in reference["outputs"]:
            self.assertIn(row["model_sha256"], hashes)
            self.assertEqual(len(row["archived_prediction_forces_eV_A"]), row["atom_count"])
            self.assertEqual(len(row["positions_angstrom"]), row["atom_count"])
            self.assertEqual(row["frame"], 0)
            self.assertTrue(row["source_energy_file"].startswith("data/accuracy/"))


if __name__ == "__main__":
    unittest.main()
