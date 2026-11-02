"""Safety and partial-download tests for the new archive adapter."""
import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restore_assets import inspect


class RestoreTests(unittest.TestCase):
    def fixture(self, root, missing=False):
        p = root / "models/canonical.bin"
        p.parent.mkdir()
        p.write_bytes(b"original")
        row = dict(file="controlled/copy.bin", canonical_file="models/canonical.bin",
                   package="models", bytes=8, sha256=hashlib.sha256(b"original").hexdigest())
        rows = [dict(row, file="models/canonical.bin"), row]
        if missing:
            rows.append(dict(row, file="other/missing.bin", canonical_file="other/missing.bin",
                             package="extended-evidence"))
        with (root / "FILE_INDEX.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        return row

    def test_inspection_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            out = inspect(root, verify=True)
            self.assertEqual(out["aliases_available"], 1)
            self.assertFalse((root / "controlled").exists())

    def test_restoration_is_byte_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            self.assertEqual(inspect(root, restore=True)["aliases_restored"], 1)
            self.assertEqual((root / "controlled/copy.bin").read_bytes(), b"original")
            self.assertEqual(inspect(root, restore=True, verify=True)["aliases_restored"], 0)

    def test_partial_download_reports_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root, missing=True)
            result = inspect(root, restore=True)
            self.assertEqual(result["missing_packages"], {"extended-evidence": 1})
            self.assertEqual(result["aliases_restored"], 1)

    def test_strict_missing_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root, missing=True)
            with self.assertRaises(ValueError):
                inspect(root, restore=True, require_all=True)
            self.assertFalse((root / "controlled").exists())

    def test_same_size_conflict_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            p = root / "controlled/copy.bin"; p.parent.mkdir(); p.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                inspect(root, restore=True)
            self.assertEqual(p.read_bytes(), b"modified")

    def test_corrupt_canonical_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            (root / "models/canonical.bin").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                inspect(root, restore=True)
            self.assertFalse((root / "controlled").exists())

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            p = root / "FILE_INDEX.csv"
            p.write_text(p.read_text().replace("controlled/copy.bin", "../outside.bin"))
            with self.assertRaises(ValueError):
                inspect(root, restore=True)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            (root / "controlled").symlink_to(root / "models", target_is_directory=True)
            with self.assertRaises(ValueError):
                inspect(root, restore=True)


if __name__ == "__main__":
    unittest.main()
