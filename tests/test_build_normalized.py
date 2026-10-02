import csv
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from ber.data.build_normalized import build, sha256
from ber.data.validate_inputs import EXPECTED_COLUMNS


class BuildTest(unittest.TestCase):
    def fixture(self, root):
        data = root / "raw"
        audit = {"structural_status": "PASS", "sources": {}}
        for split in ("train", "test"):
            (data / split).mkdir(parents=True)
            for source in (1, 2, 3):
                name = f"{split}_source{source}.tsv"
                path = data / split / name
                with path.open("w", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(EXPECTED_COLUMNS)
                    writer.writerow([f"S{source}-1", "  Café राम  ", "12, null, Road", "India"])
                    writer.writerow([f"S{source}-2", "NA", "", "US"])
                audit["sources"][name] = {"sha256": sha256(path), "total_records": 2}
        gt = data / "train/train_ground_truth.tsv"
        gt.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n")
        audit["ground_truth"] = {"sha256": sha256(gt)}
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit))
        return data, audit_path

    def test_release_preserves_raw_order_and_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, audit = self.fixture(root)
            output = root / "release"
            result = build(data, output, audit, workers=1, shard_rows=1)
            self.assertEqual(result["total_records"], 12)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual((output / "_SUCCESS").read_text().strip(), sha256(output / "manifest.json"))
            rows = pq.read_table(output / "train/source1").to_pylist()
            self.assertEqual([r["source_row"] for r in rows], [0, 1])
            self.assertEqual(rows[0]["business_name"], "  Café राम  ")
            self.assertEqual(rows[0]["name_normalized"], "café राम")
            self.assertTrue(rows[1]["name_placeholder"])
            with self.assertRaises(FileExistsError):
                build(data, output, audit)

    def test_changed_raw_fails_without_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, audit = self.fixture(root)
            with (data / "train/train_source1.tsv").open("a") as f:
                f.write("S1-3\tx\ty\tUS\n")
            output = root / "release"
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                build(data, output, audit, workers=1)
            self.assertFalse((output / "_SUCCESS").exists())
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "FAILED")
