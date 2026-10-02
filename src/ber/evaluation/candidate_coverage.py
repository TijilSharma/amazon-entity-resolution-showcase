"""Measure candidate-generation recall against training truth without training a model."""
import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def batches(values, size=50_000):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def measure(release, candidates, truth_path, output):
    if output.exists():
        raise FileExistsError(output)
    release_manifest = json.loads((release / "manifest.json").read_text())
    candidate_manifest = json.loads((candidates / "manifest.json").read_text())
    if release_manifest.get("status") != "PASS" or candidate_manifest.get("status") != "PASS":
        raise ValueError("Both release and candidate manifests must be PASS")
    if candidate_manifest.get("split") != "train" or candidate_manifest.get("release_manifest_sha256") != sha256(release / "manifest.json"):
        raise ValueError("Candidates do not belong to this training release")
    if sha256(truth_path) != release_manifest["ground_truth"]["sha256"]:
        raise ValueError("Ground truth hash differs from release")
    output.mkdir(parents=True)
    database = output / "coverage.sqlite"
    started = time.time()
    connection = sqlite3.connect(database)
    try:
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE anchors (anchor TEXT PRIMARY KEY, country TEXT NOT NULL, truth_count INTEGER NOT NULL);
            CREATE TABLE links (anchor TEXT NOT NULL, target TEXT NOT NULL);
            CREATE TABLE candidates (anchor TEXT NOT NULL, target TEXT NOT NULL);
        """)
        truth_rows, link_rows = [], []
        with truth_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                targets = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
                truth_rows.append((row["source1_entity_id"], len(targets)))
                link_rows.extend((row["source1_entity_id"], target) for target in targets)
                if len(truth_rows) >= 50_000:
                    connection.executemany("INSERT INTO anchors(anchor, country, truth_count) VALUES (?, '', ?)", truth_rows)
                    connection.executemany("INSERT INTO links VALUES (?, ?)", link_rows)
                    truth_rows.clear(); link_rows.clear()
            if truth_rows:
                connection.executemany("INSERT INTO anchors(anchor, country, truth_count) VALUES (?, '', ?)", truth_rows)
                connection.executemany("INSERT INTO links VALUES (?, ?)", link_rows)
        countries = []
        for path in sorted((release / "train" / "source1").glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=["entity_id", "country"], batch_size=50_000):
                countries.extend((r["country"], r["entity_id"]) for r in batch.to_pylist())
                if len(countries) >= 50_000:
                    connection.executemany("UPDATE anchors SET country=? WHERE anchor=?", countries)
                    countries.clear()
        if countries:
            connection.executemany("UPDATE anchors SET country=? WHERE anchor=?", countries)
        candidate_count = 0
        for path in sorted(candidates.glob("part-*.parquet")):
            table = pq.read_table(path, columns=["source1_entity_id", "candidate_id"])
            values = list(zip(table["source1_entity_id"].to_pylist(), table["candidate_id"].to_pylist()))
            connection.executemany("INSERT INTO candidates VALUES (?, ?)", values)
            candidate_count += len(values)
        connection.commit()
        connection.executescript("CREATE INDEX link_lookup ON links(anchor, target); CREATE INDEX candidate_lookup ON candidates(anchor, target);")
        duplicate_candidates = connection.execute("SELECT COUNT(*) FROM (SELECT anchor, target FROM candidates GROUP BY anchor, target HAVING COUNT(*) > 1)").fetchone()[0]
        if duplicate_candidates:
            raise ValueError(f"Candidate output contains {duplicate_candidates} duplicate pairs")
        query = """
            SELECT a.country, a.truth_count,
                   SUM(CASE WHEN c.target IS NOT NULL THEN 1 ELSE 0 END) AS hit_count
            FROM anchors a
            LEFT JOIN links l ON a.anchor=l.anchor
            LEFT JOIN candidates c ON l.anchor=c.anchor AND l.target=c.target
            GROUP BY a.anchor
        """
        result = {"all": {"anchors": 0, "positive_anchors": 0, "singleton_anchors": 0, "true_links": 0,
                          "retrieved_links": 0, "complete_positive_anchors": 0, "macro_positive_recall_sum": 0.0}}
        for country, truth_count, hit_count in connection.execute(query):
            bucket = result.setdefault(country, {"anchors": 0, "positive_anchors": 0, "singleton_anchors": 0,
                                                   "true_links": 0, "retrieved_links": 0,
                                                   "complete_positive_anchors": 0, "macro_positive_recall_sum": 0.0})
            for target in (result["all"], bucket):
                target["anchors"] += 1
                if truth_count:
                    target["positive_anchors"] += 1
                    target["true_links"] += truth_count
                    target["retrieved_links"] += hit_count
                    target["complete_positive_anchors"] += hit_count == truth_count
                    target["macro_positive_recall_sum"] += hit_count / truth_count
                else:
                    target["singleton_anchors"] += 1
        for values in result.values():
            values["micro_link_recall"] = values["retrieved_links"] / values["true_links"] if values["true_links"] else None
            values["macro_positive_anchor_recall"] = values.pop("macro_positive_recall_sum") / values["positive_anchors"] if values["positive_anchors"] else None
            values["complete_positive_anchor_recall"] = values["complete_positive_anchors"] / values["positive_anchors"] if values["positive_anchors"] else None
        report = {"status": "PASS", "scope": "training candidate coverage; not final-match accuracy",
                  "candidate_manifest_sha256": sha256(candidates / "manifest.json"),
                  "release_manifest_sha256": sha256(release / "manifest.json"),
                  "truth_sha256": sha256(truth_path), "candidate_pairs": candidate_count,
                  "coverage": result, "elapsed_seconds": round(time.time()-started, 2)}
        write_json(output / "report.json", report)
        (output / "_SUCCESS").write_text(sha256(output / "report.json") + "\n")
        return report
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.release, args.candidates, args.truth, args.output), indent=2))
