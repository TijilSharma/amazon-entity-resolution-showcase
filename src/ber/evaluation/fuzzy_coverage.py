"""Measure fuzzy's added true-link coverage on completed training shards.

The baseline DuckDB file is created by ``ber.evaluation.route_coverage``.
Passing a BUILDING fuzzy route measures a partial, explicitly scoped result;
a PASS route measures the complete training corpus.
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json
from ber.evaluation.route_coverage import bucket, finalize, update


def measure(release, baseline, fuzzy, output):
    if output.exists():
        raise FileExistsError(output)
    release_hash = sha256(release / "manifest.json")
    baseline_report = json.loads((baseline / "report.json").read_text())
    fuzzy_manifest = json.loads((fuzzy / "manifest.json").read_text())
    if (baseline_report.get("status") != "PASS" or
            baseline_report.get("release_manifest_sha256") != release_hash):
        raise ValueError("Baseline coverage does not belong to this release")
    if (fuzzy_manifest.get("status") not in ("BUILDING", "PASS") or
            fuzzy_manifest.get("split") != "train" or
            fuzzy_manifest.get("release_manifest_sha256") != release_hash):
        raise ValueError("Fuzzy route does not belong to this training release")
    if fuzzy_manifest["status"] == "PASS" and (fuzzy / "_SUCCESS").read_text().strip() != sha256(fuzzy / "manifest.json"):
        raise ValueError("Fuzzy PASS manifest lacks a valid success marker")
    parts = sorted(fuzzy.glob("part-*.parquet"))
    if not parts:
        raise ValueError("No complete fuzzy candidate shards exist yet")
    source_paths = [release / "train" / "source1" / path.name for path in parts]
    if not all(path.exists() for path in source_paths):
        raise ValueError("Fuzzy candidate shards do not align with Source-1 shards")
    output.mkdir(parents=True)
    started = time.time()
    connection = duckdb.connect(str(baseline / "coverage.duckdb"), read_only=True)
    connection.execute("SET memory_limit='5GB'")
    connection.execute("SET threads=4")
    try:
        connection.execute("CREATE TEMP TABLE scope AS SELECT entity_id AS anchor, country FROM read_parquet(?)",
                           [list(map(str, source_paths))])
        connection.execute("""
            CREATE TEMP TABLE prior_hits AS
            SELECT d.anchor, d.target FROM dense_hits d SEMI JOIN scope s USING (anchor)
            UNION
            SELECT e.anchor, e.target FROM exact_hits e SEMI JOIN scope s USING (anchor)
        """)
        connection.execute("""
            CREATE TEMP TABLE fuzzy_hits AS
            SELECT f.source1_entity_id AS anchor, f.candidate_id AS target
            FROM read_parquet(?) f
            JOIN links l ON f.source1_entity_id=l.anchor AND f.candidate_id=l.target
        """, [list(map(str, parts))])
        connection.execute("CREATE TEMP TABLE combined_hits AS SELECT * FROM prior_hits UNION SELECT * FROM fuzzy_hits")
        results = {name: {"all": bucket()} for name in ("prior", "fuzzy", "combined")}
        cursor = connection.execute("""
            SELECT s.country, a.truth_count, coalesce(p.n, 0),
                   coalesce(f.n, 0), coalesce(c.n, 0)
            FROM scope s JOIN anchors a USING (anchor)
            LEFT JOIN (SELECT anchor, count(*) AS n FROM prior_hits GROUP BY anchor) p USING (anchor)
            LEFT JOIN (SELECT anchor, count(*) AS n FROM fuzzy_hits GROUP BY anchor) f USING (anchor)
            LEFT JOIN (SELECT anchor, count(*) AS n FROM combined_hits GROUP BY anchor) c USING (anchor)
        """)
        while rows := cursor.fetchmany(50_000):
            for country, truth_count, prior_hits, fuzzy_hits, combined_hits in rows:
                for name, hits in (("prior", prior_hits), ("fuzzy", fuzzy_hits), ("combined", combined_hits)):
                    update(results[name]["all"], truth_count, hits)
                    update(results[name].setdefault(country, bucket()), truth_count, hits)
        for route in results.values():
            for values in route.values():
                finalize(values)
        full_shards = len(list((release / "train" / "source1").glob("part-*.parquet")))
        report = {"status": "PASS" if fuzzy_manifest["status"] == "PASS" and len(parts) == full_shards else "PARTIAL",
                  "scope": "training candidate coverage, not classifier accuracy",
                  "release_manifest_sha256": release_hash,
                  "baseline_report_sha256": sha256(baseline / "report.json"),
                  "fuzzy_manifest_sha256": sha256(fuzzy / "manifest.json"),
                  "completed_shards": len(parts), "total_shards": full_shards,
                  "fuzzy_candidate_pairs_in_scope": sum(pq.ParquetFile(path).metadata.num_rows for path in parts),
                  "coverage": results, "elapsed_seconds": round(time.time() - started, 2)}
        write_json(output / "report.json", report)
        (output / "_SUCCESS").write_text(sha256(output / "report.json") + "\n")
        return report
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fuzzy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.release, args.baseline, args.fuzzy, args.output), indent=2))
