"""Measure the true-link coverage of a completed, capped three-route union."""
import argparse
import json
import time
from pathlib import Path

import duckdb

from ber.data.build_normalized import sha256, write_json
from ber.evaluation.route_coverage import bucket, finalize, update


def measure(release, baseline, uncapped, union, output):
    if output.exists():
        raise FileExistsError(output)
    release_hash = sha256(release / "manifest.json")
    baseline_report = json.loads((baseline / "report.json").read_text())
    uncapped_report = json.loads((uncapped / "report.json").read_text())
    union_manifest = json.loads((union / "manifest.json").read_text())
    if (baseline_report.get("status") != "PASS" or
            baseline_report.get("release_manifest_sha256") != release_hash or
            uncapped_report.get("status") != "PASS" or
            uncapped_report.get("release_manifest_sha256") != release_hash or
            uncapped_report.get("baseline_report_sha256") != sha256(baseline / "report.json")):
        raise ValueError("Coverage reports do not belong to this release")
    expected_routes = {baseline_report["exact_manifest_sha256"],
                       baseline_report["dense_manifest_sha256"],
                       uncapped_report["fuzzy_manifest_sha256"]}
    if (union_manifest.get("status") != "PASS" or union_manifest.get("split") != "train" or
            union_manifest.get("release_manifest_sha256") != release_hash or
            set(union_manifest.get("route_manifests", [])) != expected_routes or
            (union / "_SUCCESS").read_text().strip() != sha256(union / "manifest.json")):
        raise ValueError("Capped union is incomplete or uses different routes")
    parts = sorted(union.glob("part-*.parquet"))
    expected_parts = len(list((release / "train" / "source1").glob("part-*.parquet")))
    if len(parts) != expected_parts:
        raise ValueError("Capped union has incomplete shard coverage")

    output.mkdir(parents=True)
    started = time.time()
    connection = duckdb.connect(str(baseline / "coverage.duckdb"), read_only=True)
    connection.execute("SET memory_limit='5GB'")
    connection.execute("SET threads=4")
    try:
        connection.execute("""
            CREATE TEMP TABLE capped_hit_counts AS
            SELECT c.source1_entity_id AS anchor, count(*) AS hits
            FROM read_parquet(?) c
            JOIN links l ON c.source1_entity_id=l.anchor AND c.candidate_id=l.target
            GROUP BY c.source1_entity_id
        """, [list(map(str, parts))])
        results = {"all": bucket()}
        cursor = connection.execute("""
            SELECT a.country, a.truth_count, coalesce(h.hits, 0)
            FROM anchors a LEFT JOIN capped_hit_counts h ON a.anchor=h.anchor
        """)
        while rows := cursor.fetchmany(50_000):
            for country, truth_count, hits in rows:
                update(results["all"], truth_count, hits)
                update(results.setdefault(country, bucket()), truth_count, hits)
        for values in results.values():
            finalize(values)
        differences = {}
        for country, values in results.items():
            prior = uncapped_report["coverage"]["combined"][country]
            if values["anchors"] != prior["anchors"]:
                raise ValueError(f"Anchor coverage differs for {country}")
            if values["retrieved_links"] > prior["retrieved_links"]:
                raise ValueError(f"Capped union exceeds uncapped recall for {country}")
            differences[country] = {
                "true_links_lost_to_cap": prior["retrieved_links"] - values["retrieved_links"],
                "oracle_macro_f05_loss": prior["oracle_macro_f05"] - values["oracle_macro_f05"],
            }
        report = {
            "status": "PASS",
            "scope": "training candidate coverage, not classifier accuracy",
            "candidate_cap_per_anchor": union_manifest["candidate_cap_per_anchor"],
            "candidate_pairs": union_manifest["candidate_pairs"],
            "release_manifest_sha256": release_hash,
            "uncapped_report_sha256": sha256(uncapped / "report.json"),
            "union_manifest_sha256": sha256(union / "manifest.json"),
            "coverage": results,
            "cap_impact": differences,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        write_json(output / "report.json", report)
        (output / "_SUCCESS").write_text(sha256(output / "report.json") + "\n")
        return report
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--uncapped", type=Path, required=True)
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.release, args.baseline, args.uncapped, args.union, args.output), indent=2))
