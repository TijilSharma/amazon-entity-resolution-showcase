"""Measure dense, exact, and combined training candidate recall with DuckDB.

Only true-positive candidate pairs are materialized. The full candidate routes
are scanned from Parquet and are never copied into the evaluation database.
"""
import argparse
import json
import time
from pathlib import Path

import duckdb

from ber.data.build_normalized import sha256, write_json


def bucket():
    return {"anchors": 0, "positive_anchors": 0, "singleton_anchors": 0,
            "true_links": 0, "retrieved_links": 0, "complete_positive_anchors": 0,
            "macro_positive_recall_sum": 0.0, "oracle_macro_f05_sum": 0.0}


def update(values, truth, hits):
    values["anchors"] += 1
    if truth:
        values["positive_anchors"] += 1
        values["true_links"] += truth
        values["retrieved_links"] += hits
        values["complete_positive_anchors"] += hits == truth
        values["macro_positive_recall_sum"] += hits / truth
        values["oracle_macro_f05_sum"] += 5 * hits / (4 * hits + truth)
    else:
        values["singleton_anchors"] += 1
        values["oracle_macro_f05_sum"] += 1.0


def finalize(values):
    positives = values["positive_anchors"]
    values["micro_link_recall"] = values["retrieved_links"] / values["true_links"] if values["true_links"] else None
    values["macro_positive_anchor_recall"] = values.pop("macro_positive_recall_sum") / positives if positives else None
    values["complete_positive_anchor_recall"] = values["complete_positive_anchors"] / positives if positives else None
    values["oracle_macro_f05"] = values.pop("oracle_macro_f05_sum") / values["anchors"] if values["anchors"] else None


def measure(release, truth, dense, exact, output):
    if output.exists():
        raise FileExistsError(output)
    release_manifest = json.loads((release / "manifest.json").read_text())
    if release_manifest.get("status") != "PASS" or sha256(truth) != release_manifest["ground_truth"]["sha256"]:
        raise ValueError("Normalized release or truth hash mismatch")
    release_hash = sha256(release / "manifest.json")
    for route in (dense, exact):
        manifest = json.loads((route / "manifest.json").read_text())
        if (manifest.get("status") != "PASS" or manifest.get("split") != "train"
                or manifest.get("release_manifest_sha256") != release_hash
                or (route / "_SUCCESS").read_text().strip() != sha256(route / "manifest.json")):
            raise ValueError(f"Invalid candidate route: {route}")
    output.mkdir(parents=True)
    started = time.time()
    connection = duckdb.connect(str(output / "coverage.duckdb"))
    connection.execute("SET memory_limit='5GB'")
    connection.execute("SET threads=4")
    connection.execute("SET preserve_insertion_order=false")
    source1_glob = str(release / "train" / "source1" / "*.parquet")
    dense_glob = str(dense / "part-*.parquet")
    exact_glob = str(exact / "part-*.parquet")
    connection.execute("""
        CREATE TABLE anchors AS
        SELECT g.source1_entity_id AS anchor, s.country,
               coalesce(g.matched_entity_ids, '') AS matched_entity_ids,
               CASE WHEN coalesce(g.matched_entity_ids, '') = '' THEN 0
                    ELSE array_length(string_split(g.matched_entity_ids, ',')) END AS truth_count
        FROM read_csv(?, delim='\t', header=true,
            columns={'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}) g
        JOIN read_parquet(?) s ON g.source1_entity_id=s.entity_id
    """, [str(truth), source1_glob])
    connection.execute("""
        CREATE TABLE links AS
        SELECT anchor, unnest(string_split(matched_entity_ids, ',')) AS target
        FROM anchors WHERE truth_count > 0
    """)
    print("truth loaded", flush=True)
    connection.execute("""
        CREATE TABLE dense_hits AS
        SELECT c.source1_entity_id AS anchor, c.candidate_id AS target
        FROM read_parquet(?) c
        JOIN links l ON c.source1_entity_id=l.anchor AND c.candidate_id=l.target
    """, [dense_glob])
    print("dense candidates scanned", flush=True)
    connection.execute("""
        CREATE TABLE exact_hits AS
        SELECT c.source1_entity_id AS anchor, c.candidate_id AS target
        FROM read_parquet(?) c
        JOIN links l ON c.source1_entity_id=l.anchor AND c.candidate_id=l.target
    """, [exact_glob])
    print("exact candidates scanned", flush=True)
    connection.execute("""
        CREATE TABLE hit_counts AS
        SELECT anchor, sum(CAST(dense AS INTEGER)) AS dense_hits,
               sum(CAST(exact AS INTEGER)) AS exact_hits,
               count(*) AS union_hits
        FROM (
            SELECT anchor, target, bool_or(dense) AS dense, bool_or(exact) AS exact
            FROM (
                SELECT anchor, target, true AS dense, false AS exact FROM dense_hits
                UNION ALL
                SELECT anchor, target, false AS dense, true AS exact FROM exact_hits
            ) GROUP BY anchor, target
        ) GROUP BY anchor
    """)
    results = {name: {"all": bucket()} for name in ("dense", "exact", "union")}
    cursor = connection.execute("""
        SELECT a.country, a.truth_count, coalesce(h.dense_hits, 0),
               coalesce(h.exact_hits, 0), coalesce(h.union_hits, 0)
        FROM anchors a LEFT JOIN hit_counts h USING (anchor)
    """)
    while rows := cursor.fetchmany(50_000):
        for country, truth_count, dense_hits, exact_hits, union_hits in rows:
            for name, hits in (("dense", dense_hits), ("exact", exact_hits), ("union", union_hits)):
                update(results[name]["all"], truth_count, hits)
                update(results[name].setdefault(country, bucket()), truth_count, hits)
    for route in results.values():
        for values in route.values():
            finalize(values)
    report = {"status": "PASS", "scope": "training candidate coverage, not classifier accuracy",
              "dense_candidate_pairs": json.loads((dense / "manifest.json").read_text())["candidate_pairs"],
              "exact_candidate_pairs": json.loads((exact / "manifest.json").read_text())["candidate_pairs"],
              "release_manifest_sha256": release_hash, "truth_sha256": sha256(truth),
              "dense_manifest_sha256": sha256(dense / "manifest.json"),
              "exact_manifest_sha256": sha256(exact / "manifest.json"),
              "coverage": results, "elapsed_seconds": round(time.time() - started, 2)}
    write_json(output / "report.json", report)
    (output / "_SUCCESS").write_text(sha256(output / "report.json") + "\n")
    connection.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.release, args.truth, args.dense, args.exact, args.output), indent=2))
