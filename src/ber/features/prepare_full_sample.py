"""Build a reproducible, label-preserving sample from the full candidate union.

The sample is selected by anchor and country before reading candidates. Every
candidate for a selected anchor is retained; no label is used for selection.
"""
import argparse
import json
import time
from pathlib import Path

import duckdb

from ber.data.build_normalized import sha256, write_json


RECORD_COLUMNS = ("entity_id", "country", "name_normalized", "address_normalized",
                  "name_accent_folded", "address_accent_folded", "name_scripts", "address_scripts")


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def build(release, union, output, us_anchors=12_000, india_anchors=8_000,
          seed="20260925-full-sample-v1"):
    if output.exists():
        raise FileExistsError(output)
    release_manifest = json.loads((release / "manifest.json").read_text())
    union_manifest = json.loads((union / "manifest.json").read_text())
    release_hash = sha256(release / "manifest.json")
    if (release_manifest.get("status") != "PASS" or
            (release / "_SUCCESS").read_text().strip() != release_hash or
            union_manifest.get("status") != "PASS" or union_manifest.get("split") != "train" or
            union_manifest.get("release_manifest_sha256") != release_hash or
            (union / "_SUCCESS").read_text().strip() != sha256(union / "manifest.json")):
        raise ValueError("Input release or training candidate union is incomplete")
    truth = Path(release_manifest["ground_truth"]["path"])
    if sha256(truth) != release_manifest["ground_truth"]["sha256"]:
        raise ValueError("Training truth hash mismatch")
    output.mkdir(parents=True)
    started = time.time()
    connection = duckdb.connect()
    connection.execute("SET memory_limit='5GB'")
    connection.execute("SET threads=4")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET temp_directory={sql_path(output / 'tmp')}")
    s1_glob = str(release / "train" / "source1" / "*.parquet")
    targets_glob = [str(release / "train" / source / "*.parquet") for source in ("source2", "source3")]
    union_glob = str(union / "part-*.parquet")
    columns = ", ".join(RECORD_COLUMNS)
    try:
        connection.execute("""
            CREATE TABLE sampled_anchors AS
            SELECT entity_id, country FROM (
                SELECT entity_id, country,
                       row_number() OVER (
                           PARTITION BY country ORDER BY md5(entity_id || ?), entity_id
                       ) AS sample_rank
                FROM read_parquet(?) WHERE country IN ('US', 'India')
            ) WHERE (country='US' AND sample_rank<=?)
                 OR (country='India' AND sample_rank<=?)
        """, [seed, s1_glob, us_anchors, india_anchors])
        counts = dict(connection.execute(
            "SELECT country, count(*) FROM sampled_anchors GROUP BY country").fetchall())
        if counts != {"US": us_anchors, "India": india_anchors}:
            raise ValueError(f"Requested sample cannot be filled: {counts}")
        connection.execute(f"""
            COPY (
                SELECT {', '.join('q.' + col for col in RECORD_COLUMNS)}
                FROM read_parquet(?) q JOIN sampled_anchors s ON q.entity_id=s.entity_id
                ORDER BY q.entity_id
            ) TO {sql_path(output / 'queries.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """, [s1_glob])
        connection.execute("""
            CREATE TABLE sampled_candidates AS
            SELECT c.* FROM read_parquet(?) c
            SEMI JOIN sampled_anchors s ON c.source1_entity_id=s.entity_id
        """, [union_glob])
        connection.execute(f"""
            COPY (SELECT * FROM sampled_candidates
                  ORDER BY source1_entity_id, candidate_rank)
            TO {sql_path(output / 'candidates.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.execute("""
            CREATE TABLE target_ids AS SELECT DISTINCT candidate_id AS entity_id
            FROM sampled_candidates
        """)
        connection.execute(f"""
            COPY (
                SELECT {', '.join('t.' + col for col in RECORD_COLUMNS)}
                FROM read_parquet(?) t JOIN target_ids i USING (entity_id)
                ORDER BY t.entity_id
            ) TO {sql_path(output / 'targets.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """, [targets_glob])
        connection.execute(f"""
            COPY (
                SELECT g.source1_entity_id,
                       coalesce(g.matched_entity_ids, '') AS matched_entity_ids
                FROM read_csv(?, delim='\t', header=true,
                    columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}}) g
                JOIN sampled_anchors s ON g.source1_entity_id=s.entity_id
                ORDER BY g.source1_entity_id
            ) TO {sql_path(output / 'truth.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """, [str(truth)])
        n_candidates = connection.execute("SELECT count(*) FROM sampled_candidates").fetchone()[0]
        n_target_ids = connection.execute("SELECT count(*) FROM target_ids").fetchone()[0]
        n_targets = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(output / "targets.parquet")]).fetchone()[0]
        n_truth = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(output / "truth.parquet")]).fetchone()[0]
        if n_targets != n_target_ids or n_truth != us_anchors + india_anchors:
            raise ValueError("Sample target or truth coverage is incomplete")
        files = ["queries.parquet", "candidates.parquet", "targets.parquet", "truth.parquet"]
        manifest = {
            "status": "PASS", "version": "full-train-anchor-sample-v1",
            "selection": "country-stratified deterministic anchor hash; labels unused",
            "seed": seed, "country_anchors": counts,
            "anchors": us_anchors + india_anchors, "candidate_pairs": n_candidates,
            "distinct_candidate_targets": n_target_ids,
            "release_manifest_sha256": release_hash,
            "union_manifest_sha256": sha256(union / "manifest.json"),
            "truth_sha256": sha256(truth),
            "files": {name: sha256(output / name) for name in files},
            "elapsed_seconds": round(time.time() - started, 2),
        }
        write_json(output / "manifest.json", manifest)
        (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
        return manifest
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--us-anchors", type=int, default=12_000)
    parser.add_argument("--india-anchors", type=int, default=8_000)
    parser.add_argument("--seed", default="20260925-full-sample-v1")
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.union, args.output, args.us_anchors,
                           args.india_anchors, args.seed), indent=2))
