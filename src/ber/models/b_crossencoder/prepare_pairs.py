"""Prepare a bounded, split-aligned text-pair set for Pipeline B.

This is model input preparation, not cross-encoder training. Holdout anchors are
excluded, and every retrieved positive is retained before negative sampling.
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json
from ber.models.a_catboost.train_full_sample import checked_manifest


ROLES = ("train", "early_stop", "calibration")


def quoted(path):
    return "'" + str(path).replace("'", "''") + "'"


def build(bundle, splits_path, output, negatives_per_anchor=20, full_eval=False):
    if output.exists():
        raise FileExistsError(output)
    if negatives_per_anchor < 1:
        raise ValueError("negatives_per_anchor must be positive")
    source = checked_manifest(bundle)
    splits = json.loads(splits_path.read_text())
    if len(splits) != source["anchors"] or set(splits.values()) != {*ROLES, "holdout"}:
        raise ValueError("Splits do not match the candidate sample")
    started = time.time()
    output.mkdir(parents=True)
    split_table = pa.table({"source1_entity_id": list(splits), "role": list(splits.values())})
    pq.write_table(split_table, output / "splits.parquet", compression="zstd")
    connection = duckdb.connect()
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='4GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET temp_directory={quoted(output / 'tmp')}")
    try:
        connection.execute(f"""
            CREATE TEMP VIEW labeled AS
            SELECT c.source1_entity_id, c.candidate_id, s.role,
                   c.candidate_rank,
                   q.country AS query_country, t.country AS target_country,
                   q.name_normalized AS query_name,
                   q.address_normalized AS query_address,
                   t.name_normalized AS target_name,
                   t.address_normalized AS target_address,
                   CAST(list_contains(string_split(g.matched_entity_ids, ','),
                                      c.candidate_id) AS TINYINT) AS label
            FROM read_parquet({quoted(bundle / 'candidates.parquet')}) c
            JOIN read_parquet({quoted(output / 'splits.parquet')}) s USING (source1_entity_id)
            JOIN read_parquet({quoted(bundle / 'truth.parquet')}) g USING (source1_entity_id)
            JOIN read_parquet({quoted(bundle / 'queries.parquet')}) q ON q.entity_id=c.source1_entity_id
            JOIN read_parquet({quoted(bundle / 'targets.parquet')}) t ON t.entity_id=c.candidate_id
            WHERE s.role IN ('train', 'early_stop', 'calibration')
        """)
        connection.execute(f"""
            CREATE TEMP TABLE selected AS
            SELECT * EXCLUDE (negative_rank) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY source1_entity_id, label
                    ORDER BY candidate_rank, candidate_id
                ) AS negative_rank FROM labeled
            ) WHERE label=1 OR negative_rank<={int(negatives_per_anchor)}
                OR ({str(bool(full_eval)).lower()} AND role IN ('early_stop', 'calibration'))
        """)
        counts = {}
        for role in ROLES:
            path = output / f"{role}.parquet"
            connection.execute(f"""
                COPY (SELECT source1_entity_id, candidate_id, candidate_rank,
                             query_country, target_country, query_name,
                             query_address, target_name, target_address, label
                      FROM selected WHERE role='{role}'
                      ORDER BY source1_entity_id, candidate_rank, candidate_id)
                TO {quoted(path)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            rows, positives, anchors = connection.execute(
                "SELECT count(*), sum(label), count(DISTINCT source1_entity_id) "
                "FROM selected WHERE role=?", [role]).fetchone()
            counts[role] = {"pairs": rows, "positives": positives,
                            "anchors_with_candidates": anchors}
        expected_positives = connection.execute(
            "SELECT sum(label) FROM labeled").fetchone()[0]
        if sum(item["positives"] for item in counts.values()) != expected_positives:
            raise ValueError("A retrieved positive was lost")
        files = ["splits.parquet", *(f"{role}.parquet" for role in ROLES)]
        result = {"status": "PASS", "version": "b-crossencoder-pairs-v1",
                  "scope": "sampled training candidates; no holdout or test labels",
                  "negatives_per_anchor": negatives_per_anchor,
                  "full_eval_candidates": bool(full_eval),
                  "counts": counts,
                  "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
                  "splits_sha256": sha256(splits_path),
                  "files": {name: sha256(output / name) for name in files},
                  "elapsed_seconds": round(time.time() - started, 2)}
        write_json(output / "manifest.json", result)
        (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
        return result
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negatives-per-anchor", type=int, default=20)
    parser.add_argument("--full-eval", action="store_true",
                        help="Keep all early-stop and calibration candidates for fair entity-level evaluation")
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.splits, args.output,
                           args.negatives_per_anchor, args.full_eval), indent=2))
