"""Full-corpus fuzzy lexical candidate retrieval with SQLite blocking.

It creates a disk-backed target index of normalized word tokens, four-character
token prefixes and numeric tokens. Per Source-1 shard, it retrieves only targets
sharing rare blocking keys, then reranks the bounded candidates with RapidFuzz.
This is candidate generation; its scores are evidence, never final decisions.
"""
import argparse
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from ber.data.build_normalized import sha256, write_json
from ber.features.comparisons import numbers, tokens

VERSION = "full-fuzzy-selective-v2"
COLUMNS = ["entity_id", "name_normalized", "address_normalized"]


def rows(release, split, source):
    for path in sorted((release / split / source).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=50_000):
            yield from batch.to_pylist()


def keys(row):
    """Return distinct blocking keys and deterministic weights for one record."""
    weighted = {}
    for field, base in (("name_normalized", 4), ("address_normalized", 2)):
        text = row[field]
        for token in tokens(text):
            if len(token) >= 3:
                weighted[f"word:{token}"] = max(weighted.get(f"word:{token}", 0), base)
            if len(token) >= 5:
                weighted[f"prefix:{token[:4]}"] = max(weighted.get(f"prefix:{token[:4]}", 0), base - 1)
        for value in numbers(text):
            weighted[f"number:{value}"] = max(weighted.get(f"number:{value}", 0), base + 1)
    return weighted


def fuzzy_score(query, target):
    name = max(fuzz.ratio(query["name_normalized"], target["name_normalized"]),
               fuzz.token_set_ratio(query["name_normalized"], target["name_normalized"])) / 100.0
    if not query["address_normalized"] or not target["address_normalized"]:
        return name
    address = max(fuzz.ratio(query["address_normalized"], target["address_normalized"]),
                  fuzz.token_set_ratio(query["address_normalized"], target["address_normalized"])) / 100.0
    return 0.65 * name + 0.35 * address


def build_index(release, split, database):
    connection = sqlite3.connect(database)
    try:
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'").fetchone():
            required = {"key_counts", "entries_lookup", "key_counts_lookup"}
            found = {name for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','index')")}
            if not required <= found:
                raise ValueError("Existing fuzzy target index is incomplete")
            return
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE targets (target_id TEXT PRIMARY KEY, name TEXT NOT NULL, address TEXT NOT NULL);
            CREATE TABLE entries (key TEXT NOT NULL, target_id TEXT NOT NULL);
        """)
        target_batch, key_batch = [], []
        for source in ("source2", "source3"):
            for row in rows(release, split, source):
                target_batch.append((row["entity_id"], row["name_normalized"], row["address_normalized"]))
                key_batch.extend((key, row["entity_id"]) for key in keys(row))
                if len(target_batch) >= 25_000:
                    connection.executemany("INSERT INTO targets VALUES (?, ?, ?)", target_batch)
                    connection.executemany("INSERT INTO entries VALUES (?, ?)", key_batch)
                    connection.commit()
                    target_batch.clear(); key_batch.clear()
        if target_batch:
            connection.executemany("INSERT INTO targets VALUES (?, ?, ?)", target_batch)
            connection.executemany("INSERT INTO entries VALUES (?, ?)", key_batch)
            connection.commit()
        connection.executescript("""
            CREATE TABLE key_counts AS SELECT key, COUNT(*) AS count FROM entries GROUP BY key;
            CREATE INDEX entries_lookup ON entries(key, target_id);
            CREATE INDEX key_counts_lookup ON key_counts(key, count);
        """)
        connection.commit()
    finally:
        connection.close()


def candidates_for_shard(connection, records, max_key_frequency, preselect, cap, max_keys_per_anchor=None):
    connection.execute("DROP TABLE IF EXISTS query_keys")
    connection.execute("CREATE TEMP TABLE query_keys (anchor TEXT NOT NULL, key TEXT NOT NULL, weight INTEGER NOT NULL)")
    query_by_id = {row["entity_id"]: row for row in records}
    values = [(row["entity_id"], key, weight) for row in records for key, weight in keys(row).items()]
    if not values:
        return []
    connection.executemany("INSERT INTO query_keys VALUES (?, ?, ?)", values)
    sql = """
        WITH selected AS (
          SELECT q.anchor, q.key, q.weight,
                 ROW_NUMBER() OVER (
                   PARTITION BY q.anchor ORDER BY kc.count, q.weight DESC, q.key
                 ) AS key_rank
          FROM query_keys q
          JOIN key_counts kc ON q.key=kc.key AND kc.count <= ?
        ), scored AS (
          SELECT q.anchor, e.target_id, SUM(q.weight) AS evidence
          FROM selected q
          CROSS JOIN entries e INDEXED BY entries_lookup
          WHERE q.key_rank <= ? AND e.key=q.key
          GROUP BY q.anchor, e.target_id
        ), ranked AS (
          SELECT anchor, target_id, evidence,
                 ROW_NUMBER() OVER (PARTITION BY anchor ORDER BY evidence DESC, target_id) AS block_rank
          FROM scored
        )
        SELECT r.anchor, r.target_id, r.evidence, t.name, t.address
        FROM ranked r JOIN targets t ON r.target_id=t.target_id
        WHERE r.block_rank <= ?
        ORDER BY r.anchor, r.block_rank
    """
    grouped = defaultdict(list)
    key_limit = max_keys_per_anchor if max_keys_per_anchor is not None else 1_000_000
    for anchor, target_id, evidence, name, address in connection.execute(sql, (max_key_frequency, key_limit, preselect)):
        grouped[anchor].append((target_id, evidence, {"name_normalized": name, "address_normalized": address}))
    result = []
    for anchor, values in grouped.items():
        query = query_by_id[anchor]
        scored = [(target_id, fuzzy_score(query, target), evidence) for target_id, evidence, target in values]
        scored.sort(key=lambda value: (-value[1], -value[2], value[0]))
        result.extend({"source1_entity_id": anchor, "candidate_id": target_id, "route": "fuzzy_token_prefix",
                       "route_rank": rank, "score": float(score), "blocking_evidence": evidence}
                      for rank, (target_id, score, evidence) in enumerate(scored[:cap], 1))
    return result


SCHEMA = pa.schema([("source1_entity_id", pa.string()), ("candidate_id", pa.string()),
                    ("route", pa.string()), ("route_rank", pa.int32()), ("score", pa.float32()),
                    ("blocking_evidence", pa.int32())])


def process_shard(database, path, output_path, shard, max_key_frequency,
                  preselect, cap, max_keys_per_anchor, batch_size):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        emitted = []
        anchors = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=batch_size):
            records = batch.to_pylist()
            emitted.extend(candidates_for_shard(connection, records, max_key_frequency,
                                                preselect, cap, max_keys_per_anchor))
            anchors += len(records)
        temporary = output_path / f"part-{shard:05d}.parquet.tmp"
        final = output_path / f"part-{shard:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(emitted, schema=SCHEMA), temporary, compression="zstd")
        temporary.replace(final)
        return shard, anchors, len(emitted)
    finally:
        connection.close()


def run(release, split, output, cap=50, preselect=250, max_key_frequency=5_000,
        max_keys_per_anchor=3, batch_size=100, workers=4, resume=False):
    if output.exists() and not resume:
        raise FileExistsError(output)
    raw = json.loads((release / "manifest.json").read_text())
    if raw.get("status") != "PASS" or (release / "_SUCCESS").read_text().strip() != sha256(release / "manifest.json"):
        raise ValueError("Input release is not complete")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "BUILDING", "version": VERSION, "split": split, "candidate_cap_per_anchor": cap,
                "preselect_per_anchor": preselect, "maximum_blocking_key_frequency": max_key_frequency,
                "maximum_keys_per_anchor": max_keys_per_anchor, "query_batch_size": batch_size,
                "workers": workers,
                "release_manifest_sha256": sha256(release / "manifest.json")}
    if resume:
        prior = json.loads((output / "manifest.json").read_text())
        if prior.get("status") not in ("BUILDING", "FAILED") or prior.get("split") != split or prior.get("release_manifest_sha256") != manifest["release_manifest_sha256"]:
            raise ValueError("Existing fuzzy output is not a resumable run for this release")
        parts = list(output.glob("part-*.parquet"))
        if parts and any(prior.get(key) != value for key, value in manifest.items() if key not in ("status", "workers")):
            raise ValueError("Cannot change fuzzy retrieval settings after candidate shards were written")
    write_json(output / "manifest.json", manifest)
    started = time.time(); database = output / "fuzzy_index.sqlite"
    try:
        build_index(release, split, database)
        paths = sorted((release / split / "source1").glob("*.parquet"))
        pending = [(shard, path) for shard, path in enumerate(paths)
                   if not (output / f"part-{shard:05d}.parquet").exists()]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_shard, database, path, output, shard,
                                       max_key_frequency, preselect, cap,
                                       max_keys_per_anchor, batch_size)
                       for shard, path in pending]
            for future in as_completed(futures):
                shard, anchors, pairs = future.result()
                print(f"{split}: fuzzy shard {shard + 1}/{len(paths)}: {anchors:,} anchors; {pairs:,} candidates", flush=True)
        if len(list(output.glob("part-*.parquet"))) != len(paths):
            raise ValueError("Fuzzy candidate shard count is incomplete")
        total_anchors = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
        total_pairs = sum(pq.ParquetFile(path).metadata.num_rows for path in sorted(output.glob("part-*.parquet")))
        manifest.update({"status": "PASS", "anchors": total_anchors, "candidate_pairs": total_pairs,
                         "database_sha256": sha256(database), "elapsed_seconds": round(time.time()-started, 2)})
        write_json(output / "manifest.json", manifest)
        (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
    except BaseException as error:
        manifest.update({"status": "FAILED", "error": repr(error)})
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=50)
    parser.add_argument("--preselect", type=int, default=250)
    parser.add_argument("--max-key-frequency", type=int, default=5_000)
    parser.add_argument("--max-keys-per-anchor", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.release, args.split, args.output, args.cap, args.preselect, args.max_key_frequency,
        args.max_keys_per_anchor, args.batch_size, args.workers, args.resume)
