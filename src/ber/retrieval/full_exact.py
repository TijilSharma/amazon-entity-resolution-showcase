"""Resumable full-corpus lexical candidate generation.

This is the fast, conservative lexical route: exact normalized and
Latin-accent-folded name/address keys plus capped exact-name/address keys.
It is deliberately separate from fuzzy/TF-IDF retrieval; each emitted pair
retains its route provenance and is never called a final match.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json

VERSION = "full-exact-lexical-v1"
TARGET_COLUMNS = ["entity_id", "name_normalized", "address_normalized",
                  "name_accent_folded", "address_accent_folded"]


def rows(release, split, source, columns):
    for path in sorted((release / split / source).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=50_000):
            yield from batch.to_pylist()


def keys(row):
    name, address = row["name_normalized"], row["address_normalized"]
    folded_name, folded_address = row["name_accent_folded"], row["address_accent_folded"]
    if name and address:
        yield "normalized_name_address", name + "\x1f" + address
    if folded_name and folded_address:
        yield "accent_name_address", folded_name + "\x1f" + folded_address
    if name:
        yield "normalized_name", name
    if address:
        yield "normalized_address", address


def create_index(release, split, database, manifest):
    connection = sqlite3.connect(database)
    try:
        exists = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'").fetchone()
        if exists:
            count = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            indexed = connection.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='entries_lookup'").fetchone()
            if not indexed:
                print(f"Resuming lexical index creation from {count:,} stored rows", flush=True)
                connection.execute("CREATE INDEX entries_lookup ON entries(route, value)")
                connection.commit()
            manifest["index_rows"] = count
            write_json(database.with_suffix(".index_manifest.json"), manifest)
            return
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE entries (route TEXT NOT NULL, value TEXT NOT NULL, target_id TEXT NOT NULL);
        """)
        inserted = 0
        batch = []
        for source in ("source2", "source3"):
            for row in rows(release, split, source, TARGET_COLUMNS):
                batch.extend((route, value, row["entity_id"]) for route, value in keys(row))
                if len(batch) >= 100_000:
                    connection.executemany("INSERT INTO entries VALUES (?, ?, ?)", batch)
                    inserted += len(batch)
                    batch.clear()
                    connection.commit()
        if batch:
                connection.executemany("INSERT INTO entries VALUES (?, ?, ?)", batch)
                inserted += len(batch)
                batch.clear()
                connection.commit()
        connection.execute("CREATE INDEX entries_lookup ON entries(route, value)")
        connection.commit()
        manifest["index_rows"] = inserted
        write_json(database.with_suffix(".index_manifest.json"), manifest)
    finally:
        connection.close()


def generate(release, split, output, cap, resume=False):
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists: {output}; pass --resume only for an interrupted BUILDING run")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw_manifest = json.loads((release / "manifest.json").read_text())
    if raw_manifest.get("status") != "PASS" or (release / "_SUCCESS").read_text().strip() != sha256(release / "manifest.json"):
        raise ValueError("Input release is not complete")
    manifest = {"status": "BUILDING", "version": VERSION, "split": split,
                "candidate_cap_per_anchor": cap, "release_manifest_sha256": sha256(release / "manifest.json"),
                "sources": ["source2", "source3"], "routes": ["normalized_name_address", "accent_name_address", "normalized_name", "normalized_address"]}
    if resume and (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("status") not in {"BUILDING", "FAILED"} or previous.get("release_manifest_sha256") != manifest["release_manifest_sha256"]:
            raise ValueError("Only a matching incomplete release can be resumed")
    write_json(output / "manifest.json", manifest)
    database = output / "lexical_index.sqlite"
    try:
        create_index(release, split, database, manifest)
        connection = sqlite3.connect(database)
        total_anchors = total_pairs = 0
        try:
            for shard, path in enumerate(sorted((release / split / "source1").glob("*.parquet"))):
                part = output / f"part-{shard:05d}.parquet"
                if part.exists():
                    metadata = pq.ParquetFile(part).metadata.num_rows
                    total_anchors += pq.ParquetFile(path).metadata.num_rows
                    total_pairs += metadata
                    continue
                emitted = []
                for row in pq.ParquetFile(path).read(columns=TARGET_COLUMNS).to_pylist():
                    found, seen = [], set()
                    for route, value in keys(row):
                        # Capping each lookup prevents generic names from dominating the pool.
                        values = connection.execute("SELECT target_id FROM entries WHERE route=? AND value=? LIMIT ?", (route, value, cap)).fetchall()
                        for rank, (target,) in enumerate(values, 1):
                            if target not in seen and len(found) < cap:
                                seen.add(target)
                                found.append(target)
                                emitted.append({"source1_entity_id": row["entity_id"], "candidate_id": target,
                                                "route": route, "route_rank": rank, "score": 1.0})
                    total_anchors += 1
                pq.write_table(pa.Table.from_pylist(emitted, schema=pa.schema([
                    ("source1_entity_id", pa.string()), ("candidate_id", pa.string()),
                    ("route", pa.string()), ("route_rank", pa.int32()), ("score", pa.float32())])),
                    part, compression="zstd")
                total_pairs += len(emitted)
                if shard % 25 == 0:
                    print(f"{split}: {total_anchors:,} anchors; {total_pairs:,} lexical candidates", flush=True)
        finally:
            connection.close()
        manifest.update({"status": "PASS", "anchors": total_anchors, "candidate_pairs": total_pairs,
                         "elapsed_seconds": round(time.time()-started, 2),
                         "database_sha256": sha256(database)})
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    generate(args.release, args.split, args.output, args.cap, args.resume)
