"""Score capped test candidates in independent shards and write a valid submission."""
import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool

from ber.data.build_normalized import sha256, write_json
from ber.features.build_full_features import route_features
from ber.features.comparisons import pair_features


RECORD = ("entity_id", "country", "name_normalized", "address_normalized",
          "name_accent_folded", "address_accent_folded", "name_scripts", "address_scripts")


def checked(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "PASS" or (path / "_SUCCESS").read_text().strip() != sha256(path / "manifest.json"):
        raise ValueError(f"Incomplete artifact: {path}")
    return manifest


def prepare(release, database):
    checked(release)
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        raise FileExistsError(database)
    con = duckdb.connect(str(database))
    con.execute("SET threads=6")
    cols = ", ".join(RECORD)
    glob = str(release / "test" / "source[23]" / "part-*.parquet")
    con.execute(f"CREATE TABLE targets AS SELECT {cols} FROM read_parquet(?)", [glob])
    count, unique_count = con.execute("SELECT count(*), count(DISTINCT entity_id) FROM targets").fetchone()
    if (count, unique_count) != (9_969_589, 9_969_589):
        raise ValueError(f"Unexpected target counts: {count}, {unique_count}")
    con.execute("CHECKPOINT")
    con.close()
    (database.parent / (database.name + ".ready")).write_text(str(count) + "\n")
    return count


def score_part(args):
    part, release, union, database, model_dir, output, batch_size = args
    result = output / part.name
    if result.exists() and (output / (part.stem + ".json")).exists():
        return json.loads((output / (part.stem + ".json")).read_text())
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "model.cbm"))
    names = json.loads((model_dir / "feature_names.json").read_text())
    threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]
    con = duckdb.connect(str(database), read_only=True)
    con.execute("SET threads=1")
    con.execute("SET memory_limit='4GB'")
    qpart = release / "test" / "source1" / part.name
    qcols = ", ".join(f'q.{c} AS q_{c}' for c in RECORD)
    tcols = ", ".join(f't.{c} AS t_{c}' for c in RECORD)
    sql = (f"SELECT c.*, {qcols}, {tcols} FROM read_parquet(?) c "
           "JOIN read_parquet(?) q ON c.source1_entity_id=q.entity_id "
           "JOIN targets t ON c.candidate_id=t.entity_id")
    cursor = con.execute(sql, [str(part), str(qpart)])
    reader = cursor.fetch_record_batch(rows_per_batch=batch_size)
    schema = pa.schema([("source1_entity_id", pa.string()), ("candidate_id", pa.string()),
                        ("probability", pa.float32())])
    tmp = output / (part.name + ".tmp")
    total = accepted = 0
    began = time.time()
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in reader:
            rows = batch.to_pylist()
            if not rows:
                continue
            matrix = np.empty((len(rows), len(names)), dtype=np.float32)
            for i, row in enumerate(rows):
                query = {c: row["q_" + c] for c in RECORD}
                target = {c: row["t_" + c] for c in RECORD}
                values = pair_features(query, target)
                values.update(route_features(row))
                matrix[i] = [float(values[name]) for name in names]
            if np.isinf(matrix).any():
                raise ValueError(f"Infinite feature in {part}")
            probabilities = model.predict_proba(Pool(matrix, feature_names=names))[:, 1]
            emitted = [{"source1_entity_id": row["source1_entity_id"],
                        "candidate_id": row["candidate_id"], "probability": float(prob)}
                       for row, prob in zip(rows, probabilities) if prob >= threshold]
            if emitted:
                writer.write_table(pa.Table.from_pylist(emitted, schema=schema))
            total += len(rows)
            accepted += len(emitted)
    expected = pq.ParquetFile(part).metadata.num_rows
    if total != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Join lost rows for {part}: {total} != {expected}")
    os.replace(tmp, result)
    report = {"part": part.name, "candidate_pairs": total, "accepted_pairs": accepted,
              "elapsed_seconds": round(time.time() - began, 2)}
    write_json(output / (part.stem + ".json"), report)
    con.close()
    return report


def score(release, union, database, model, output, workers, batch_size, external_range=None):
    checked(union)
    if (model / "_SUCCESS").read_text().strip() != sha256(model / "report.json"):
        raise ValueError(f"Incomplete model: {model}")
    if not (database.parent / (database.name + ".ready")).exists():
        raise ValueError("Target lookup incomplete")
    output.mkdir(parents=True, exist_ok=True)
    parts = sorted(union.glob("part-*.parquet"))
    if not parts:
        raise ValueError("No candidate shards")
    external = set()
    if external_range:
        lo, hi = map(int, external_range.split(":"))
        external = {f"part-{i:05d}.parquet" for i in range(lo, hi + 1)}
        if not external <= {part.name for part in parts}:
            raise ValueError("External range includes missing shards")
    jobs = [(part, release, union, database, model, output, batch_size)
            for part in parts if part.name not in external]
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_part, job) for job in jobs]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(f"Scored locally {len(reports)}/{len(jobs)}: {report}", flush=True)
    deadline = time.time() + 4 * 3600
    for part in parts:
        if part.name not in external:
            continue
        report_path = output / (part.stem + ".json")
        while not (output / part.name).exists() or not report_path.exists():
            if time.time() > deadline:
                raise TimeoutError(f"Waiting for laptop-scored shard {part.name}")
            time.sleep(15)
        report = json.loads(report_path.read_text())
        if report["candidate_pairs"] != pq.ParquetFile(part).metadata.num_rows:
            raise ValueError(f"External shard row count mismatch: {part.name}")
        reports.append(report)
    if sum(r["candidate_pairs"] for r in reports) != checked(union)["candidate_pairs"]:
        raise ValueError("Scored candidate count mismatch")
    manifest = {"status": "PASS", "version": "catboost-full-test-v1", "parts": len(parts),
                "candidate_pairs": sum(r["candidate_pairs"] for r in reports),
                "accepted_pairs": sum(r["accepted_pairs"] for r in reports),
                "model_report_sha256": sha256(model / "report.json"),
                "union_manifest_sha256": sha256(union / "manifest.json")}
    write_json(output / "manifest.json", manifest)
    (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
    return manifest


def finalize(release, union, scores, output):
    checked(scores)
    checked(union)
    output.mkdir(parents=True, exist_ok=True)
    matching = output / "matching_results.tsv"
    candidates = output / "candidate_pairs.tsv"
    rows = matched = 0
    with matching.open("w", newline="") as mf, candidates.open("w", newline="") as cf:
        mw, cw = csv.writer(mf, delimiter="\t"), csv.writer(cf, delimiter="\t")
        mw.writerow(("source1_entity_id", "matched_entity_ids"))
        cw.writerow(("source1_entity_id", "candidate_entity_ids"))
        for part in sorted((release / "test" / "source1").glob("part-*.parquet")):
            union_part = union / part.name
            score_part_path = scores / part.name
            candidate_map = {}
            if union_part.exists():
                for row in pq.read_table(union_part, columns=["source1_entity_id", "candidate_id", "candidate_rank"]).to_pylist():
                    candidate_map.setdefault(row["source1_entity_id"], []).append((row["candidate_rank"], row["candidate_id"]))
            match_map = {}
            if score_part_path.exists():
                for row in pq.read_table(score_part_path).to_pylist():
                    match_map.setdefault(row["source1_entity_id"], []).append((row["probability"], row["candidate_id"]))
            for batch in pq.ParquetFile(part).iter_batches(columns=["entity_id"], batch_size=50_000):
                for anchor in batch.column(0).to_pylist():
                    picks = [target for _, target in sorted(match_map.get(anchor, ()), reverse=True)]
                    pool = [target for _, target in sorted(candidate_map.get(anchor, ()))]
                    if not set(picks) <= set(pool):
                        raise ValueError(f"Predictions not a subset of candidates: {anchor}")
                    mw.writerow((anchor, ",".join(picks)))
                    cw.writerow((anchor, ",".join(pool)))
                    rows += 1
                    matched += bool(picks)
            print(f"Wrote {part.name}: {rows:,} anchors total", flush=True)
    if rows != 1_732_544:
        raise ValueError(f"Wrong number of anchors: {rows}")
    report = {"status": "PASS", "rows": rows, "anchors_with_matches": matched,
              "matching_sha256": sha256(matching), "candidate_sha256": sha256(candidates)}
    write_json(output / "manifest.json", report)
    (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score", "score-part", "finalize"))
    parser.add_argument("--release", type=Path, default=Path("data/processed/conservative-v1-py311-u14"))
    parser.add_argument("--union", type=Path, default=Path("candidates/full-union-v1/test"))
    parser.add_argument("--database", type=Path, default=Path("artifacts/a_tijil/test-lookup-v1.duckdb"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/a_tijil/full-catboost-depth8-dev"))
    parser.add_argument("--scores", type=Path, default=Path("artifacts/a_tijil/full-test-scores-v1"))
    parser.add_argument("--output", type=Path, default=Path("output/a_tijil/first-submission-v1"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--part", type=Path)
    parser.add_argument("--external-range", help="Inclusive shard indices supplied by another machine, e.g. 20:49")
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.release, args.database))
    elif args.action == "score":
        print(score(args.release, args.union, args.database, args.model, args.scores,
                    args.workers, args.batch_size, args.external_range))
    elif args.action == "score-part":
        if args.part is None:
            parser.error("--part is required for score-part")
        args.scores.mkdir(parents=True, exist_ok=True)
        print(score_part((args.part, args.release, args.union, args.database,
                          args.model, args.scores, args.batch_size)))
    else:
        print(finalize(args.release, args.union, args.scores, args.output))
