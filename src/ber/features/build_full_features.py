"""Generate label-free pair features in bounded candidate batches.

The input is an ID-aligned record/candidate bundle from prepare_full_sample.
The same contract can later be used for test inference bundles. Truth is never
read by this module.
"""
import argparse
import json
import math
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json
from ber.features.comparisons import pair_features


ROUTES = ("accent_name_address", "normalized_address", "normalized_name",
          "normalized_name_address", "fuzzy_token_prefix", "e5_name_address_ivfpq")
IDENTIFIERS = ("source1_entity_id", "candidate_id")


def records(path):
    result = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        for row in batch.to_pylist():
            key = row["entity_id"]
            if key in result:
                raise ValueError(f"Duplicate record ID in {path}: {key}")
            result[key] = row
    return result


def route_features(row):
    names, ranks, scores = row["routes"], row["route_ranks"], row["route_scores"]
    if len(names) != len(ranks) or len(names) != len(scores) or len(names) != len(set(names)):
        raise ValueError("Malformed route provenance")
    evidence = dict(zip(names, zip(ranks, scores)))
    if not set(evidence) <= set(ROUTES):
        raise ValueError("Unknown candidate route")
    result = {"candidate_rank": float(row["candidate_rank"]),
              "route_count": float(len(evidence))}
    for route in ROUTES:
        hit = evidence.get(route)
        result[f"retrieved_{route}"] = float(hit is not None)
        result[f"retrieval_{route}_rank"] = float(hit[0]) if hit else math.nan
        result[f"retrieval_{route}_score"] = float(hit[1]) if hit else math.nan
    return result


def build(bundle, output, batch_size=50_000):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("status") != "PASS" or (bundle / "_SUCCESS").read_text().strip() != sha256(bundle / "manifest.json"):
        raise ValueError("Input bundle is incomplete")
    for name in ("queries.parquet", "targets.parquet", "candidates.parquet"):
        if sha256(bundle / name) != manifest["files"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    started = time.time()
    queries = records(bundle / "queries.parquet")
    targets = records(bundle / "targets.parquet")
    output.mkdir(parents=True)
    output_path = output / "pair_features.parquet"
    writer = None
    feature_names = None
    total = 0
    try:
        for batch in pq.ParquetFile(bundle / "candidates.parquet").iter_batches(batch_size=batch_size):
            emitted = []
            for candidate in batch.to_pylist():
                anchor, target = candidate["source1_entity_id"], candidate["candidate_id"]
                if anchor not in queries or target not in targets:
                    raise ValueError("Candidate refers to an unknown record")
                values = pair_features(queries[anchor], targets[target])
                values.update(route_features(candidate))
                if feature_names is None:
                    feature_names = list(values)
                elif list(values) != feature_names:
                    raise ValueError("Feature order changed between candidate pairs")
                if any(math.isinf(float(value)) for value in values.values()):
                    raise ValueError("Infinite feature value")
                emitted.append({"source1_entity_id": anchor, "candidate_id": target,
                                **{name: float(value) for name, value in values.items()}})
            if not emitted:
                continue
            if writer is None:
                schema = pa.schema([pa.field(name, pa.string(), nullable=False) for name in IDENTIFIERS] +
                                   [pa.field(name, pa.float32(), nullable=False) for name in feature_names])
                writer = pq.ParquetWriter(output_path, schema, compression="zstd")
            writer.write_table(pa.Table.from_pylist(emitted, schema=writer.schema))
            total += len(emitted)
            if total % 500_000 < len(emitted):
                print(f"Features: {total:,}/{manifest['candidate_pairs']:,} pairs", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if total != manifest["candidate_pairs"] or feature_names is None:
        raise ValueError("Feature output is incomplete")
    schema_doc = {"version": "full-pair-features-v1", "identifiers": list(IDENTIFIERS),
                  "features": feature_names, "numeric_dtype": "float32",
                  "label_column": None, "missing_value": "IEEE NaN"}
    write_json(output / "feature_schema.json", schema_doc)
    result = {"status": "PASS", "version": "full-pair-features-v1", "rows": total,
              "feature_count": len(feature_names), "batch_size": batch_size,
              "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
              "output_hashes": {name: sha256(output / name) for name in
                                ("pair_features.parquet", "feature_schema.json")},
              "elapsed_seconds": round(time.time() - started, 2)}
    write_json(output / "manifest.json", result)
    (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.output, args.batch_size), indent=2))
