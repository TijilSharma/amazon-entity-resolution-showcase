"""Union full-corpus candidate routes with deterministic per-route quotas.

Candidate scores from lexical and dense retrieval have different scales, so this
module never averages them. It takes each route's rank order, round-robins the
routes, deduplicates IDs, and preserves every contributing route as provenance.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json

VERSION = "full-candidate-union-v1"
INPUT_COLUMNS = ["source1_entity_id", "candidate_id", "route", "route_rank", "score"]


def validate_input(path, split):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "PASS" or (path / "_SUCCESS").read_text().strip() != sha256(path / "manifest.json"):
        raise ValueError(f"Incomplete route: {path}")
    if manifest.get("split") != split:
        raise ValueError(f"Wrong split for {path}")
    return manifest


def merge(lists, cap):
    """Return distinct candidates in fair rank-round-robin order."""
    result, seen, support = [], set(), defaultdict(list)
    by_route_rank = {}
    for route, values in lists.items():
        ranks = defaultdict(list)
        for row in values:
            support[row["candidate_id"]].append(row)
            ranks[row["route_rank"]].append(row)
        by_route_rank[route] = ranks
    maximum = max((max(ranks, default=0) for ranks in by_route_rank.values()), default=0)
    for rank in range(1, maximum + 1):
        for route in sorted(by_route_rank):
            for row in by_route_rank[route].get(rank, ()):
                if row["candidate_id"] in seen:
                    continue
                seen.add(row["candidate_id"])
                result.append((row["candidate_id"], support[row["candidate_id"]]))
                if len(result) == cap:
                    return result
    return result


def union(split, inputs, output, cap):
    if output.exists():
        raise FileExistsError(output)
    manifests = [validate_input(path, split) for path in inputs]
    release_hashes = {m.get("release_manifest_sha256") for m in manifests}
    if len(release_hashes) != 1:
        raise ValueError("Routes were built from different normalized releases")
    part_sets = [{p.name for p in path.glob("part-*.parquet")} for path in inputs]
    if len({tuple(sorted(parts)) for parts in part_sets}) != 1:
        raise ValueError("Routes do not have matching anchor shard layouts")
    output.mkdir(parents=True)
    total_anchors = total_pairs = 0
    for part_name in sorted(part_sets[0]):
        per_anchor = defaultdict(lambda: defaultdict(list))
        for path in inputs:
            table = pq.read_table(path / part_name, columns=INPUT_COLUMNS)
            for row in table.to_pylist():
                per_anchor[row["source1_entity_id"]][row["route"]].append(row)
        emitted = []
        for anchor in sorted(per_anchor):
            candidates = merge(per_anchor[anchor], cap)
            for rank, (candidate, evidence) in enumerate(candidates, 1):
                emitted.append({"source1_entity_id": anchor, "candidate_id": candidate,
                                "candidate_rank": rank,
                                "routes": sorted(row["route"] for row in evidence),
                                "route_ranks": [row["route_rank"] for row in sorted(evidence, key=lambda x: x["route"])],
                                "route_scores": [row["score"] for row in sorted(evidence, key=lambda x: x["route"]) ]})
        schema = pa.schema([("source1_entity_id", pa.string()), ("candidate_id", pa.string()),
                            ("candidate_rank", pa.int32()), ("routes", pa.list_(pa.string())),
                            ("route_ranks", pa.list_(pa.int32())), ("route_scores", pa.list_(pa.float32()))])
        pq.write_table(pa.Table.from_pylist(emitted, schema=schema), output / part_name, compression="zstd")
        total_anchors += len(per_anchor)
        total_pairs += len(emitted)
    manifest = {"status": "PASS", "version": VERSION, "split": split, "candidate_cap_per_anchor": cap,
                "release_manifest_sha256": release_hashes.pop(), "route_manifests": [sha256(path / "manifest.json") for path in inputs],
                "anchors_with_candidates": total_anchors, "candidate_pairs": total_pairs}
    write_json(output / "manifest.json", manifest)
    (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(union(args.split, args.inputs, args.output, args.cap), indent=2))
