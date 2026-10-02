"""Verify every normalized Parquet shard after copying a release to a worker."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from ber.data.build_normalized import sha256


def verify(release, split):
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest.get("status") != "PASS" or (release / "_SUCCESS").read_text().strip() != sha256(release / "manifest.json"):
        raise ValueError("Normalized release manifest is incomplete")
    summary = {"split": split, "status": "PASS", "sources": {}}
    for source in ("source1", "source2", "source3"):
        entry = manifest["sources"][f"{split}_{source}.tsv"]
        listed = {row["file"] for row in entry["shards"]}
        directory = release / split / source
        actual = {path.name for path in directory.glob("*.parquet")}
        if actual != listed:
            raise ValueError(f"Missing or unexpected {source} shards: {len(actual)} vs {len(listed)}")
        rows = 0
        for item in entry["shards"]:
            path = directory / item["file"]
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            count = pq.ParquetFile(path).metadata.num_rows
            if count != item["rows"]:
                raise ValueError(f"Row count mismatch: {path}")
            rows += count
        if rows != entry["rows"]:
            raise ValueError(f"Source row count mismatch: {source}")
        summary["sources"][source] = {"shards": len(listed), "rows": rows}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.release, args.split), indent=2))
