"""Verify a shared normalized release without access to Tijil's original paths."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from .build_normalized import sha256


def verify(root, labels=None):
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'PASS' or (root / '_SUCCESS').read_text().strip() != sha256(manifest_path):
        raise ValueError('Release is incomplete or manifest checksum differs')
    total = 0
    for name, entry in manifest['sources'].items():
        split, source = name.removesuffix('.tsv').split('_')
        directory = root / split / source
        expected = {s['file'] for s in entry['shards']}
        if {p.name for p in directory.glob('*.parquet')} != expected:
            raise ValueError(f'Shard set differs: {name}')
        count = 0
        for shard in entry['shards']:
            path = directory / shard['file']
            if sha256(path) != shard['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            if pq.ParquetFile(path).metadata.num_rows != shard['rows'] or shard['first_source_row'] != count:
                raise ValueError(f'Row metadata mismatch: {path}')
            count += shard['rows']
        if count != entry['rows']:
            raise ValueError(f'Count mismatch: {name}')
        total += count
        print(f'PASS {name}: {count:,}', flush=True)
    if total != manifest['total_records']:
        raise ValueError('Total count differs')
    if labels and sha256(labels) != manifest['ground_truth']['sha256']:
        raise ValueError('Ground-truth checksum mismatch')
    return {'status': 'PASS', 'records': total, 'manifest_sha256': sha256(manifest_path),
            'labels_verified': labels is not None}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--labels', type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.release, a.labels), indent=2))
