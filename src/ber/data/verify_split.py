"""Check shared split hashes, release identity and membership counts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pyarrow.parquet as pq
from .build_normalized import sha256


def verify(root, release):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'PASS' or (root / '_SUCCESS').read_text().strip() != sha256(root / 'manifest.json'):
        raise ValueError('Split is incomplete or manifest differs')
    if manifest['release_manifest_sha256'] != sha256(release / 'manifest.json'):
        raise ValueError('Split belongs to a different normalized release')
    for name, expected in manifest['files'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Split file changed: {name}')
    counts = Counter()
    for batch in pq.ParquetFile(root / 'anchor_membership.parquet').iter_batches(columns=['split']):
        counts.update(batch.column(0).to_pylist())
    if dict(counts) != manifest['counts']:
        raise ValueError('Anchor counts differ')
    if pq.ParquetFile(root / 'record_membership.parquet').metadata.num_rows != manifest['owned_records']:
        raise ValueError('Owned-record count differs')
    if manifest['cross_partition_signature_collisions'] != 0:
        raise ValueError('Split contains known signature leakage')
    result = {'status':'PASS','counts':dict(counts),'manifest_sha256':sha256(root/'manifest.json')}
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',required=True,type=Path)
    p.add_argument('--release',required=True,type=Path)
    a=p.parse_args()
    verify(a.split,a.release)
