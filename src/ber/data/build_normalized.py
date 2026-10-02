"""Stream audited source TSVs into a new, verified Parquet release.

Never overwrites a release or edits raw data. Each shard is read back and all
raw fields are compared before publication. A _SUCCESS marker is written only
when all source hashes, counts, shards and ground truth have passed.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from .normalize import normalize_record, NORMALIZATION_VERSION, UNICODE_VERSION
from .validate_inputs import EXPECTED_COLUMNS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    # A per-write name prevents two independently launched jobs from deleting
    # each other's temporary manifest before os.replace executes.
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def make_schema():
    example = normalize_record(dict.fromkeys(EXPECTED_COLUMNS, ''))
    fields = []
    for key, value in example.items():
        kind = pa.bool_() if isinstance(value, bool) else pa.list_(pa.string()) if isinstance(value, list) else pa.string()
        fields.append(pa.field(key, kind, nullable=False))
    fields.append(pa.field('source_row', pa.int64(), nullable=False))
    return pa.schema(fields)


def write_shard(rows, destination, start_row):
    pa.set_cpu_count(1)
    normalized = []
    stats = Counter()
    for i, row in enumerate(rows):
        result = normalize_record(row)
        result['source_row'] = start_row + i
        normalized.append(result)
        for key in ('name_blank', 'address_blank', 'name_placeholder', 'address_placeholder',
                    'name_missing', 'address_missing', 'address_null_component_removed'):
            stats[key] += result[key]
    table = pa.Table.from_pylist(normalized, schema=make_schema())
    path = Path(destination)
    temp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, temp, compression='zstd', compression_level=3,
                   row_group_size=10000, write_statistics=True)
    restored = pq.read_table(temp)
    if not restored.equals(table):
        raise ValueError(f'Parquet round-trip mismatch: {path}')
    # Independent comparison against the original records, not just derived data.
    expected = pa.Table.from_pylist(rows, schema=pa.schema([table.schema.field(k) for k in EXPECTED_COLUMNS]))
    if not restored.select(EXPECTED_COLUMNS).equals(expected):
        raise ValueError(f'Raw fields changed: {path}')
    temp.replace(path)
    return {'file': path.name, 'rows': len(rows), 'first_source_row': start_row,
            'sha256': sha256(path), 'bytes': path.stat().st_size, 'flags': dict(stats)}


def chunks(path, size):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', strict=True)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f'Unexpected header: {path}')
        batch = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed record near line {reader.line_num}: {path}')
            batch.append(row)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def build(data_root, output, audit_path, workers=3, shard_rows=20000):
    if workers < 1 or shard_rows < 1:
        raise ValueError('workers and shard_rows must be positive')
    audit = json.loads(audit_path.read_text())
    if audit.get('structural_status') != 'PASS':
        raise ValueError('A passing structural audit is required')
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {'status': 'BUILDING', 'normalization_version': NORMALIZATION_VERSION,
                'unicode_version': UNICODE_VERSION, 'python': platform.python_version(),
                'pyarrow': pa.__version__, 'data_root': str(data_root.resolve()),
                'normalizer_sha256': sha256(Path(__file__).with_name('normalize.py')),
                'builder_sha256': sha256(Path(__file__)), 'audit_sha256': sha256(audit_path),
                'shard_rows': shard_rows, 'workers': workers, 'sources': {}}
    write_json(output / 'manifest.json', manifest)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for split in ('train', 'test'):
                for source in (1, 2, 3):
                    name = f'{split}_source{source}.tsv'
                    path = data_root / split / name
                    audited = audit['sources'][name]
                    if sha256(path) != audited['sha256']:
                        raise ValueError(f'Raw file changed since audit: {path}')
                    directory = output / split / f'source{source}'
                    directory.mkdir(parents=True)
                    entry = {'raw_sha256': audited['sha256'], 'rows': 0, 'shards': [], 'flags': {}}
                    pending = deque()
                    count = 0
                    flags = Counter()

                    def collect():
                        info = pending.popleft().result()
                        entry['shards'].append(info)
                        entry['rows'] += info['rows']
                        flags.update(info['flags'])
                        if len(entry['shards']) % 25 == 0:
                            print(f'{name}: {entry["rows"]:,} verified', flush=True)

                    for number, rows in enumerate(chunks(path, shard_rows)):
                        target = directory / f'part-{number:05d}.parquet'
                        pending.append(pool.submit(write_shard, rows, str(target), count))
                        count += len(rows)
                        if len(pending) >= workers * 2:
                            collect()
                    while pending:
                        collect()
                    if entry['rows'] != audited['total_records']:
                        raise ValueError(f'Unexpected row count: {path}')
                    if sha256(path) != audited['sha256']:
                        raise ValueError(f'Raw file changed during processing: {path}')
                    entry['flags'] = dict(flags)
                    entry['status'] = 'PASS'
                    manifest['sources'][name] = entry
                    write_json(output / 'manifest.json', manifest)
                    print(f'COMPLETE {name}: {entry["rows"]:,}', flush=True)
        gt = data_root / 'train/train_ground_truth.tsv'
        if sha256(gt) != audit['ground_truth']['sha256']:
            raise ValueError('Ground truth changed since audit')
        manifest['ground_truth'] = {'path': str(gt.resolve()), 'sha256': audit['ground_truth']['sha256'],
                                    'policy': 'Unmodified raw ground truth; no labels embedded in record shards.'}
        manifest['status'] = 'PASS'
        manifest['total_records'] = sum(e['rows'] for e in manifest['sources'].values())
        manifest['elapsed_seconds'] = round(time.time() - started, 2)
        write_json(output / 'manifest.json', manifest)
        (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
        return manifest
    except BaseException as error:
        manifest['status'] = 'FAILED'
        manifest['error'] = repr(error)
        write_json(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--shard-rows', type=int, default=20000)
    args = parser.parse_args()
    result = build(args.data_root, args.output, args.audit, args.workers, args.shard_rows)
    print(f'PASS: {result["total_records"]:,} records; {result["elapsed_seconds"]} seconds')


if __name__ == '__main__':
    main()
