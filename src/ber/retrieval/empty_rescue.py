"""Field-balanced rescue for currently unmatched US test entities.

The selected IDs come only from model predictions; labels are never read.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ber.retrieval.india_rescue import Rescue, SCHEMA


def prepare_ids(release, matching, output):
    output.mkdir(parents=True, exist_ok=True)
    count = total = 0
    parts = sorted((release / 'test' / 'source1').glob('part-*.parquet'))
    with matching.open(newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        if next(reader) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected matching header')
        for part in parts:
            selected = []
            for batch in pq.ParquetFile(part).iter_batches(columns=['entity_id', 'country']):
                for row in batch.to_pylist():
                    prior = next(reader)
                    if prior[0] != row['entity_id']:
                        raise ValueError(f'Matching order differs at {row["entity_id"]}')
                    if row['country'] == 'US' and not prior[1]:
                        selected.append(row['entity_id'])
                    total += 1
            (output / f'{part.stem}.json').write_text(json.dumps(selected))
            count += len(selected)
        if next(reader, None) is not None:
            raise ValueError('Extra matching rows')
    print(json.dumps({'anchors': total, 'selected_us_empty': count,
                      'parts': len(parts)}), flush=True)


def run_shard(index, aliases, release, union, ids_dir, output, shard, cap, preselect):
    part = release / 'test' / 'source1' / f'part-{shard:05d}.parquet'
    destination = output / part.name
    if destination.exists():
        print(f'{part.name}: already complete', flush=True)
        return
    selected = set(json.loads((ids_dir / f'{part.stem}.json').read_text()))
    queries = {}
    for batch in pq.ParquetFile(part).iter_batches(columns=['entity_id', 'country',
                                                              'name_normalized', 'address_normalized']):
        for row in batch.to_pylist():
            if row['entity_id'] in selected:
                if row['country'] != 'US':
                    raise ValueError('Selected non-US query')
                queries[row['entity_id']] = row
    if set(queries) != selected:
        raise ValueError('Missing selected query')
    excluded = defaultdict(set)
    if selected:
        for batch in pq.ParquetFile(union / part.name).iter_batches(
                columns=['source1_entity_id', 'candidate_id']):
            for query, target in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if query in selected:
                    excluded[query].add(target)
    rescue = Rescue(index, aliases, preselect=preselect)
    rows = []
    for i, (anchor, query) in enumerate(queries.items(), 1):
        for rank, (target, ns, ads, evidence, name, address) in enumerate(
                rescue.search(query, excluded[anchor], cap), 1):
            rows.append({'source1_entity_id': anchor, 'candidate_id': target,
                         'rescue_rank': rank, 'name_similarity': ns,
                         'address_similarity': ads, 'idf_evidence': evidence,
                         'query_name': query['name_normalized'],
                         'query_address': query['address_normalized'],
                         'target_name': name, 'target_address': address})
        if i % 250 == 0:
            print(f'{part.name}: {i}/{len(queries)} queries', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temp, compression='zstd')
    temp.replace(destination)
    print(f'{part.name}: PASS {len(queries)} queries, {len(rows)} pairs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run-shard'))
    parser.add_argument('--release', type=Path,
                        default=Path('data/processed/conservative-v1-py311-u14'))
    parser.add_argument('--matching', type=Path)
    parser.add_argument('--ids-dir', type=Path,
                        default=Path('artifacts/a_tijil/us-empty-ids-v1'))
    parser.add_argument('--index', type=Path,
                        default=Path('candidates/full-fuzzy-selective-v2/test/fuzzy_index.sqlite'))
    parser.add_argument('--aliases', type=Path,
                        default=Path('artifacts/a_tijil/india-recall-research-v2/aliases.json'))
    parser.add_argument('--union', type=Path, default=Path('candidates/full-union-v1/test'))
    parser.add_argument('--output', type=Path,
                        default=Path('candidates/us-empty-rescue-v1/test'))
    parser.add_argument('--shard', type=int)
    parser.add_argument('--cap', type=int, default=50)
    parser.add_argument('--preselect', type=int, default=1000)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.matching is None:
            parser.error('--matching is required for prepare')
        prepare_ids(args.release, args.matching, args.ids_dir)
    else:
        if args.shard is None:
            parser.error('--shard is required for run-shard')
        run_shard(args.index, args.aliases, args.release, args.union,
                  args.ids_dir, args.output, args.shard, args.cap, args.preselect)
