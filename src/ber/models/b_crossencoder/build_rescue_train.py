"""Build labeled India rescue pairs from TRAIN-role anchors only."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json
from ber.retrieval.india_rescue import Rescue


SCHEMA = pa.schema([
    ('source1_entity_id', pa.string()), ('candidate_id', pa.string()),
    ('rescue_rank', pa.int32()), ('name_similarity', pa.float32()),
    ('address_similarity', pa.float32()), ('idf_evidence', pa.float32()),
    ('query_name', pa.string()), ('query_address', pa.string()),
    ('target_name', pa.string()), ('target_address', pa.string()),
    ('query_country', pa.string()), ('label', pa.int8())])


def build(sample, splits_path, index, aliases, output, n, preselect, role, country='India'):
    if output.exists():
        raise FileExistsError(output)
    splits = json.loads(splits_path.read_text())
    queries = [q for q in pq.read_table(sample / 'queries.parquet').to_pylist()
               if q['country'] == country and splits[q['entity_id']] == role]
    queries.sort(key=lambda q: hashlib.sha256(
        ('rescue-train-v1' + q['entity_id']).encode()).digest())
    queries = queries[:n]
    ids = {q['entity_id'] for q in queries}
    truth = {r['source1_entity_id']: set(r['matched_entity_ids'].split(','))
             if r['matched_entity_ids'] else set()
             for r in pq.read_table(sample / 'truth.parquet').to_pylist()
             if r['source1_entity_id'] in ids}
    excluded = defaultdict(set)
    for batch in pq.ParquetFile(sample / 'candidates.parquet').iter_batches(
            columns=['source1_entity_id', 'candidate_id']):
        for q, tid in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if q in ids:
                excluded[q].add(tid)
    rescue = Rescue(index, aliases, preselect=preselect)
    output.mkdir(parents=True)
    raw = output / 'pairs_raw.parquet'
    positives = total = 0
    with pq.ParquetWriter(raw, SCHEMA, compression='zstd') as writer:
        buffer = []
        for i, q in enumerate(queries, 1):
            qid = q['entity_id']
            for rank, (tid, ns, ads, evidence, name, address) in enumerate(
                    rescue.search(q, excluded[qid], 50), 1):
                label = int(tid in truth[qid])
                buffer.append({'source1_entity_id': qid, 'candidate_id': tid,
                               'rescue_rank': rank, 'name_similarity': ns,
                               'address_similarity': ads, 'idf_evidence': evidence,
                               'query_name': q['name_normalized'],
                               'query_address': q['address_normalized'],
                               'target_name': name, 'target_address': address,
                               'query_country': country, 'label': label})
                positives += label
                total += 1
            if len(buffer) >= 10_000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                buffer.clear()
            if i % 250 == 0:
                print(f'{i}/{len(queries)} anchors; {total} pairs; {positives} positives',
                      flush=True)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='6GB'")
    source = 'data/processed/conservative-v1-py311-u14/train/source[23]/part-*.parquet'
    joined = output / 'pairs.parquet'
    con.execute("COPY (SELECT p.*, t.country AS target_country "
                "FROM read_parquet(?) p JOIN read_parquet(?) t "
                "ON p.candidate_id=t.entity_id) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(joined), str(raw), source])
    con.close()
    if pq.ParquetFile(joined).metadata.num_rows != total:
        raise ValueError('Country join lost rescue training pairs')
    report = {'status': 'PASS', 'role': role, 'country': country, 'anchors': len(queries),
              'pairs': total, 'positives': positives, 'preselect': preselect,
              'aliases_sha256': sha256(aliases), 'pairs_sha256': sha256(joined)}
    write_json(output / 'manifest.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=Path, default=Path('artifacts/a_tijil/full-train-sample-v1'))
    p.add_argument('--splits', type=Path,
                   default=Path('artifacts/a_tijil/full-catboost-depth8-dev/splits.json'))
    p.add_argument('--index', type=Path,
                   default=Path('candidates/full-fuzzy-selective-v2/train/fuzzy_index.sqlite'))
    p.add_argument('--aliases', type=Path,
                   default=Path('artifacts/a_tijil/india-recall-research-v2/aliases.json'))
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-train-v1'))
    p.add_argument('--n', type=int, default=2500)
    p.add_argument('--preselect', type=int, default=2000)
    p.add_argument('--role', choices=('train', 'calibration', 'early_stop', 'holdout'),
                   default='train')
    p.add_argument('--country', choices=('US', 'India'), default='India')
    a = p.parse_args()
    build(a.sample, a.splits, a.index, a.aliases, a.output, a.n, a.preselect, a.role,
          a.country)
