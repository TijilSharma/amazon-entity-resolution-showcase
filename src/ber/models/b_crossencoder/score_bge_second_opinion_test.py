"""Frozen BGE second opinion for India rescue pairs uncertain under specialist B."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import sha256, write_json
from ber.models.b_crossencoder.train_pilot import text_pair


SCHEMA = pa.schema([('source1_entity_id', pa.string()),
                    ('candidate_id', pa.string()), ('B_probability', pa.float32()),
                    ('bge_logit', pa.float32())])


def run(routes, scores, database, model_id, output, min_b, max_b, batch_size):
    if len(routes) != len(scores) or len(routes) != 3:
        raise ValueError('Three route and score directories are required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=torch.float16).to('cuda').eval()
    con = duckdb.connect(str(database), read_only=True)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    totals = {}
    with torch.inference_mode():
        for name, route, specialist in zip(('zero', 'one_two', 'three_plus'), routes, scores):
            destination = output / name
            destination.mkdir(parents=True, exist_ok=True)
            parts = sorted(route.glob('part-*.parquet'))
            if len(parts) != 87:
                raise ValueError(f'{name}: expected 87 route shards')
            count = 0
            for part in parts:
                score_part = specialist / part.name
                result = destination / part.name
                report_path = destination / f'{part.stem}.json'
                route_hash = sha256(part)
                specialist_hash = sha256(score_part)
                if result.exists() and report_path.exists():
                    report = json.loads(report_path.read_text())
                    if report['route_sha256'] != route_hash or \
                            report['specialist_sha256'] != specialist_hash or \
                            report['rows'] != pq.ParquetFile(result).metadata.num_rows:
                        raise ValueError(f'Stale BGE score shard: {name}/{part.name}')
                    count += report['rows']
                    continue
                began = time.time()
                cursor = con.execute(
                    'SELECT r.*,s.probability AS B_probability,t.country AS target_country '
                    'FROM read_parquet(?) s JOIN read_parquet(?) r '
                    'USING(source1_entity_id,candidate_id) '
                    'JOIN targets t ON t.entity_id=s.candidate_id '
                    'WHERE s.probability>=? AND s.probability<?',
                    [str(score_part), str(part), min_b, max_b])
                temp = result.with_suffix('.parquet.tmp')
                seen = 0
                with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer:
                    for batch in cursor.fetch_record_batch(rows_per_batch=10_000):
                        raw = batch.to_pylist()
                        for start in range(0, len(raw), batch_size):
                            group = raw[start:start + batch_size]
                            for row in group:
                                row['query_country'] = 'India'
                            pairs = [text_pair(row) for row in group]
                            encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                                padding=True, truncation=True,
                                                max_length=128, return_tensors='pt').to('cuda')
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = model(**encoded).logits.reshape(-1).float().cpu().tolist()
                            writer.write_table(pa.Table.from_pylist([
                                {'source1_entity_id': row['source1_entity_id'],
                                 'candidate_id': row['candidate_id'],
                                 'B_probability': row['B_probability'],
                                 'bge_logit': float(logit)}
                                for row, logit in zip(group, logits)], schema=SCHEMA))
                            seen += len(group)
                temp.replace(result)
                report = {'route': name, 'part': part.name, 'rows': seen,
                          'route_sha256': route_hash,
                          'specialist_sha256': specialist_hash,
                          'elapsed_seconds': round(time.time() - began, 2)}
                write_json(report_path, report)
                count += seen
                print(json.dumps(report), flush=True)
            totals[name] = count
    con.close()
    write_json(output / 'manifest.json', {'status': 'PASS', 'model_id': model_id,
                                          'min_b_probability': min_b,
                                          'max_b_probability': max_b,
                                          'routes': totals})
    print(json.dumps({'status': 'PASS', 'routes': totals}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', type=Path, action='append', default=[])
    parser.add_argument('--specialist-scores', type=Path, action='append', default=[])
    parser.add_argument('--database', type=Path,
                        default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    parser.add_argument('--model', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/b_anushka/bge-second-opinion-test-v1'))
    parser.add_argument('--min-b-probability', type=float, default=0.01)
    parser.add_argument('--max-b-probability', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not args.route:
        args.route = [Path('candidates/india-rescue-v1/test'),
                      Path('candidates/india-rescue-1to2-v1/test'),
                      Path('candidates/india-rescue-3plus-v1/test')]
    if not args.specialist_scores:
        args.specialist_scores = [Path('artifacts/b_anushka/india-rescue-specialist-test-v1') / name
                                  for name in ('zero', 'one_two', 'three_plus')]
    run(args.route, args.specialist_scores, args.database, args.model,
        args.output, args.min_b_probability, args.max_b_probability,
        args.batch_size)
