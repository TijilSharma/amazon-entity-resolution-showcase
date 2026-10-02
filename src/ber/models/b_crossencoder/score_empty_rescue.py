"""Score all US empty-rescue shards with one frozen B model instance."""
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
                    ('candidate_id', pa.string()), ('probability', pa.float32())])


def run(candidates, database, model_path, output, batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    parts = sorted(candidates.glob('part-*.parquet'))
    if not parts or len(parts) > 87:
        raise ValueError(f'Unexpected candidate shard count: {len(parts)}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda').eval()
    con = duckdb.connect(str(database), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='6GB'")
    total = 0
    with torch.inference_mode():
        for part in parts:
            result = output / part.name
            report_path = output / f'{part.stem}.json'
            input_hash = sha256(part)
            expected = pq.ParquetFile(part).metadata.num_rows
            if result.exists() and report_path.exists():
                report = json.loads(report_path.read_text())
                if report['input_sha256'] != input_hash or report['rows'] != expected or \
                        pq.ParquetFile(result).metadata.num_rows != expected:
                    raise ValueError(f'Stale score part: {part.name}')
                total += expected
                continue
            began = time.time()
            cursor = con.execute('SELECT r.*, t.country AS target_country '
                                 'FROM read_parquet(?) r JOIN targets t '
                                 'ON r.candidate_id=t.entity_id', [str(part)])
            temp = result.with_suffix('.parquet.tmp')
            seen = 0
            with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer:
                for arrow in cursor.fetch_record_batch(rows_per_batch=10_000):
                    raw = arrow.to_pylist()
                    for start in range(0, len(raw), batch_size):
                        group = raw[start:start + batch_size]
                        for row in group:
                            row['query_country'] = 'US'
                        pairs = [text_pair(row) for row in group]
                        encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                            padding=True, truncation=True, max_length=128,
                                            return_tensors='pt').to('cuda')
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits = model(**encoded).logits.reshape(-1)
                        probs = torch.sigmoid(logits.float()).cpu().tolist()
                        writer.write_table(pa.Table.from_pylist([
                            {'source1_entity_id': row['source1_entity_id'],
                             'candidate_id': row['candidate_id'], 'probability': float(prob)}
                            for row, prob in zip(group, probs)], schema=SCHEMA))
                        seen += len(group)
            if seen != expected:
                temp.unlink(missing_ok=True)
                raise ValueError(f'Country join lost candidates: {part.name}: {seen}/{expected}')
            temp.replace(result)
            report = {'part': part.name, 'rows': seen, 'input_sha256': input_hash,
                      'elapsed_seconds': round(time.time() - began, 2),
                      'model_path': str(model_path)}
            write_json(report_path, report)
            total += seen
            print(json.dumps(report), flush=True)
    con.close()
    if len(parts) == 87:
        write_json(output / 'manifest.json', {'status': 'PASS', 'parts': len(parts),
                                              'rows': total, 'model_path': str(model_path)})
        print(f'PASS: {len(parts)} shards, {total} scored pairs', flush=True)
    else:
        print(f'Partial: {len(parts)} shards, {total} scored pairs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path,
                        default=Path('candidates/us-empty-rescue-v1/test'))
    parser.add_argument('--database', type=Path,
                        default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    parser.add_argument('--model', type=Path,
                        default=Path('artifacts/b_anushka/crossencoder-full-epoch-v2/model'))
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/b_anushka/us-empty-rescue-b-v1'))
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    run(args.candidates, args.database, args.model, args.output, args.batch_size)
