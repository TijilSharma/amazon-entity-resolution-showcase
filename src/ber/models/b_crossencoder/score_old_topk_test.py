"""Score existing top-k test candidates with the frozen multilingual model."""
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


SCHEMA = pa.schema([('source1_entity_id', pa.string()), ('candidate_id', pa.string()),
                    ('candidate_rank', pa.int32()), ('probability', pa.float32())])


def score_part(part, release, database, model_path, output, top_k=10, batch_size=128):
    output.mkdir(parents=True, exist_ok=True)
    result = output / part.name
    report_path = output / (part.stem + '.json')
    if result.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if report['input_sha256'] == sha256(part) and report['top_k'] == top_k:
            return report
        raise ValueError(f'Existing score does not match input or top_k: {part.name}')
    if not torch.cuda.is_available():
        raise RuntimeError('Old-candidate B scoring requires the laptop GPU')
    query_part = release / 'test' / 'source1' / part.name
    if not query_part.exists():
        raise FileNotFoundError(query_part)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda').eval()
    con = duckdb.connect(str(database), read_only=True)
    con.execute('SET threads=4')
    con.execute("SET memory_limit='6GB'")
    expected = con.execute('SELECT count(*) FROM read_parquet(?) WHERE candidate_rank<=?',
                           [str(part), top_k]).fetchone()[0]
    sql = '''SELECT c.source1_entity_id,c.candidate_id,c.candidate_rank,
                    q.name_normalized AS query_name,
                    q.address_normalized AS query_address,q.country AS query_country,
                    t.name_normalized AS target_name,
                    t.address_normalized AS target_address,t.country AS target_country
             FROM read_parquet(?) c
             JOIN read_parquet(?) q ON c.source1_entity_id=q.entity_id
             JOIN targets t ON c.candidate_id=t.entity_id
             WHERE c.candidate_rank<=?'''
    cursor = con.execute(sql, [str(part), str(query_part), top_k])
    reader = cursor.fetch_record_batch(rows_per_batch=10_000)
    began = time.time()
    seen = 0
    temp = result.with_suffix('.parquet.tmp')
    with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer, torch.inference_mode():
        for arrow in reader:
            rows = arrow.to_pylist()
            for start in range(0, len(rows), batch_size):
                group = rows[start:start + batch_size]
                pairs = [text_pair(row) for row in group]
                encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                    padding=True, truncation=True, max_length=128,
                                    return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**encoded).logits.reshape(-1)
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                out = [{'source1_entity_id': row['source1_entity_id'],
                        'candidate_id': row['candidate_id'],
                        'candidate_rank': row['candidate_rank'],
                        'probability': float(prob)}
                       for row, prob in zip(group, probs)]
                writer.write_table(pa.Table.from_pylist(out, schema=SCHEMA))
                seen += len(group)
    con.close()
    if seen != expected:
        temp.unlink(missing_ok=True)
        raise ValueError(f'Join lost top-k candidates: {seen} != {expected}')
    temp.replace(result)
    report = {'part': part.name, 'top_k': top_k, 'rows': seen,
              'input_sha256': sha256(part), 'elapsed_seconds': round(time.time()-began, 2)}
    write_json(report_path, report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', type=Path, required=True)
    p.add_argument('--release', type=Path, default=Path('data/processed/conservative-v1-py311-u14'))
    p.add_argument('--database', type=Path, default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    p.add_argument('--model', type=Path,
                   default=Path('artifacts/b_anushka/crossencoder-full-epoch-v2/model'))
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/b_anushka/old-top10-test-probabilities-v1'))
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    a = p.parse_args()
    score_part(a.part, a.release, a.database, a.model, a.output, a.top_k, a.batch_size)
