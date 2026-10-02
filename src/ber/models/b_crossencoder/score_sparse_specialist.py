"""Score only plausible India rescue pairs with the fine-tuned specialist."""
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
                    ('candidate_id', pa.string()),
                    ('probability', pa.float32())])


def score_part(con, tokenizer, model, part, base_part, output, prefilter, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    result = output / part.name
    report_path = output / (part.stem + '.json')
    input_hash = sha256(part)
    if result.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get('input_sha256') == input_hash and
                report.get('prefilter_min_B') == prefilter and
                report.get('score_mode') == 'sparse_prefilter'):
            return report
        raise ValueError(f'Existing specialist score does not match: {result}')
    base_report = json.loads((base_part.parent / (part.stem + '.json')).read_text())
    candidate_pairs = pq.ParquetFile(part).metadata.num_rows
    if base_report['input_sha256'] != input_hash or \
            pq.ParquetFile(base_part).metadata.num_rows != candidate_pairs:
        raise ValueError(f'Original B scores do not align with {part}')
    expected = con.execute('SELECT count(*) FROM read_parquet(?) WHERE probability>=?',
                           [str(base_part), prefilter]).fetchone()[0]
    cursor = con.execute('''SELECT r.*, t.country AS target_country
        FROM read_parquet(?) r
        JOIN read_parquet(?) b
          ON r.source1_entity_id=b.source1_entity_id AND r.candidate_id=b.candidate_id
        JOIN targets t ON r.candidate_id=t.entity_id
        WHERE b.probability>=?''', [str(part), str(base_part), prefilter])
    reader = cursor.fetch_record_batch(rows_per_batch=10_000)
    temp = result.with_suffix('.parquet.tmp')
    seen = accepted = 0
    began = time.time()
    with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer, torch.inference_mode():
        for arrow in reader:
            rows = arrow.to_pylist()
            for start in range(0, len(rows), batch_size):
                group = rows[start:start + batch_size]
                for row in group:
                    row['query_country'] = 'India'
                pairs = [text_pair(row) for row in group]
                encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                    padding=True, truncation=True, max_length=128,
                                    return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**encoded).logits.reshape(-1)
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                out = [{'source1_entity_id': row['source1_entity_id'],
                        'candidate_id': row['candidate_id'],
                        'probability': float(prob)}
                       for row, prob in zip(group, probs)]
                if out:
                    writer.write_table(pa.Table.from_pylist(out, schema=SCHEMA))
                seen += len(out)
                accepted += sum(prob >= .99 for prob in probs)
    if seen != expected:
        temp.unlink(missing_ok=True)
        raise ValueError(f'Specialist join lost pairs for {part}: {seen} != {expected}')
    temp.replace(result)
    report = {'part': part.name, 'score_mode': 'sparse_prefilter',
              'candidate_pairs': candidate_pairs, 'rows_with_probabilities': seen,
              'above_selected_threshold': accepted, 'prefilter_min_B': prefilter,
              'input_sha256': input_hash, 'base_score_sha256': sha256(base_part),
              'elapsed_seconds': round(time.time() - began, 2)}
    write_json(report_path, report)
    print(json.dumps(report), flush=True)
    return report


def run(routes, base_scores, outputs, database, model_path, prefilter, batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    if not (len(routes) == len(base_scores) == len(outputs)):
        raise ValueError('Each route requires one original score and output directory')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda').eval()
    con = duckdb.connect(str(database), read_only=True)
    con.execute('SET threads=4')
    con.execute("SET memory_limit='6GB'")
    try:
        for route, base, output in zip(routes, base_scores, outputs):
            parts = sorted(route.glob('part-*.parquet'))
            if len(parts) != 87:
                raise ValueError(f'Expected 87 parts in {route}, got {len(parts)}')
            for index, part in enumerate(parts, 1):
                report = score_part(con, tokenizer, model, part, base / part.name,
                                    output, prefilter, batch_size)
                print(f'{route.name}: {index}/87, scored {report["rows_with_probabilities"]}',
                      flush=True)
    finally:
        con.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route', type=Path, action='append', required=True)
    p.add_argument('--base-scores', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, action='append', required=True)
    p.add_argument('--database', type=Path,
                   default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    p.add_argument('--model', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-specialist-v1/model'))
    p.add_argument('--prefilter', type=float, default=8e-5)
    p.add_argument('--batch-size', type=int, default=128)
    a = p.parse_args()
    run(a.route, a.base_scores, a.output, a.database, a.model,
        a.prefilter, a.batch_size)
