"""Score additional India rescue pairs with the frozen Pipeline B cross-encoder."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import sha256, write_json
from ber.models.b_crossencoder.train_pilot import text_pair


SCHEMA = pa.schema([('source1_entity_id', pa.string()), ('candidate_id', pa.string()),
                    ('probability', pa.float32())])


def run(part, database, model_path, output, batch_size=128):
    output.mkdir(parents=True, exist_ok=True)
    result = output / part.name
    report_path = output / (part.stem + '.json')
    if result.exists() and report_path.exists():
        return json.loads(report_path.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('Pipeline B rescue scoring requires the laptop GPU')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda').eval()
    con = duckdb.connect(str(database), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='6GB'")
    cursor = con.execute('SELECT r.*, t.country AS target_country FROM read_parquet(?) r '
                         'JOIN targets t ON r.candidate_id=t.entity_id', [str(part)])
    reader = cursor.fetch_record_batch(rows_per_batch=10_000)
    seen = accepted = 0
    countries = Counter()
    began = time.time()
    temp = result.with_suffix('.parquet.tmp')
    with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer, torch.inference_mode():
        for arrow in reader:
            rows = arrow.to_pylist()
            for start in range(0, len(rows), batch_size):
                group = rows[start:start + batch_size]
                for row in group:
                    row['query_country'] = 'India'
                    countries[row['target_country']] += 1
                pairs = [text_pair(row) for row in group]
                encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                    padding=True, truncation=True, max_length=128,
                                    return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**encoded).logits.reshape(-1)
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                out = [{'source1_entity_id': row['source1_entity_id'],
                        'candidate_id': row['candidate_id'], 'probability': float(prob)}
                       for row, prob in zip(group, probs)]
                if out:
                    writer.write_table(pa.Table.from_pylist(out, schema=SCHEMA))
                seen += len(group)
                accepted += sum(prob >= 0.7218412160873413 for prob in probs)
    con.close()
    expected = pq.ParquetFile(part).metadata.num_rows
    if seen != expected:
        temp.unlink(missing_ok=True)
        raise ValueError(f'Country join lost candidates: {seen} != {expected}')
    temp.replace(result)
    report = {'part': part.name, 'candidates': seen,
              'rows_with_probabilities': seen,
              'above_original_calibration_threshold': accepted,
              'target_countries': dict(countries),
              'elapsed_seconds': round(time.time() - began, 2),
              'input_sha256': sha256(part)}
    write_json(report_path, report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', type=Path, required=True)
    p.add_argument('--database', type=Path, default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    p.add_argument('--model', type=Path,
                   default=Path('artifacts/b_anushka/crossencoder-full-epoch-v2/model'))
    p.add_argument('--output', type=Path, default=Path('artifacts/b_anushka/india-rescue-test-probabilities-v2'))
    p.add_argument('--batch-size', type=int, default=128)
    a = p.parse_args()
    run(a.part, a.database, a.model, a.output, a.batch_size)
