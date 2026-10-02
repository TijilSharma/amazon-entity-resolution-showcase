"""Score only target IDs claimed by multiple Source-1 records."""
import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import sha256, write_json
from ber.models.b_crossencoder.train_pilot import text_pair


SCHEMA = pa.schema([('source1_entity_id', pa.string()),
                    ('candidate_id', pa.string()), ('probability', pa.float32())])


def run(input_path, output, model_path, batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    output.mkdir(parents=True, exist_ok=True)
    input_hash = sha256(input_path)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda').eval()
    total = 0
    with torch.inference_mode():
        for index, batch in enumerate(pq.ParquetFile(input_path).iter_batches(batch_size=10_000)):
            rows = batch.to_pylist()
            part = output / f'part-{index:05d}.parquet'
            report_path = part.with_suffix('.json')
            if part.exists() and report_path.exists():
                report = json.loads(report_path.read_text())
                if (report['input_sha256'] != input_hash or
                        report['rows'] != len(rows) or
                        pq.ParquetFile(part).metadata.num_rows != len(rows)):
                    raise ValueError(f'Stale conflict score: {part}')
                total += len(rows)
                continue
            began = time.time()
            scored = []
            for start in range(0, len(rows), batch_size):
                group = rows[start:start + batch_size]
                pairs = [text_pair(row) for row in group]
                encoded = tokenizer([x[0] for x in pairs], [x[1] for x in pairs],
                                    padding=True, truncation=True, max_length=128,
                                    return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**encoded).logits.reshape(-1)
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                scored.extend({'source1_entity_id': row['source1_entity_id'],
                               'candidate_id': row['candidate_id'],
                               'probability': float(prob)}
                              for row, prob in zip(group, probs))
            temp = part.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(scored, schema=SCHEMA), temp,
                           compression='zstd')
            temp.replace(part)
            report = {'index': index, 'rows': len(rows), 'input_sha256': input_hash,
                      'model_path': str(model_path),
                      'elapsed_seconds': round(time.time() - began, 2)}
            write_json(report_path, report)
            total += len(rows)
            print(f'Scored conflict part {index}: {len(rows)} rows', flush=True)
    if total != pq.ParquetFile(input_path).metadata.num_rows:
        raise ValueError('Conflict score row count mismatch')
    write_json(output / 'manifest.json', {'status': 'PASS', 'rows': total,
                                          'input_sha256': input_hash,
                                          'parts': index + 1,
                                          'model_path': str(model_path)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path,
                   default=Path('artifacts/a_tijil/conflict-resolution-v1/conflict_pairs.parquet'))
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/b_anushka/conflict-b-scores-v1'))
    p.add_argument('--model', type=Path,
                   default=Path('artifacts/b_anushka/crossencoder-full-epoch-v2/model'))
    p.add_argument('--batch-size', type=int, default=128)
    a = p.parse_args()
    run(a.input, a.output, a.model, a.batch_size)
