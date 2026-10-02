"""Evaluate a frozen Apache-2.0 multilingual reranker on labeled India hard pairs."""
import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import write_json
from ber.models.b_crossencoder.train_pilot import text_pair


def run(model_id, output, batch_size, min_b_probability):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=torch.float16).to('cuda').eval()
    counts = {}
    with torch.inference_mode():
        for role in ('calibration', 'early-stop', 'holdout'):
            source = Path(f'artifacts/b_anushka/india-rescue-{role}-specialist-v1/pairs.parquet')
            table = pq.read_table(source)
            rows = [row for row in table.to_pylist()
                    if row['B_probability'] >= min_b_probability]
            schema = table.schema.append(pa.field('bge_logit', pa.float32()))
            began = time.time()
            for offset in range(0, len(rows), batch_size):
                group = rows[offset:offset + batch_size]
                pairs = [text_pair(row) for row in group]
                enc = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                padding=True, truncation=True, max_length=128,
                                return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**enc).logits.reshape(-1).float().cpu().tolist()
                for row, logit in zip(group, logits):
                    row['bge_logit'] = float(logit)
                if offset % 512 == 0:
                    print(f'{role}: {offset + len(group)}/{len(rows)}', flush=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                           output / f'{role}.parquet', compression='zstd')
            counts[role] = {'pairs': len(rows), 'positives': sum(row['label'] for row in rows),
                            'seconds': round(time.time() - began, 2)}
            print(json.dumps({role: counts[role]}), flush=True)
    write_json(output / 'manifest.json', {'status': 'PASS', 'model_id': model_id,
                                          'min_b_probability': min_b_probability,
                                          'roles': counts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/b_anushka/bge-second-opinion-v1'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--min-b-probability', type=float, default=0.00008)
    args = parser.parse_args()
    run(args.model, args.output, args.batch_size, args.min_b_probability)
