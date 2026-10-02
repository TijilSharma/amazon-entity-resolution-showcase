"""Add frozen B probabilities to labeled rescue research pairs."""
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


def score_pairs(pairs, model_path, output, batch_size=128, swap=False):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda').eval()
    source = pq.ParquetFile(pairs)
    schema = source.schema_arrow.append(pa.field('B_probability', pa.float32()))
    temp = output / 'pairs.parquet.tmp'
    start_time = time.time()
    rows = 0
    with pq.ParquetWriter(temp, schema, compression='zstd') as writer, torch.inference_mode():
        for batch in source.iter_batches(batch_size=10_000):
            raw = batch.to_pylist()
            augmented = []
            for start in range(0, len(raw), batch_size):
                group = raw[start:start+batch_size]
                text = [text_pair(r) for r in group]
                if swap:
                    text = [(b, a) for a, b in text]
                enc = tokenizer([a for a, _ in text], [b for _, b in text],
                                padding=True, truncation=True, max_length=128,
                                return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**enc).logits.reshape(-1)
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                for row, prob in zip(group, probs):
                    row['B_probability'] = float(prob)
                    augmented.append(row)
                rows += len(group)
            writer.write_table(pa.Table.from_pylist(augmented, schema=schema))
    if rows != source.metadata.num_rows:
        temp.unlink(missing_ok=True)
        raise ValueError('Scoring row count mismatch')
    temp.replace(output / 'pairs.parquet')
    report = {'status': 'PASS', 'rows': rows,
              'swapped': swap,
              'input_sha256': sha256(pairs),
              'elapsed_seconds': round(time.time()-start_time, 2),
              'output_sha256': sha256(output / 'pairs.parquet')}
    write_json(output / 'manifest.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json')+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--model', type=Path,
                   default=Path('artifacts/b_anushka/crossencoder-full-epoch-v2/model'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--swap', action='store_true')
    a = p.parse_args()
    score_pairs(a.pairs, a.model, a.output, a.batch_size, a.swap)
