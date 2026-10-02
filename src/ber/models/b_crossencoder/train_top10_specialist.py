"""Fine-tune the India specialist on train-role top-10 candidate pairs only."""
import argparse
import json
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import sha256, write_json
from ber.models.b_crossencoder.train_pilot import text_pair


def train(data, base, output, epochs=2, seed=1163):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    rows = pq.read_table(data).to_pylist()
    if len(rows) != 25_000 or sum(r['label'] for r in rows) < 5_000:
        raise ValueError('Unexpected top-10 training data')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    output.mkdir(parents=True)
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(rows)
        model.train()
        loss_sum = 0.0
        for start in range(0, len(rows), 16):
            group = rows[start:start + 16]
            pairs = [text_pair(r) for r in group]
            encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                padding=True, truncation=True, max_length=128,
                                return_tensors='pt').to('cuda')
            labels = torch.tensor([r['label'] for r in group],
                                  dtype=torch.float32, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(**encoded).logits.reshape(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu())
            step = start // 16 + 1
            if step % 100 == 0:
                print(f'epoch={epoch} step={step} loss={loss_sum/step:.5f}',
                      flush=True)
        path = output / f'epoch-{epoch}'
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        result = {'epoch': epoch, 'train_rows': len(rows),
                  'train_loss': loss_sum / ((len(rows) + 15) // 16),
                  'elapsed_seconds': round(time.time() - started, 2)}
        history.append(result)
        write_json(output / f'epoch-{epoch}.json', result)
        print(json.dumps(result), flush=True)
    report = {'status': 'PASS', 'base_model': str(base), 'train_sha256': sha256(data),
              'epochs': history}
    write_json(output / 'report.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'report.json') + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path,
                   default=Path('artifacts/b_anushka/india-top10-finetune-v1/train_pairs.parquet'))
    p.add_argument('--base', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-specialist-v1/model'))
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/b_anushka/india-top10-finetune-v1/training'))
    p.add_argument('--epochs', type=int, default=2)
    a = p.parse_args()
    train(a.data, a.base, a.output, a.epochs)
