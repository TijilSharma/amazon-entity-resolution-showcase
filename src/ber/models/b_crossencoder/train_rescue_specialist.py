"""Fine-tune the frozen multilingual reranker on hard India rescue pairs.

Only train-role labels update weights. Calibration chooses a threshold, and
early-stop selects an epoch; neither split contributes training examples.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import sha256, write_json
from ber.evaluation.entity_score import score
from ber.models.b_crossencoder.train_pilot import text_pair


THRESHOLDS = (.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99)


def selected_train_rows(rows, seed):
    rng = random.Random(seed)
    positive_easy = [r for r in rows if r['label'] and r['B_probability'] >= .7]
    positive_hard = [r for r in rows if r['label'] and r['B_probability'] < .7]
    negative_hard = [r for r in rows if not r['label'] and r['B_probability'] >= .01]
    negative_easy = [r for r in rows if not r['label'] and r['B_probability'] < .01]
    selected = (positive_easy + positive_hard * 3 + negative_hard * 4 +
                rng.sample(negative_easy, min(8_000, len(negative_easy))))
    rng.shuffle(selected)
    return selected


def predict(model, tokenizer, rows, batch_size=128):
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            pairs = [text_pair(r) for r in group]
            encoded = tokenizer([a for a, _ in pairs], [b for _, b in pairs],
                                padding=True, truncation=True, max_length=128,
                                return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(**encoded).logits.reshape(-1)
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return probabilities


def evaluate_roles(model, tokenizer, evaluation, truth, old, base_scores):
    probabilities = {role: predict(model, tokenizer, rows)
                     for role, rows in evaluation.items()}

    def metric(role, values, threshold):
        chosen = {q: set(old[role][q]) for q in truth[role]}
        for row, probability in zip(evaluation[role], values):
            if probability >= threshold:
                chosen[row['source1_entity_id']].add(row['candidate_id'])
        result = score(truth[role], {q: list(v) for q, v in chosen.items()})
        return {k: result[k] for k in ('macro_f05', 'tp', 'fp', 'fn')}

    threshold = max(THRESHOLDS,
                    key=lambda t: (metric('calibration', probabilities['calibration'], t)['macro_f05'], t))
    return {'threshold': threshold,
            'calibration': metric('calibration', probabilities['calibration'], threshold),
            'early_stop': metric('early_stop', probabilities['early_stop'], threshold),
            'original_B_calibration': metric('calibration', base_scores['calibration'], threshold),
            'original_B_early_stop': metric('early_stop', base_scores['early_stop'], threshold)}


def train(train_data, calibration_data, early_stop_data, model_path, sample,
          a_predictions, a_threshold_path, output,
          epochs=2, seed=2811):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    began = time.time()
    training = pq.read_table(train_data / 'pairs.parquet').to_pylist()
    evaluation = {
        'calibration': pq.read_table(calibration_data / 'pairs.parquet').to_pylist(),
        'early_stop': pq.read_table(early_stop_data / 'pairs.parquet').to_pylist(),
    }
    truth_rows = pq.read_table(sample / 'truth.parquet',
                               columns=['source1_entity_id', 'matched_entity_ids']).to_pylist()
    truth_all = {r['source1_entity_id']: set(r['matched_entity_ids'].split(','))
                 if r['matched_entity_ids'] else set() for r in truth_rows}
    truth = {role: {q: truth_all[q] for q in {r['source1_entity_id'] for r in rows}}
             for role, rows in evaluation.items()}
    old_threshold = json.loads(a_threshold_path.read_text())['threshold']
    old = {role: defaultdict(set) for role in evaluation}
    for batch in pq.ParquetFile(a_predictions).iter_batches(
            columns=['source1_entity_id', 'candidate_id', 'split', 'probability']):
        for q, t, role, probability in zip(*(batch.column(i).to_pylist() for i in range(4))):
            if role in old and q in truth[role] and probability >= old_threshold:
                old[role][q].add(t)
    base_scores = {role: [r['B_probability'] for r in rows]
                   for role, rows in evaluation.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    output.mkdir(parents=True)
    history = []
    best_score = float('-inf')
    for epoch in range(1, epochs + 1):
        epoch_rows = selected_train_rows(training, seed + epoch)
        model.train()
        losses = []
        for start in range(0, len(epoch_rows), 16):
            group = epoch_rows[start:start + 16]
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
            losses.append(float(loss.detach().cpu()))
            if len(losses) % 100 == 0:
                print(f'epoch={epoch} step={len(losses)} loss={sum(losses[-100:])/100:.4f}',
                      flush=True)
        result = evaluate_roles(model, tokenizer, evaluation, truth, old, base_scores)
        result.update({'epoch': epoch, 'train_rows': len(epoch_rows),
                       'mean_train_loss': sum(losses)/len(losses)})
        history.append(result)
        print(json.dumps(result), flush=True)
        if result['early_stop']['macro_f05'] > best_score:
            best_score = result['early_stop']['macro_f05']
            model.save_pretrained(output / 'model', safe_serialization=True)
            tokenizer.save_pretrained(output / 'model')
    report = {'status': 'PASS', 'scope': 'India rescue specialist; development only',
              'base_model': str(model_path), 'epochs': history,
              'best_epoch': max(history, key=lambda r: r['early_stop']['macro_f05'])['epoch'],
              'best_early_stop_macro_f05': best_score,
              'input_sha256': sha256(train_data / 'pairs.parquet'),
              'elapsed_seconds': round(time.time()-began, 2)}
    write_json(output / 'report.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'report.json') + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-train-b-scores-v1'))
    p.add_argument('--calibration', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-calibration-b-scores-v1'))
    p.add_argument('--early-stop', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-early-stop-b-scores-v1'))
    p.add_argument('--base-model', type=Path,
                   default=Path('artifacts/b_anushka/crossencoder-full-epoch-v2/model'))
    p.add_argument('--sample', type=Path,
                   default=Path('artifacts/a_tijil/full-train-sample-v1'))
    p.add_argument('--a-predictions', type=Path,
                   default=Path('artifacts/a_tijil/full-catboost-depth8-dev/validation_predictions.parquet'))
    p.add_argument('--a-threshold', type=Path,
                   default=Path('artifacts/a_tijil/full-catboost-depth8-dev/threshold.json'))
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/b_anushka/india-rescue-specialist-v1'))
    p.add_argument('--epochs', type=int, default=2)
    a = p.parse_args()
    train(a.train, a.calibration, a.early_stop, a.base_model, a.sample,
          a.a_predictions, a.a_threshold, a.output, a.epochs)
