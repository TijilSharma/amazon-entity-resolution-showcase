"""Train/evaluate a rescue-specific CatBoost using frozen B and lexical evidence."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

from ber.data.build_normalized import sha256, write_json
from ber.evaluation.entity_score import score
from ber.features.comparisons import compare_field


def feature_values(row):
    values = {
        'B_probability': row['B_probability'],
        'rescue_rank': row['rescue_rank'],
        'name_similarity': row['name_similarity'] / 100,
        'address_similarity': row['address_similarity'] / 100,
        'idf_evidence': row['idf_evidence'],
    }
    for prefix in ('name', 'address'):
        values.update(compare_field(row['query_' + prefix], row['target_' + prefix], prefix))
    return values


def matrix(rows, names=None):
    features = [feature_values(row) for row in rows]
    if names is None:
        names = list(features[0])
    return np.asarray([[float(row[name]) for name in names] for row in features],
                      dtype=np.float32), names


def evaluate(model, names, role, data, sample, predictions, splits_path):
    rows = pq.read_table(data / 'pairs.parquet').to_pylist()
    x, _ = matrix(rows, names)
    probs = model.predict_proba(x)[:, 1]
    splits = json.loads(splits_path.read_text())
    anchors = {row['source1_entity_id'] for row in rows}
    truth = {r['source1_entity_id']: r['matched_entity_ids'].split(',')
             if r['matched_entity_ids'] else []
             for r in pq.read_table(sample / 'truth.parquet').to_pylist()
             if r['source1_entity_id'] in anchors}
    old_threshold = json.loads(Path('artifacts/a_tijil/full-catboost-depth8-dev/threshold.json').read_text())['threshold']
    old = defaultdict(set)
    for batch in pq.ParquetFile(predictions).iter_batches(
            columns=['source1_entity_id', 'candidate_id', 'probability']):
        for q, t, p in zip(*(batch.column(i).to_pylist() for i in range(3))):
            if q in anchors and p >= old_threshold:
                old[q].add(t)
    base = score(truth, {q: list(old[q]) for q in truth})
    trials = {}
    for name, values in (('B', [row['B_probability'] for row in rows]),
                         ('stacker', probs)):
        results = []
        for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99):
            choices = {q:set(old[q]) for q in truth}
            for row, p in zip(rows, values):
                if p >= threshold:
                    choices[row['source1_entity_id']].add(row['candidate_id'])
            m = score(truth, {q:list(v) for q,v in choices.items()})
            results.append({'threshold':threshold,'macro_f05':m['macro_f05'],
                            'tp':m['tp'],'fp':m['fp'],'fn':m['fn']})
        trials[name] = results
    return {'role':role,'anchors':len(truth),'baseline_A':base['macro_f05'],
            'trials':trials}


def train(train_data, calibration, early_stop, sample, predictions, splits_path, output):
    if output.exists():
        raise FileExistsError(output)
    rows = pq.read_table(train_data / 'pairs.parquet').to_pylist()
    x, names = matrix(rows)
    labels = np.asarray([r['label'] for r in rows], dtype=np.int8)
    model = CatBoostClassifier(iterations=800, depth=6, learning_rate=.05,
                               l2_leaf_reg=5, loss_function='Logloss',
                               scale_pos_weight=10, thread_count=4,
                               random_seed=2026, verbose=False)
    model.fit(x, labels)
    output.mkdir(parents=True)
    model.save_model(str(output / 'model.cbm'))
    write_json(output / 'feature_names.json', names)
    calibration_result = evaluate(model, names, 'calibration', calibration,
                                  sample, predictions, splits_path)
    best = max(calibration_result['trials']['stacker'],
               key=lambda r:(r['macro_f05'],r['threshold']))
    early_result = evaluate(model, names, 'early_stop', early_stop,
                            sample, predictions, splits_path)
    report = {'status':'PASS','train_rows':len(rows),'train_positives':int(labels.sum()),
              'selected_threshold':best['threshold'],'calibration':calibration_result,
              'early_stop':early_result,
              'train_input_sha256':sha256(train_data / 'pairs.parquet')}
    write_json(output / 'report.json',report)
    (output / '_SUCCESS').write_text(sha256(output / 'report.json')+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,default=Path('artifacts/b_anushka/india-rescue-train-b-scores-v1'))
    p.add_argument('--calibration',type=Path,default=Path('artifacts/b_anushka/india-rescue-calibration-b-scores-v1'))
    p.add_argument('--early-stop',type=Path,default=Path('artifacts/b_anushka/india-rescue-early-stop-b-scores-v1'))
    p.add_argument('--sample',type=Path,default=Path('artifacts/a_tijil/full-train-sample-v1'))
    p.add_argument('--predictions',type=Path,default=Path('artifacts/a_tijil/full-catboost-depth8-dev/validation_predictions.parquet'))
    p.add_argument('--splits',type=Path,default=Path('artifacts/a_tijil/full-catboost-depth8-dev/splits.json'))
    p.add_argument('--output',type=Path,default=Path('artifacts/b_anushka/india-rescue-stacker-v1'))
    a=p.parse_args()
    train(a.train,a.calibration,a.early_stop,a.sample,a.predictions,a.splits,a.output)
