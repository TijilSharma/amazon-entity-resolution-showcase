"""Test whether frozen BGE safely recovers specialist misses on held-out anchors."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.evaluation.entity_score import score


A_THRESHOLD = 0.6420290141254977
B_THRESHOLD = 0.99


def run(pilot):
    truth = {row['source1_entity_id']: row['matched_entity_ids'].split(',')
             if row['matched_entity_ids'] else []
             for row in pq.read_table('artifacts/a_tijil/full-train-sample-v1/truth.parquet').to_pylist()}
    a_validation = pq.read_table(
        'artifacts/a_tijil/full-catboost-depth8-dev/validation_predictions.parquet',
        columns=['source1_entity_id', 'candidate_id', 'split', 'probability']).to_pylist()
    a_holdout = pq.read_table('artifacts/a_tijil/depth8-holdout-rescue-v1.parquet').to_pylist()
    results = {}
    prepared = {}
    for role in ('calibration', 'early-stop', 'holdout'):
        rows = pq.read_table(pilot / f'{role}.parquet').to_pylist()
        specialist = pq.read_table(
            f'artifacts/b_anushka/india-rescue-{role}-specialist-v1/pairs.parquet',
            columns=['source1_entity_id', 'candidate_id', 'B_probability']).to_pylist()
        ids = {row['source1_entity_id'] for row in specialist}
        baseline = {q: [] for q in ids}
        if role == 'holdout':
            old = a_holdout
        else:
            old = [row for row in a_validation if row['split'] == role.replace('-', '_')]
        for row in old:
            q = row['source1_entity_id']
            if q in baseline and row['probability'] >= A_THRESHOLD:
                baseline[q].append(row['candidate_id'])
        for row in specialist:
            q = row['source1_entity_id']
            if q in baseline and row['B_probability'] >= B_THRESHOLD:
                baseline[q].append(row['candidate_id'])
        actual = {q: truth[q] for q in ids}
        base_score = score(actual, baseline)
        base_score.pop('per_anchor')
        prepared[role] = (actual, baseline, rows)
        results[role] = {'anchors': len(ids), 'baseline': base_score}

    def measure(role, threshold):
        actual, baseline, rows = prepared[role]
        predictions = {q: list(values) for q, values in baseline.items()}
        added = true = false = 0
        for row in rows:
            if row['B_probability'] >= B_THRESHOLD or row['bge_logit'] < threshold:
                continue
            q, target = row['source1_entity_id'], row['candidate_id']
            if target in predictions[q]:
                continue
            predictions[q].append(target)
            added += 1
            true += bool(row['label'])
            false += not bool(row['label'])
        metric = score(actual, predictions)
        metric.pop('per_anchor')
        metric.update({'added_links': added, 'added_true': true, 'added_false': false})
        return metric

    logits = np.array([r['bge_logit'] for r in prepared['calibration'][2]
                       if r['B_probability'] < B_THRESHOLD], dtype=float)
    thresholds = np.unique(np.r_[np.quantile(logits, np.linspace(0, 1, 201)),
                                  np.nextafter(np.max(logits), np.inf)])
    choices = [(measure('calibration', float(t))['macro_f05'], float(t))
               for t in thresholds]
    best_score, best_threshold = max(choices)
    for role in prepared:
        results[role]['with_bge'] = measure(role, best_threshold)
        results[role]['delta'] = (results[role]['with_bge']['macro_f05'] -
                                  results[role]['baseline']['macro_f05'])
    report = {'threshold_selected_on': 'calibration',
              'bge_logit_threshold': best_threshold,
              'calibration_best_macro_f05': best_score, 'roles': results}
    (pilot / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path,
                        default=Path('artifacts/b_anushka/bge-second-opinion-v1'))
    args = parser.parse_args()
    run(args.pilot)
