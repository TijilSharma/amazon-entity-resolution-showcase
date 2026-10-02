"""Merge frozen Pipeline A predictions with scored India rescue candidates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def finalize(release, old, rescue, scores, output, threshold, extra_routes=(),
             topk_scores=None, topk_candidates=None, topk_threshold=0.7,
             topk_keep_threshold=None, topk_add_threshold=None,
             topk_india_threshold=None):
    if output.exists():
        raise FileExistsError(output)
    if (topk_keep_threshold is None) != (topk_add_threshold is None):
        raise ValueError('Both top-k gate thresholds are required together')
    if topk_keep_threshold is not None and topk_scores is None:
        raise ValueError('Top-k gate thresholds require top-k scores')
    old_manifest = json.loads((old / 'manifest.json').read_text())
    if old_manifest['status'] != 'PASS' or (old / '_SUCCESS').read_text().strip() != sha256(old / 'manifest.json'):
        raise ValueError('Pipeline A submission is incomplete')
    parts = sorted((release / 'test' / 'source1').glob('part-*.parquet'))
    routes = ((rescue, scores), *extra_routes)
    for part in parts:
        for route, probabilities in routes:
            if not (route / part.name).exists() or not (probabilities / part.name).exists():
                raise ValueError(f'Missing rescue or B scores: {part.name} in {route}')
            report = json.loads((probabilities / (part.stem + '.json')).read_text())
            if report['input_sha256'] != sha256(route / part.name):
                raise ValueError(f'Rescue score/input mismatch: {part.name} in {route}')
            route_rows = pq.ParquetFile(route / part.name).metadata.num_rows
            if report.get('score_mode') == 'sparse_prefilter':
                if report['candidate_pairs'] != route_rows or report['rows_with_probabilities'] != \
                        pq.ParquetFile(probabilities / part.name).metadata.num_rows:
                    raise ValueError(f'Sparse rescue probability count mismatch: {part.name} in {route}')
            elif report['rows_with_probabilities'] != route_rows:
                raise ValueError(f'Rescue probability count mismatch: {part.name} in {route}')
        if topk_scores is not None:
            if not (topk_scores / part.name).exists() or not (topk_candidates / part.name).exists():
                raise ValueError(f'Missing old top-k score/input: {part.name}')
            report = json.loads((topk_scores / (part.stem + '.json')).read_text())
            if report['input_sha256'] != sha256(topk_candidates / part.name):
                raise ValueError(f'Old top-k score/input mismatch: {part.name}')
            if report['rows'] != pq.ParquetFile(topk_scores / part.name).metadata.num_rows:
                raise ValueError(f'Old top-k score row count mismatch: {part.name}')
    output.mkdir(parents=True)
    rows = nonempty = added = total_candidates = 0
    with (old / 'matching_results.tsv').open(newline='') as af, \
            (old / 'candidate_pairs.tsv').open(newline='') as acf, \
            (output / 'matching_results.tsv').open('w', newline='') as mf, \
            (output / 'candidate_pairs.tsv').open('w', newline='') as cf:
        ar, acr = csv.reader(af, delimiter='\t'), csv.reader(acf, delimiter='\t')
        mw, cw = csv.writer(mf, delimiter='\t'), csv.writer(cf, delimiter='\t')
        if next(ar) != ['source1_entity_id', 'matched_entity_ids'] or \
                next(acr) != ['source1_entity_id', 'candidate_entity_ids']:
            raise ValueError('Pipeline A headers changed')
        mw.writerow(('source1_entity_id', 'matched_entity_ids'))
        cw.writerow(('source1_entity_id', 'candidate_entity_ids'))
        for part in parts:
            extras = defaultdict(list)
            for route, _ in routes:
                for batch in pq.ParquetFile(route / part.name).iter_batches(
                        columns=['source1_entity_id', 'candidate_id', 'rescue_rank']):
                    for q, t, rank in zip(*(batch.column(i).to_pylist() for i in range(3))):
                        extras[q].append((rank, t))
            picks = defaultdict(list)
            for _, probabilities in routes:
                for batch in pq.ParquetFile(probabilities / part.name).iter_batches():
                    for row in batch.to_pylist():
                        if row['probability'] >= threshold:
                            picks[row['source1_entity_id']].append((row['probability'], row['candidate_id']))
            topk_seen = defaultdict(set)
            topk_picks = defaultdict(list)
            topk_probabilities = defaultdict(list)
            if topk_scores is not None:
                for batch in pq.ParquetFile(topk_scores / part.name).iter_batches():
                    for row in batch.to_pylist():
                        q, tid = row['source1_entity_id'], row['candidate_id']
                        topk_seen[q].add(tid)
                        if topk_keep_threshold is not None or topk_india_threshold is not None:
                            topk_probabilities[q].append((row['candidate_rank'], tid,
                                                          row['probability']))
                        elif row['probability'] >= topk_threshold:
                            topk_picks[q].append((row['candidate_rank'], tid))
            for batch in pq.ParquetFile(part).iter_batches(columns=['entity_id', 'country'], batch_size=50_000):
                for q, country in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    prior = next(ar)
                    pool = next(acr)
                    if prior[0] != q or pool[0] != q:
                        raise ValueError(f'Pipeline A row order mismatch: {q}')
                    old_matches = prior[1].split(',') if prior[1] else []
                    old_pool = pool[1].split(',') if pool[1] else []
                    new_pool = [t for _, t in sorted(extras.get(q, ()))]
                    new_matches = [t for _, t in sorted(picks.get(q, ()), reverse=True)]
                    if len(new_pool) != len(set(new_pool)) or set(new_pool) & set(old_pool):
                        raise ValueError(f'Rescue duplicate candidate: {q}')
                    if not set(new_matches) <= set(new_pool):
                        raise ValueError(f'Rescue decision absent from candidate pool: {q}')
                    retained = [t for t in old_matches if t not in topk_seen[q]]
                    if topk_keep_threshold is not None:
                        old_set = set(old_matches)
                        topk_picks[q] = [
                            (rank, tid) for rank, tid, probability in topk_probabilities[q]
                            if probability >= (
                                topk_india_threshold if country == 'India' and
                                topk_india_threshold is not None else
                                (topk_keep_threshold if tid in old_set
                                 else topk_add_threshold))
                        ]
                    elif topk_india_threshold is not None:
                        chosen_threshold = (topk_india_threshold if country == 'India'
                                            else topk_threshold)
                        topk_picks[q] = [
                            (rank, tid) for rank, tid, probability in topk_probabilities[q]
                            if probability >= chosen_threshold
                        ]
                    topk_matches = [t for _, t in sorted(topk_picks[q])]
                    final_matches = retained + topk_matches + new_matches
                    final_pool = old_pool + new_pool
                    if len(final_matches) != len(set(final_matches)) or not set(final_matches) <= set(final_pool):
                        raise ValueError(f'Final decisions invalid for {q}')
                    mw.writerow((q, ','.join(final_matches)))
                    cw.writerow((q, ','.join(final_pool)))
                    rows += 1
                    nonempty += bool(final_matches)
                    added += len(new_matches)
                    total_candidates += len(final_pool)
            print(f'Wrote {part.name}: {rows:,} anchors', flush=True)
        if next(ar, None) is not None or next(acr, None) is not None:
            raise ValueError('Pipeline A has surplus rows')
    if rows != 1_732_544:
        raise ValueError(f'Expected 1,732,544 anchors, got {rows}')
    report = {'status': 'PASS', 'version': 'india-rescue-b-v1', 'anchors': rows,
              'routes': [str(route) for route, _ in routes],
              'old_topk_scores': str(topk_scores) if topk_scores else None,
              'old_topk_threshold': topk_threshold if topk_scores else None,
              'old_topk_india_threshold': topk_india_threshold,
              'old_topk_keep_threshold': topk_keep_threshold,
              'old_topk_add_threshold': topk_add_threshold,
              'rescue_threshold': threshold,
              'nonempty_anchors': nonempty, 'new_matches': added,
              'candidate_pairs': total_candidates,
              'matching_sha256': sha256(output / 'matching_results.tsv'),
              'candidate_sha256': sha256(output / 'candidate_pairs.tsv'),
              'baseline_manifest_sha256': sha256(old / 'manifest.json')}
    write_json(output / 'manifest.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, default=Path('data/processed/conservative-v1-py311-u14'))
    p.add_argument('--old', type=Path, default=Path('output/a_tijil/first-submission-v1'))
    p.add_argument('--rescue', type=Path, default=Path('candidates/india-rescue-v1/test'))
    p.add_argument('--scores', type=Path, default=Path('artifacts/b_anushka/india-rescue-test-probabilities-v2'))
    p.add_argument('--output', type=Path, default=Path('output/a_tijil/india-rescue-b-v1'))
    p.add_argument('--threshold', type=float, default=0.7218412160873413)
    p.add_argument('--extra-rescue', type=Path, action='append', default=[])
    p.add_argument('--extra-scores', type=Path, action='append', default=[])
    p.add_argument('--topk-scores', type=Path)
    p.add_argument('--topk-candidates', type=Path,
                   default=Path('candidates/full-union-v1/test'))
    p.add_argument('--topk-threshold', type=float, default=0.7)
    p.add_argument('--topk-india-threshold', type=float)
    p.add_argument('--topk-keep-threshold', type=float)
    p.add_argument('--topk-add-threshold', type=float)
    a = p.parse_args()
    if len(a.extra_rescue) != len(a.extra_scores):
        p.error('each --extra-rescue requires one --extra-scores')
    extra = tuple(zip(a.extra_rescue, a.extra_scores))
    print(json.dumps(finalize(a.release, a.old, a.rescue, a.scores, a.output,
                              a.threshold, extra, a.topk_scores,
                              a.topk_candidates, a.topk_threshold,
                              a.topk_keep_threshold, a.topk_add_threshold,
                              a.topk_india_threshold), indent=2))
