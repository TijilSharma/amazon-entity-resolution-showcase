"""One conservative France top-10 addition per sparse matched entity.

This changes only France records with one or two current matches. New target
IDs must be unclaimed globally and already present in the candidate pool.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def run(baseline, scores, release, output, lower, upper):
    if output.exists():
        raise FileExistsError(output)
    base = json.loads((baseline / 'manifest.json').read_text())
    if base['status'] != 'PASS':
        raise ValueError('Incomplete baseline')
    france = set()
    for part in sorted((release / 'test' / 'source1').glob('part-*.parquet')):
        for batch in pq.ParquetFile(part).iter_batches(columns=['entity_id', 'country']):
            france.update(q for q, country in zip(batch.column(0).to_pylist(),
                                                  batch.column(1).to_pylist())
                          if country == 'France')
    claimed = set()
    sparse = {}
    with (baseline / 'matching_results.tsv').open(newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        if next(reader) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected baseline header')
        for q, joined in reader:
            matches = joined.split(',') if joined else []
            claimed.update(matches)
            if q in france and 1 <= len(matches) <= 2:
                sparse[q] = set(matches)
    proposals = defaultdict(list)
    for part in sorted(scores.glob('part-*.parquet')):
        for batch in pq.ParquetFile(part).iter_batches(
                columns=['source1_entity_id', 'candidate_id', 'probability', 'candidate_rank']):
            for q, target, probability, rank in zip(*(batch.column(i).to_pylist()
                                                       for i in range(4))):
                if q in sparse and lower <= probability < upper and target not in claimed:
                    proposals[q].append((probability, -rank, target))
    del claimed
    by_target = defaultdict(list)
    for q, options in proposals.items():
        probability, neg_rank, target = max(options)
        by_target[target].append((probability, q))
    additions = {}
    for target, owners in by_target.items():
        probability, q = max(owners)
        additions[q] = target
    output.mkdir(parents=True)
    rows = 0
    with (baseline / 'matching_results.tsv').open(newline='') as old, \
            (output / 'matching_results.tsv').open('w', newline='') as new:
        reader = csv.reader(old, delimiter='\t')
        writer = csv.writer(new, delimiter='\t')
        if next(reader) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected baseline header')
        writer.writerow(('source1_entity_id', 'matched_entity_ids'))
        for q, joined in reader:
            target = additions.get(q)
            writer.writerow((q, joined + ',' + target if target else joined))
            rows += 1
    if rows != base['anchors']:
        raise ValueError('Anchor count changed')
    os.link(baseline / 'candidate_pairs.tsv', output / 'candidate_pairs.tsv')
    report = {'status': 'PASS', 'version': 'france-gate-probe-v1',
              'anchors': rows, 'lower': lower, 'upper': upper,
              'eligible_queries': len(sparse), 'proposed_queries': len(proposals),
              'added_links': len(additions),
              'baseline_manifest_sha256': sha256(baseline / 'manifest.json'),
              'matching_sha256': sha256(output / 'matching_results.tsv'),
              'candidate_sha256': sha256(output / 'candidate_pairs.tsv')}
    write_json(output / 'manifest.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path,
                        default=Path('output/a_tijil/india-specialist-hybrid-gate-unique-v1'))
    parser.add_argument('--scores', type=Path,
                        default=Path('artifacts/b_anushka/old-top10-hybrid-specialist-v1'))
    parser.add_argument('--release', type=Path,
                        default=Path('data/processed/conservative-v1-py311-u14'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/a_tijil/france-gate-probe-v1'))
    parser.add_argument('--lower', type=float, default=0.7)
    parser.add_argument('--upper', type=float, default=0.8)
    args = parser.parse_args()
    run(args.baseline, args.scores, args.release, args.output, args.lower, args.upper)
