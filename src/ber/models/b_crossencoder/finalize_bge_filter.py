"""Remove India rescue links rejected by an independent multilingual reranker."""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def run(baseline, scores, output, threshold):
    if output.exists():
        raise FileExistsError(output)
    base = json.loads((baseline / 'manifest.json').read_text())
    report = json.loads((scores / 'manifest.json').read_text())
    if base['status'] != 'PASS' or report['status'] != 'PASS':
        raise ValueError('Input manifests must be PASS')
    rejected = defaultdict(set)
    total_scored = 0
    for name in ('zero', 'one_two', 'three_plus'):
        parts = sorted((scores / name).glob('part-*.parquet'))
        if len(parts) != 87:
            raise ValueError(f'Missing BGE score shards: {name}')
        for part in parts:
            for batch in pq.ParquetFile(part).iter_batches(
                    columns=['source1_entity_id', 'candidate_id', 'bge_logit']):
                for q, target, value in zip(*(batch.column(i).to_pylist()
                                              for i in range(3))):
                    total_scored += 1
                    if value < threshold:
                        rejected[q].add(target)
    if total_scored != sum(report['routes'].values()):
        raise ValueError('BGE score count differs from manifest')
    output.mkdir(parents=True)
    rows = changed = removed = remaining = 0
    with (baseline / 'matching_results.tsv').open(newline='') as old, \
            (output / 'matching_results.tsv').open('w', newline='') as new:
        reader = csv.reader(old, delimiter='\t')
        writer = csv.writer(new, delimiter='\t')
        if next(reader) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected matching TSV header')
        writer.writerow(('source1_entity_id', 'matched_entity_ids'))
        for q, joined in reader:
            prior = joined.split(',') if joined else []
            kept = [target for target in prior if target not in rejected.get(q, ())]
            changed += len(kept) != len(prior)
            removed += len(prior) - len(kept)
            remaining += len(kept)
            rows += 1
            writer.writerow((q, ','.join(kept)))
    if rows != base['anchors']:
        raise ValueError('Anchor count changed')
    os.link(baseline / 'candidate_pairs.tsv', output / 'candidate_pairs.tsv')
    result = {'status': 'PASS', 'version': 'bge-filter-india-v1',
              'anchors': rows, 'bge_logit_threshold': threshold,
              'scored_pairs': total_scored, 'rejected_pairs': sum(map(len, rejected.values())),
              'changed_anchors': changed, 'removed_links': removed,
              'remaining_links': remaining,
              'baseline_manifest_sha256': sha256(baseline / 'manifest.json'),
              'scores_manifest_sha256': sha256(scores / 'manifest.json'),
              'matching_sha256': sha256(output / 'matching_results.tsv'),
              'candidate_sha256': sha256(output / 'candidate_pairs.tsv')}
    write_json(output / 'manifest.json', result)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path,
                        default=Path('output/a_tijil/india-specialist-hybrid-gate-unique-v1'))
    parser.add_argument('--scores', type=Path,
                        default=Path('artifacts/b_anushka/bge-filter-selected-test-v1'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/a_tijil/india-bge-filter-v1'))
    parser.add_argument('--threshold', type=float, default=1.0)
    args = parser.parse_args()
    run(args.baseline, args.scores, args.output, args.threshold)
