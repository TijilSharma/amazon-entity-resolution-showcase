"""Add high-confidence US rescue matches without changing existing decisions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import duckdb

from ber.data.build_normalized import sha256, write_json


def run(baseline, candidates_path, scores, database, output, threshold,
        min_name_similarity, min_address_similarity):
    if output.exists():
        raise FileExistsError(output)
    base_report = json.loads((baseline / 'manifest.json').read_text())
    score_report = json.loads((scores / 'manifest.json').read_text())
    if base_report['status'] != 'PASS' or score_report['status'] != 'PASS':
        raise ValueError('Input manifests must be PASS')
    con = duckdb.connect(str(database), read_only=True)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    candidates = con.execute(
        'SELECT s.source1_entity_id,s.candidate_id,s.probability '
        'FROM read_parquet(?) s '
        'JOIN read_parquet(?) r USING(source1_entity_id,candidate_id) '
        'JOIN targets t ON s.candidate_id=t.entity_id '
        'WHERE s.probability>=? AND t.country=\'US\' '
        'AND r.name_similarity>=? AND r.address_similarity>=?',
        [str(scores / 'part-*.parquet'), str(candidates_path / 'part-*.parquet'),
         threshold, min_name_similarity, min_address_similarity]).fetchall()
    con.close()
    proposed_by_target = defaultdict(list)
    for q, tid, probability in candidates:
        proposed_by_target[tid].append((float(probability), q))
    claimed = set()
    empty_queries = set()
    baseline_matching = baseline / 'matching_results.tsv'
    with baseline_matching.open(newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        if next(reader) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Bad baseline matching header')
        for q, joined in reader:
            if joined:
                claimed.update(joined.split(','))
            else:
                empty_queries.add(q)
    selected = defaultdict(list)
    prior_claims = 0
    nonempty_proposals = 0
    for tid, proposals in proposed_by_target.items():
        if tid in claimed:
            prior_claims += 1
            continue
        eligible = [(p, q) for p, q in proposals if q in empty_queries]
        nonempty_proposals += len(proposals) - len(eligible)
        if eligible:
            probability, q = max(eligible, key=lambda row: (row[0], row[1]))
            selected[q].append((probability, tid))
    del claimed
    output.mkdir(parents=True)
    rows = changed = links = 0
    with baseline_matching.open(newline='') as ms, \
            (baseline / 'candidate_pairs.tsv').open(newline='') as cs, \
            (output / 'matching_results.tsv').open('w', newline='') as md, \
            (output / 'candidate_pairs.tsv').open('w', newline='') as cd:
        mr, cr = csv.reader(ms, delimiter='\t'), csv.reader(cs, delimiter='\t')
        mw, cw = csv.writer(md, delimiter='\t'), csv.writer(cd, delimiter='\t')
        if next(mr) != ['source1_entity_id', 'matched_entity_ids'] or \
                next(cr) != ['source1_entity_id', 'candidate_entity_ids']:
            raise ValueError('Bad baseline TSV headers')
        mw.writerow(('source1_entity_id', 'matched_entity_ids'))
        cw.writerow(('source1_entity_id', 'candidate_entity_ids'))
        for q, prior in mr:
            pool_q, prior_pool = next(cr)
            if pool_q != q:
                raise ValueError('TSV rows are not aligned')
            new_ids = [tid for _, tid in sorted(selected.get(q, ()), reverse=True)]
            if new_ids and prior:
                raise ValueError('Tried to change a nonempty baseline query')
            if new_ids and (set(new_ids) & set(prior_pool.split(','))):
                raise ValueError('Rescue pair already in candidate pool')
            mw.writerow((q, ','.join(new_ids) if new_ids else prior))
            cw.writerow((q, prior_pool + (',' if prior_pool else '') + ','.join(new_ids)
                         if new_ids else prior_pool))
            rows += 1
            changed += bool(new_ids)
            links += len(new_ids)
        if next(cr, None) is not None:
            raise ValueError('Extra candidate rows')
    if rows != base_report['anchors']:
        raise ValueError('Anchor count changed')
    report = {'status': 'PASS', 'version': 'us-empty-rescue-v1',
              'anchors': rows, 'threshold': threshold, 'changed_anchors': changed,
              'min_name_similarity': min_name_similarity,
              'min_address_similarity': min_address_similarity,
              'added_links': links, 'high_score_proposals': len(candidates),
              'already_claimed_targets': prior_claims,
              'nonempty_query_proposals': nonempty_proposals,
              'baseline_manifest_sha256': sha256(baseline / 'manifest.json'),
              'score_manifest_sha256': sha256(scores / 'manifest.json'),
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
                        default=Path('artifacts/b_anushka/us-empty-rescue-b-v1'))
    parser.add_argument('--candidates', type=Path,
                        default=Path('candidates/us-empty-rescue-v1/test'))
    parser.add_argument('--database', type=Path,
                        default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/a_tijil/india-hybrid-unique-us-empty-rescue-v1'))
    parser.add_argument('--threshold', type=float, default=0.999)
    parser.add_argument('--min-name-similarity', type=float, default=60.0)
    parser.add_argument('--min-address-similarity', type=float, default=60.0)
    args = parser.parse_args()
    run(args.baseline, args.candidates, args.scores, args.database, args.output,
        args.threshold, args.min_name_similarity, args.min_address_similarity)
