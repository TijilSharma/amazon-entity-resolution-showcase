"""Give each Source-2/3 target to at most one Source-1 entity."""
import argparse
import csv
import json
import os
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def resolve(baseline, scores, output):
    if output.exists():
        raise FileExistsError(output)
    baseline_report = json.loads((baseline / 'manifest.json').read_text())
    score_report = json.loads((scores / 'manifest.json').read_text())
    if baseline_report['status'] != 'PASS' or score_report['status'] != 'PASS':
        raise ValueError('Incomplete baseline or conflict scores')
    parts = sorted(scores.glob('part-*.parquet'))
    if len(parts) != score_report['parts'] or sum(pq.ParquetFile(p).metadata.num_rows for p in parts) != score_report['rows']:
        raise ValueError('Conflict score shard mismatch')
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=4')
    score_glob = str(scores / 'part-*.parquet')
    winner_rows = con.execute('''SELECT candidate_id,source1_entity_id,probability FROM (
        SELECT *,row_number() OVER (PARTITION BY candidate_id
                                    ORDER BY probability DESC,source1_entity_id) AS n
        FROM read_parquet(?)
    ) WHERE n=1''', [score_glob]).fetchall()
    winners = {target: owner for target, owner, _ in winner_rows}
    if len(winners) != len(winner_rows):
        raise ValueError('Duplicate winner IDs')
    con.close()
    output.mkdir(parents=True)
    old_file = baseline / 'matching_results.tsv'
    new_file = output / 'matching_results.tsv'
    rows = changed = removed = total = 0
    with old_file.open(newline='') as old, new_file.open('w', newline='') as new:
        reader = csv.reader(old, delimiter='\t')
        writer = csv.writer(new, delimiter='\t')
        header = next(reader)
        if header != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected matching file header')
        writer.writerow(header)
        for q, joined in reader:
            ids = joined.split(',') if joined else []
            kept = [t for t in ids if winners.get(t, q) == q]
            changed += len(kept) != len(ids)
            removed += len(ids) - len(kept)
            total += len(kept)
            rows += 1
            writer.writerow((q, ','.join(kept)))
    if rows != baseline_report['anchors'] or removed != score_report['rows'] - len(winners):
        raise ValueError(f'Unexpected conflict removal count: {rows}, {removed}')
    os.link(baseline / 'candidate_pairs.tsv', output / 'candidate_pairs.tsv')
    report = {'status': 'PASS', 'version': 'unique-target-b-v1',
              'anchors': rows, 'remaining_matches': total,
              'conflicting_targets': len(winners),
              'removed_links': removed, 'changed_anchors': changed,
              'baseline_manifest_sha256': sha256(baseline / 'manifest.json'),
              'score_manifest_sha256': sha256(scores / 'manifest.json'),
              'matching_sha256': sha256(new_file),
              'candidate_sha256': sha256(output / 'candidate_pairs.tsv')}
    write_json(output / 'manifest.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path,
                   default=Path('output/a_tijil/india-rescue-specialist-top10-3plus-v1'))
    p.add_argument('--scores', type=Path,
                   default=Path('artifacts/b_anushka/conflict-b-scores-v1'))
    p.add_argument('--output', type=Path,
                   default=Path('output/a_tijil/india-specialist-unique-b-v1'))
    a = p.parse_args()
    print(json.dumps(resolve(a.baseline, a.scores, a.output), indent=2))
