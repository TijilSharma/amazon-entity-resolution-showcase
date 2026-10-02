"""Audit source TSVs and training links using bounded-memory SQLite storage.

This script only reads the dataset. Its temporary database is deleted on exit.
Run with --data-root /path/to/dataset --report /path/to/report.json.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from validate_inputs import inspect_source


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def batches(rows, size=20000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def audit(data_root, report_path):
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'sources': {}, 'ground_truth': {}, 'checks': {}}
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    with tempfile.TemporaryDirectory(prefix='amazon_er_audit_') as temp:
        db = sqlite3.connect(str(Path(temp) / 'audit.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-65536')
            db.execute('PRAGMA temp_store=FILE')
            # Disposable audit DB; the source data is never edited.
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('CREATE TABLE records(split TEXT, source INTEGER, id TEXT)')
            db.execute('CREATE TABLE truth(anchor TEXT)')
            db.execute('CREATE TABLE links(anchor TEXT, target TEXT)')
            for split in ['train', 'test']:
                for source in [1, 2, 3]:
                    path = data_root / split / f'{split}_source{source}.tsv'
                    summary = inspect_source(path, f'S{source}-')
                    summary['sha256'] = sha256(path)
                    report['sources'][path.name] = summary
                    if summary['malformed_records']:
                        raise ValueError(f'Malformed source records: {path}')
                    with path.open(encoding='utf-8', newline='') as stream:
                        rows = ((split, source, row['entity_id']) for row in
                                csv.DictReader(stream, delimiter='\t', strict=True))
                        for batch in batches(rows):
                            db.executemany('INSERT INTO records VALUES (?, ?, ?)', batch)
                    db.commit()
                    save()
                    print(f'Inspected {path.name}: {summary["total_records"]:,} records', flush=True)
            print('Indexing source IDs on disk', flush=True)
            db.execute('CREATE INDEX record_key ON records(split, id)')
            db.commit()
            gt = data_root / 'train/train_ground_truth.tsv'
            g = {'rows': 0, 'positive_links': 0, 'singleton_rows': 0,
                 'duplicate_ids_within_lists': 0, 'invalid_match_prefixes': 0,
                 'ids_with_surrounding_whitespace': 0, 'sha256': sha256(gt)}
            anchors, links = [], []
            with gt.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream, delimiter='\t', strict=True)
                if reader.fieldnames != ['source1_entity_id', 'matched_entity_ids']:
                    raise ValueError(f'Unexpected ground-truth header: {reader.fieldnames}')
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f'Malformed ground truth near line {reader.line_num}')
                    anchor = row['source1_entity_id']
                    ids = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
                    g['rows'] += 1
                    g['positive_links'] += len(ids)
                    g['singleton_rows'] += not ids
                    g['duplicate_ids_within_lists'] += len(ids) - len(set(ids))
                    g['invalid_match_prefixes'] += sum(not i.startswith(('S2-', 'S3-')) for i in ids)
                    g['ids_with_surrounding_whitespace'] += sum(i != i.strip() for i in [anchor] + ids)
                    anchors.append((anchor,))
                    links.extend((anchor, target) for target in ids)
                    if len(anchors) >= 20000:
                        db.executemany('INSERT INTO truth VALUES (?)', anchors)
                        db.executemany('INSERT INTO links VALUES (?, ?)', links)
                        anchors.clear(); links.clear()
                db.executemany('INSERT INTO truth VALUES (?)', anchors)
                db.executemany('INSERT INTO links VALUES (?, ?)', links)
            report['ground_truth'] = g
            db.commit()
            print('Ground truth loaded; checking coverage and references', flush=True)
            db.execute('CREATE INDEX truth_key ON truth(anchor)')
            db.execute('CREATE INDEX link_target ON links(target)')
            queries = {
                'duplicate_source_id_groups': 'SELECT COUNT(*) FROM (SELECT split,id FROM records GROUP BY split,id HAVING COUNT(*)>1)',
                'duplicate_ground_truth_anchor_groups': 'SELECT COUNT(*) FROM (SELECT anchor FROM truth GROUP BY anchor HAVING COUNT(*)>1)',
                'missing_ground_truth_for_train_s1': "SELECT COUNT(*) FROM records r WHERE split='train' AND source=1 AND NOT EXISTS (SELECT 1 FROM truth t WHERE t.anchor=r.id)",
                'ground_truth_anchors_not_in_train_s1': "SELECT COUNT(*) FROM truth t WHERE NOT EXISTS (SELECT 1 FROM records r WHERE r.split='train' AND r.source=1 AND r.id=t.anchor)",
                'links_not_in_train_s2_s3': "SELECT COUNT(*) FROM links l WHERE NOT EXISTS (SELECT 1 FROM records r WHERE r.split='train' AND r.source IN (2,3) AND r.id=l.target)",
                'targets_linked_to_multiple_s1': 'SELECT COUNT(*) FROM (SELECT target FROM links GROUP BY target HAVING COUNT(DISTINCT anchor)>1)',
                'train_test_shared_record_ids': "SELECT COUNT(*) FROM records a WHERE a.split='train' AND EXISTS (SELECT 1 FROM records b WHERE b.split='test' AND b.id=a.id)",
            }
            for name, query in queries.items():
                report['checks'][name] = db.execute(query).fetchone()[0]
                print(f'{name}: {report["checks"][name]}', flush=True)
                save()
            required = ['duplicate_source_id_groups', 'duplicate_ground_truth_anchor_groups',
                        'missing_ground_truth_for_train_s1', 'ground_truth_anchors_not_in_train_s1',
                        'links_not_in_train_s2_s3']
            errors = sum(report['checks'][k] for k in required)
            errors += sum(g[k] for k in ['duplicate_ids_within_lists', 'invalid_match_prefixes',
                                         'ids_with_surrounding_whitespace'])
            errors += sum(s['invalid_id_prefixes'] for s in report['sources'].values())
            report['structural_status'] = 'PASS' if not errors else 'FAIL'
            report['notes'] = [
                'Empty names and addresses are reported, not rejected; literal null remains text.',
                'Shared targets and train/test ID overlap are diagnostics, not automatically structural failures.',
                'This audit does not detect duplicate business content or establish entity-level train/test independence.',
            ]
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            save()
        finally:
            db.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.data_root, args.report)
    print(result['structural_status'], flush=True)
    raise SystemExit(0 if result['structural_status'] == 'PASS' else 1)
