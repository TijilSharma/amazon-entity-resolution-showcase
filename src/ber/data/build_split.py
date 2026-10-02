"""Freeze business-group splits; no learned models or leaderboard evaluation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import unicodedata
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build_normalized import sha256, write_json


def tokens(text):
    # Preserve non-Latin combining marks. Only punctuation/whitespace separates.
    text = ''.join(' ' if c.isspace() or unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(sorted(text.split()))


def signature(country, name, address):
    if not country or not name or not address:
        return None
    fields = [country.casefold(), tokens(name), tokens(address)]
    if not fields[1] or not fields[2]:
        return None
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).digest()


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return x


def union(parent, a, b):
    a, b = find(parent, a), find(parent, b)
    if a != b:
        parent[max(a, b)] = min(a, b)


def choose_split(anchor, seed):
    bucket = int.from_bytes(hashlib.sha256(f'{seed}:{anchor}'.encode()).digest()[:8], 'big') % 10000
    return 'train' if bucket < 8000 else 'validation' if bucket < 9000 else 'holdout'


def build(release, labels, pilot, output, seed=20260925):
    m = json.loads((release / 'manifest.json').read_text())
    if m['status'] != 'PASS' or sha256(release / 'manifest.json') != (release / '_SUCCESS').read_text().strip():
        raise ValueError('Invalid release')
    if sha256(labels) != m['ground_truth']['sha256']:
        raise ValueError('Wrong labels')
    pilot_meta = json.loads((pilot / 'manifest.json').read_text())
    if sha256(pilot / 'truth.json') != pilot_meta['files']['truth.json']:
        raise ValueError('Pilot truth differs')
    used_queries = set(json.loads((pilot / 'truth.json').read_text()))
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    db = sqlite3.connect(output / 'build.sqlite')
    db.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-131072;
        CREATE TABLE anchors (idx INTEGER PRIMARY KEY, id TEXT UNIQUE, matches TEXT, country TEXT);
        CREATE TABLE owners (id TEXT PRIMARY KEY, anchor INTEGER) WITHOUT ROWID;
        CREATE TABLE signatures (sig BLOB, anchor INTEGER);
        CREATE TABLE batch (id TEXT, sig BLOB, country TEXT);
    """)
    info = {'status': 'BUILDING', 'seed': seed, 'release_manifest_sha256': sha256(release / 'manifest.json'),
            'labels_sha256': sha256(labels), 'pilot_manifest_sha256': sha256(pilot / 'manifest.json'),
            'builder_sha256': sha256(Path(__file__)), 'policy': '80/10/10 hash of connected business group; pilot groups forced train',
            'signature': 'country + Latin-accent-folded name/address, punctuation separated and tokens sorted; both fields required',
            'limitations': ['Signatures catch exact and specified formatting-equivalent content, not all fuzzy business duplicates.',
                            'Missing-address and spelling/transliteration variants across different anchor labels may remain.',
                            'Official test is untouched. This split provides no full-corpus retrieval or F0.5 result.']}
    write_json(output / 'manifest.json', info)
    try:
        anchors, owner_batch, anchor_batch = [], [], []
        with labels.open(encoding='utf-8', newline='') as f:
            for idx, row in enumerate(csv.DictReader(f, delimiter='\t')):
                anchor, values = row['source1_entity_id'], row['matched_entity_ids']
                anchors.append(anchor)
                anchor_batch.append((idx, anchor, values, ''))
                owner_batch.append((anchor, idx))
                owner_batch.extend((t, idx) for t in values.split(',') if t)
                if len(anchor_batch) >= 10000:
                    db.executemany('INSERT INTO anchors VALUES (?,?,?,?)', anchor_batch)
                    db.executemany('INSERT INTO owners VALUES (?,?)', owner_batch)
                    anchor_batch.clear(); owner_batch.clear()
            db.executemany('INSERT INTO anchors VALUES (?,?,?,?)', anchor_batch)
            db.executemany('INSERT INTO owners VALUES (?,?)', owner_batch)
        db.commit()
        parent = np.arange(len(anchors), dtype=np.int32)
        print(f'Ownership loaded: {len(anchors):,} anchors', flush=True)
        scanned = Counter()
        for source in (1, 2, 3):
            key = f'train_source{source}.tsv'
            expected = {s['file']: s for s in m['sources'][key]['shards']}
            paths = sorted((release / 'train' / f'source{source}').glob('*.parquet'))
            if {p.name for p in paths} != set(expected):
                raise ValueError('Unexpected source shards')
            for number, path in enumerate(paths):
                if sha256(path) != expected[path.name]['sha256']:
                    raise ValueError(f'Changed shard: {path}')
                rows = pq.read_table(path, columns=['entity_id', 'country', 'name_accent_folded',
                                                   'address_accent_folded']).to_pylist()
                batch = [(r['entity_id'], signature(r['country'], r['name_accent_folded'],
                                                   r['address_accent_folded']), r['country']) for r in rows]
                db.execute('DELETE FROM batch')
                db.executemany('INSERT INTO batch VALUES (?,?,?)', batch)
                db.execute('INSERT INTO signatures SELECT b.sig,o.anchor FROM batch b JOIN owners o ON b.id=o.id WHERE b.sig IS NOT NULL')
                if source == 1:
                    db.executemany('UPDATE anchors SET country=? WHERE id=?', [(r['country'], r['entity_id']) for r in rows])
                else:
                    scanned['linked_targets_with_different_country'] += db.execute(
                        'SELECT COUNT(*) FROM batch b JOIN owners o ON b.id=o.id JOIN anchors a ON o.anchor=a.idx WHERE b.country<>a.country').fetchone()[0]
                scanned['linked_records_without_signature'] += db.execute(
                    'SELECT COUNT(*) FROM batch b JOIN owners o ON b.id=o.id WHERE b.sig IS NULL').fetchone()[0]
                scanned[key] += len(rows)
                db.commit()
                if (number+1) % 50 == 0:
                    print(f'{key}: {scanned[key]:,} fingerprinted', flush=True)
            if scanned[key] != m['sources'][key]['rows']:
                raise ValueError('Wrong source count')
        print('Building duplicate-signature groups', flush=True)
        db.execute('CREATE INDEX sig_key ON signatures(sig,anchor)')
        previous, first = None, None
        duplicate_signatures = 0
        for sig, anchor in db.execute('SELECT sig,anchor FROM signatures GROUP BY sig,anchor ORDER BY sig,anchor'):
            if sig == previous:
                union(parent, first, anchor)
                duplicate_signatures += 1
            else:
                previous, first = sig, anchor
        roots = np.array([find(parent, i) for i in range(len(anchors))], dtype=np.int32)
        forced = {int(roots[i]) for i, a in enumerate(anchors) if a in used_queries}
        if len(used_queries & set(anchors)) != len(used_queries):
            raise ValueError('Pilot anchors absent')
        codes = {}
        for root in np.unique(roots):
            codes[int(root)] = 'train' if int(root) in forced else choose_split(anchors[root], seed)
        db.execute('CREATE TABLE assignment (anchor INTEGER PRIMARY KEY, group_id TEXT, split TEXT)')
        db.executemany('INSERT INTO assignment VALUES (?,?,?)',
                       ((i, anchors[int(root)], codes[int(root)]) for i, root in enumerate(roots)))
        db.commit()
        print('Assignments created; verifying cross-partition leakage', flush=True)
        collisions = db.execute("""SELECT COUNT(*) FROM (
            SELECT s.sig FROM signatures s JOIN assignment a ON a.anchor=s.anchor
            GROUP BY s.sig HAVING COUNT(DISTINCT a.split)>1)""").fetchone()[0]
        if collisions:
            raise ValueError(f'{collisions} content signatures cross partitions')
        # UNIQUE ownership + one assignment per anchor guarantees linked targets stay together.
        schema = pa.schema([('source1_entity_id', pa.string()), ('group_id', pa.string()),
                            ('split', pa.string()), ('country', pa.string()), ('positive_count', pa.int32())])
        writer = pq.ParquetWriter(output / 'anchor_membership.parquet', schema, compression='zstd')
        counts, countries, positives, singletons = Counter(), Counter(), Counter(), Counter()
        files = {s: (output / f'{s}_ground_truth.tsv').open('w', newline='', encoding='utf-8')
                 for s in ('train', 'validation', 'holdout')}
        writers = {s: csv.writer(f, delimiter='\t') for s, f in files.items()}
        for w in writers.values(): w.writerow(['source1_entity_id', 'matched_entity_ids'])
        cursor = db.execute('SELECT a.id,x.group_id,x.split,a.country,a.matches FROM anchors a JOIN assignment x ON a.idx=x.anchor ORDER BY a.idx')
        while records := cursor.fetchmany(20000):
            out = []
            for anchor, group, split, country, matches in records:
                n = len(matches.split(',')) if matches else 0
                out.append(dict(source1_entity_id=anchor, group_id=group, split=split, country=country, positive_count=n))
                writers[split].writerow([anchor, matches])
                counts[split] += 1; countries[f'{split}/{country}'] += 1
                positives[split] += n; singletons[split] += n == 0
            writer.write_table(pa.Table.from_pylist(out, schema=schema))
        writer.close()
        for f in files.values(): f.close()
        schema = pa.schema([('entity_id', pa.string()), ('source1_entity_id', pa.string()),
                            ('group_id', pa.string()), ('split', pa.string())])
        writer = pq.ParquetWriter(output / 'record_membership.parquet', schema, compression='zstd')
        cursor = db.execute('SELECT o.id,a.id,x.group_id,x.split FROM owners o JOIN anchors a ON o.anchor=a.idx JOIN assignment x ON o.anchor=x.anchor')
        n_owned = 0
        while records := cursor.fetchmany(20000):
            writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, r)) for r in records], schema=schema))
            n_owned += len(records)
        writer.close()
        _, sizes = np.unique(roots, return_counts=True)
        info.update(status='PASS', counts=dict(counts), country_counts=dict(countries),
                    positive_links=dict(positives), singleton_counts=dict(singletons),
                    total_anchors=len(anchors), owned_records=n_owned,
                    connected_groups=len(sizes), largest_group_anchors=int(sizes.max()),
                    groups_with_multiple_anchors=int(np.sum(sizes>1)),
                    pilot_queries_forced_train=len(used_queries), forced_train_groups=len(forced),
                    forced_train_anchors=sum(int(r) in forced for r in roots),
                    duplicate_signature_owner_edges=duplicate_signatures,
                    cross_partition_signature_collisions=collisions, scan_counts=dict(scanned))
        if sum(counts.values()) != len(anchors):
            raise ValueError('Incomplete assignments')
        db.close()
        (output / 'build.sqlite').unlink()
        info['elapsed_seconds'] = round(time.time()-start, 2)
        info['files'] = {p.name: sha256(p) for p in sorted(output.iterdir())
                         if p.suffix in ('.parquet', '.tsv')}
        write_json(output / 'manifest.json', info)
        (output / '_SUCCESS').write_text(sha256(output / 'manifest.json')+'\n')
        print(json.dumps(info, indent=2), flush=True)
        return info
    except BaseException as e:
        info.update(status='FAILED', error=repr(e))
        write_json(output / 'manifest.json', info)
        db.close()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release','labels','pilot','output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--seed', type=int, default=20260925)
    a=p.parse_args()
    build(a.release,a.labels,a.pilot,a.output,a.seed)
