"""Prepare a deterministic sampled-corpus diagnostic, NOT a final validation split."""
import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ber.data.build_normalized import sha256, write_json

COLUMNS = ['entity_id', 'country', 'name_normalized', 'address_normalized',
           'name_scripts', 'address_scripts', 'name_missing', 'address_missing']


def select_truth(path, n, seed):
    def entries():
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                key = hashlib.sha256(f"{seed}:{row['source1_entity_id']}".encode()).digest()
                yield key, row['source1_entity_id'], row['matched_entity_ids']
    selected = heapq.nsmallest(n, entries())
    return {anchor: value.split(',') if value else [] for _, anchor, value in selected}


def read_selected(root, split, source, wanted, positions=()):
    found = []
    offset = 0
    wanted_array = pa.array(sorted(wanted), type=pa.string())
    positions = np.asarray(positions, dtype=np.int64)
    for path in sorted((root / split / source).glob('*.parquet')):
        table = pq.read_table(path, columns=COLUMNS)
        mask = pc.is_in(table['entity_id'], value_set=wanted_array).to_numpy()
        chosen = positions[(positions >= offset) & (positions < offset + len(table))] - offset
        mask = mask.copy()
        mask[chosen] = True
        if mask.any():
            found.extend(table.filter(pa.array(mask)).to_pylist())
        offset += len(table)
    return found


def prepare(release, labels, output, queries=500, distractors=30000, fit_rows=10000, seed=20260925):
    m = json.loads((release / 'manifest.json').read_text())
    if m['status'] != 'PASS' or (release / '_SUCCESS').read_text().strip() != sha256(release / 'manifest.json'):
        raise ValueError('Incomplete release')
    if sha256(labels) != m['ground_truth']['sha256']:
        raise ValueError('Wrong ground truth')
    if min(queries, distractors, fit_rows) < 1:
        raise ValueError('Sample counts must be positive')
    output.mkdir(parents=True, exist_ok=False)
    truth = select_truth(labels, queries, seed)
    positives = {target for values in truth.values() for target in values}
    anchors = read_selected(release, 'train', 'source1', set(truth))
    assert len(anchors) == len(truth)
    rng = np.random.default_rng(seed)
    counts = [m['sources'][f'train_source{s}.tsv']['rows'] for s in (2, 3)]
    if distractors + fit_rows > sum(counts):
        raise ValueError('Sample exceeds target population')
    sampled = rng.choice(sum(counts), size=distractors + fit_rows, replace=False)
    # The first positions are corpus distractors; remaining positions are IDF fit data.
    corpus_positions = np.sort(sampled[:distractors])
    fit_positions = np.sort(sampled[distractors:])
    corpus, fit = [], []
    offset = 0
    for s, count in zip((2, 3), counts):
        cp = corpus_positions[(corpus_positions >= offset) & (corpus_positions < offset + count)] - offset
        fp = fit_positions[(fit_positions >= offset) & (fit_positions < offset + count)] - offset
        corpus.extend(read_selected(release, 'train', f'source{s}', positives, cp))
        fit.extend(read_selected(release, 'train', f'source{s}', set(), fp))
        offset += count
    fit = [r for r in fit if r['entity_id'] not in positives]
    assert positives <= {r['entity_id'] for r in corpus}
    assert not ({r['entity_id'] for r in fit} & {r['entity_id'] for r in corpus})
    for name, rows in [('queries', anchors), ('targets', corpus), ('lexical_fit', fit)]:
        pq.write_table(pa.Table.from_pylist(sorted(rows, key=lambda r: r['entity_id'])),
                       output / f'{name}.parquet', compression='zstd')
    write_json(output / 'truth.json', truth)
    metadata = {'status': 'PASS', 'kind': 'sampled-corpus diagnostic; NOT leaderboard forecast',
                'seed': seed, 'queries': len(anchors), 'targets': len(corpus), 'lexical_fit_rows': len(fit),
                'positive_links': sum(map(len, truth.values())),
                'release_manifest_sha256': sha256(release / 'manifest.json'),
                'labels_sha256': sha256(labels), 'script_sha256': sha256(Path(__file__)),
                'limitations': ['All sampled-query positives are deliberately included; target universe is much easier than full corpus.',
                                'Anchor IDs are disjoint from IDF fit records; near-duplicate business content has NOT been audited.',
                                'No learned ranker or retriever is trained. This is not the final train/validation split.']}
    metadata['files'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'manifest.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--labels', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--queries', type=int, default=500)
    p.add_argument('--distractors', type=int, default=30000)
    p.add_argument('--fit-rows', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20260925)
    a = p.parse_args()
    prepare(a.release, a.labels, a.output, a.queries, a.distractors, a.fit_rows, a.seed)
