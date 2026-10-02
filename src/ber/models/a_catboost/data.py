"""Validated pair labels and diagnostic group partitions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from ber.data.build_normalized import sha256

ROLES=('train','early_stop','calibration','holdout')


def groups(truth, records):
    parent={q:q for q in truth}
    def find(q):
        while parent[q]!=q:
            parent[q]=parent[parent[q]]
            q=parent[q]
        return q
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    owners={}
    fingerprints={}
    for q in sorted(truth):
        for t in truth[q]:
            if t in owners:
                union(q,owners[t])
            owners[t]=q
        r=records[q]
        if r['name_normalized'] and r['address_normalized']:
            key=(r['country'],r['name_normalized'],r['address_normalized'])
            if key in fingerprints:
                union(q,fingerprints[key])
            fingerprints[key]=q
    return {q:find(q) for q in truth}


def make_splits(truth, records, seed):
    group_map=groups(truth,records)
    unique=sorted(set(group_map.values()),key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).digest())
    if len(unique)<8:
        raise ValueError('At least eight diagnostic groups required')
    n=len(unique)
    boundaries=(int(.60*n),int(.75*n),int(.90*n),n)
    assignments={}
    start=0
    for role,end in zip(ROLES,boundaries):
        for group in unique[start:end]:
            assignments[group]=role
        start=end
    return {q:assignments[g] for q,g in group_map.items()},group_map


def validate_splits(splits, truth, group_map):
    if set(splits)!=set(truth) or set(splits.values())!=set(ROLES):
        raise ValueError('Split must assign every anchor exactly once to all four roles')
    seen={}
    for q,g in group_map.items():
        if g in seen and seen[g]!=splits[q]:
            raise ValueError('Related anchors cross split boundaries')
        seen[g]=splits[q]


def load(feature_root,pilot):
    manifest=json.loads((feature_root/'manifest.json').read_text())
    if manifest['status']!='PASS' or (feature_root/'_SUCCESS').read_text().strip()!=sha256(feature_root/'manifest.json'):
        raise ValueError('Incomplete feature release')
    for name,h in manifest['output_hashes'].items():
        if sha256(feature_root/name)!=h:
            raise ValueError('Feature checksum mismatch')
    pm=json.loads((pilot/'manifest.json').read_text())
    pilot_hash=sha256(pilot/'manifest.json')
    recorded=[h for name,h in manifest['input_hashes'].items() if Path(name).name=='manifest.json']
    if pilot_hash not in recorded:
        raise ValueError('Feature/pilot mismatch')
    for name,h in pm['files'].items():
        if sha256(pilot/name)!=h:
            raise ValueError(f'Pilot checksum mismatch: {name}')
    truth=json.loads((pilot/'truth.json').read_text())
    records={r['entity_id']:r for r in pq.read_table(pilot/'queries.parquet').to_pylist()}
    if set(records)!=set(truth):
        raise ValueError('Query/label coverage differs')
    target_ids=set(pq.read_table(pilot/'targets.parquet',columns=['entity_id'])['entity_id'].to_pylist())
    for q,values in truth.items():
        if len(values)!=len(set(values)):
            raise ValueError('Duplicate ground-truth target')
        if not set(values)<=target_ids:
            raise ValueError('Ground truth references missing target')
    schema=json.loads((feature_root/'feature_schema.json').read_text())
    names=schema['features']
    if len(names)!=len(set(names)) or set(names)&{'source1_entity_id','candidate_id','label','target'}:
        raise ValueError('Invalid model feature list')
    table=pq.read_table(feature_root/'pair_features.parquet')
    pairs=list(zip(table['source1_entity_id'].to_pylist(),table['candidate_id'].to_pylist()))
    if len(pairs)!=len(set(pairs)) or any(q not in truth or t not in target_ids for q,t in pairs):
        raise ValueError('Invalid candidate pairs')
    matrix=np.column_stack([table[name].to_numpy() for name in names]).astype(np.float32)
    if np.isinf(matrix).any():
        raise ValueError('Infinite feature values')
    sets={q:set(v) for q,v in truth.items()}
    labels=np.array([int(t in sets[q]) for q,t in pairs],dtype=np.int8)
    return matrix,labels,names,pairs,truth,records
