"""Build label-free features for the frozen pilot candidate pool; not a full-corpus job."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from ber.data.build_normalized import sha256, write_json
from ber.features.comparisons import pair_features, VERSION
from ber.retrieval.run_pilot import MODEL_ID, REVISION, LEXICAL, DENSE, text_view

ROUTES = LEXICAL + DENSE + ['exact_name_address']


def keyed(rows):
    lookup = {r['entity_id']: i for i,r in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate record IDs')
    return lookup


def validate_candidates(candidates, queries, targets):
    seen = set()
    for row in candidates:
        q,t = row['source1_entity_id'],row['candidate_id']
        if q not in queries or t not in targets:
            raise ValueError('Candidate refers to an unknown record')
        if (q,t) in seen:
            raise ValueError('Duplicate candidate pair')
        if row['candidate_rank'] < 1:
            raise ValueError('Invalid candidate rank')
        seen.add((q,t))


def encode_view(model, rows, view, batch_size):
    texts = [r['address_normalized'] if view=='address' else text_view(r,view=='name_address') for r in rows]
    valid = np.array([bool(t) for t in texts])
    vectors = np.zeros((len(rows), model.get_sentence_embedding_dimension()), dtype=np.float32)
    if valid.any():
        vectors[valid] = model.encode(['query: '+t for t in texts if t], batch_size=batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return vectors, valid


def aligned_cosines(qvectors, tvectors, qi, ti, qvalid, tvalid):
    # Compute only nominated pairs, never all query x target similarities.
    out = np.full(len(qi), np.nan, dtype=np.float32)
    for start in range(0,len(qi),10000):
        q,t=qi[start:start+10000],ti[start:start+10000]
        values=np.einsum('ij,ij->i',qvectors[q],tvectors[t])
        values[~(qvalid[q]&tvalid[t])]=np.nan
        out[start:start+len(q)]=np.clip(values,-1,1)
    return out


def build(pilot, retrieval, output, device='cuda', batch_size=64):
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from ber.features import comparisons
    started=time.time()
    m=json.loads((pilot/'manifest.json').read_text())
    if m['status']!='PASS':
        raise ValueError('Incomplete pilot')
    # Deliberately do not read truth.json: labels must not enter feature generation.
    for name in ('queries.parquet','targets.parquet','lexical_fit.parquet'):
        if sha256(pilot/name)!=m['files'][name]:
            raise ValueError(f'Pilot checksum mismatch: {name}')
    report=json.loads((retrieval/'report.json').read_text())
    if report['status']!='PASS' or (retrieval/'_SUCCESS').read_text().strip()!=sha256(retrieval/'report.json'):
        raise ValueError('Incomplete retrieval')
    if report['pilot_manifest_sha256']!=sha256(pilot/'manifest.json'):
        raise ValueError('Retrieval and pilot releases differ')
    queries,targets,fit=[pq.read_table(pilot/f'{name}.parquet').to_pylist()
                         for name in ('queries','targets','lexical_fit')]
    qlookup,tlookup=keyed(queries),keyed(targets)
    candidates=pq.read_table(retrieval/'hybrid_candidates.parquet').to_pylist()
    validate_candidates(candidates,qlookup,tlookup)
    if not candidates:
        raise ValueError('No candidate pairs')
    output.mkdir(parents=True,exist_ok=False)
    qi=np.array([qlookup[r['source1_entity_id']] for r in candidates])
    ti=np.array([tlookup[r['candidate_id']] for r in candidates])
    rows=[{**c, **pair_features(queries[q],targets[t])} for c,q,t in zip(candidates,qi,ti)]
    print(f'Explicit comparisons: {len(rows):,} pairs',flush=True)
    positions={(r['source1_entity_id'],r['candidate_id']):i for i,r in enumerate(rows)}
    for row in rows:
        for route in ROUTES:
            row[f'retrieved_{route}']=False
            row[f'retrieval_{route}_rank']=np.nan
            row[f'retrieval_{route}_score']=np.nan
    seen_routes=set()
    for r in pq.read_table(retrieval/'route_candidates.parquet').to_pylist():
        if r['route'] not in ROUTES or r['source1_entity_id'] not in qlookup or r['candidate_id'] not in tlookup:
            raise ValueError('Invalid route provenance')
        key=(r['source1_entity_id'],r['candidate_id'],r['route'])
        if key in seen_routes:
            raise ValueError('Duplicate route provenance')
        seen_routes.add(key)
        index=positions.get(key[:2])
        if index is not None:
            row=rows[index]
            row[f"retrieved_{r['route']}"]=True
            row[f"retrieval_{r['route']}_rank"]=r['route_rank']
            row[f"retrieval_{r['route']}_score"]=r['score']
    for row in rows:
        row['retrieval_route_count']=sum(row[f'retrieved_{route}'] for route in ROUTES)
    # Calculate TF-IDF cosine for EVERY nominated pair, including pairs absent
    # from that lexical route's top-200. Absence is not zero similarity.
    lexical_details={}
    for route in LEXICAL:
        field='name_normalized' if route.startswith('name') else 'address_normalized'
        char=route.endswith('char')
        vectorizer=TfidfVectorizer(analyzer='char_wb' if char else 'word', ngram_range=(3,5) if char else (1,2),
                                  lowercase=False,dtype=np.float32,min_df=1,max_features=250000,
                                  token_pattern=r'(?u)\b\w+\b')
        vectorizer.fit([r[field] for r in fit])
        qmat=vectorizer.transform([r[field] for r in queries]).tocsr()
        tmat=vectorizer.transform([r[field] for r in targets]).tocsr()
        qvalid=np.diff(qmat.indptr)>0
        tvalid=np.diff(tmat.indptr)>0
        values=np.full(len(rows),np.nan,dtype=np.float32)
        for start in range(0,len(rows),10000):
            q,t=qi[start:start+10000],ti[start:start+10000]
            v=np.asarray(qmat[q].multiply(tmat[t]).sum(axis=1)).ravel()
            v[~(qvalid[q]&tvalid[t])]=np.nan
            values[start:start+len(q)]=v
        for row,value,q,t in zip(rows,values,qi,ti):
            row[f'tfidf_{route}_cosine']=float(value)
            row[f'tfidf_{route}_query_zero_vector']=not bool(qvalid[q])
            row[f'tfidf_{route}_target_zero_vector']=not bool(tvalid[t])
        lexical_details[route]={'vocabulary':len(vectorizer.vocabulary_),'missing_pair_comparisons':int(np.isnan(values).sum())}
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    patterns=['config.json','model.safetensors','modules.json','1_Pooling/config.json','sentence_bert_config.json',
              'sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json','README.md']
    snapshot=snapshot_download(MODEL_ID,revision=REVISION,cache_dir='cache/huggingface',allow_patterns=patterns)
    model=SentenceTransformer(snapshot,device=device,local_files_only=True)
    model.max_seq_length=128
    embedding_details={}
    for view in ('name','address','name_address'):
        qvectors,qvalid=encode_view(model,queries,view,batch_size)
        tvectors,tvalid=encode_view(model,targets,view,batch_size)
        values=aligned_cosines(qvectors,tvectors,qi,ti,qvalid,tvalid)
        for row,value in zip(rows,values):
            row[f'embedding_{view}_cosine']=float(value)
        embedding_details[view]={'missing_pair_comparisons':int(np.isnan(values).sum())}
        print(f'Embedding similarities complete: {view}',flush=True)
    # Explicit float32 numeric schema: NaN denotes undefined evidence.
    identifiers=['source1_entity_id','candidate_id']
    feature_names=[k for k in rows[0] if k not in identifiers]
    schema=pa.schema([pa.field(k,pa.string(),nullable=False) for k in identifiers]+
                     [pa.field(k,pa.float32(),nullable=False) for k in feature_names])
    table=pa.Table.from_pylist(rows,schema=schema)
    values=np.array([[row[k] for k in feature_names] for row in rows],dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError('Infinite feature value')
    path=output/'pair_features.parquet'
    pq.write_table(table,path,compression='zstd',row_group_size=10000)
    restored=pq.read_table(path)
    if restored.num_rows!=len(candidates) or restored.select(identifiers).to_pylist()!=table.select(identifiers).to_pylist():
        raise ValueError('Feature write changed pair alignment')
    for column in feature_names:
        np.testing.assert_allclose(restored[column].to_numpy(),table[column].to_numpy(),equal_nan=True,rtol=0,atol=0)
    write_json(output/'feature_schema.json',{'version':VERSION,'identifiers':identifiers,
        'features':feature_names,'numeric_dtype':'float32','missing_value':'IEEE NaN; never fill indiscriminately with zero',
        'label_column':None,'id_policy':'Identifiers are join keys, never model features.'})
    result={'status':'PASS','version':VERSION,'scope':'sampled-pilot candidate pairs; not a final training/validation dataset',
            'rows':len(rows),'feature_count':len(feature_names),'device':device,'batch_size':batch_size,
            'model':MODEL_ID,'revision':REVISION,'prefix':'query:','max_seq_length':128,
            'input_hashes':{str(p):sha256(p) for p in [pilot/'manifest.json',retrieval/'report.json',
                                                     retrieval/'hybrid_candidates.parquet',retrieval/'route_candidates.parquet']},
            'code_hashes':{p.name:sha256(p) for p in [Path(__file__),Path(comparisons.__file__)]},
            'versions':{n:metadata.version(n) for n in ['numpy','pyarrow','rapidfuzz','scikit-learn','torch','sentence-transformers']},
            'lexical':lexical_details,'embeddings':embedding_details,
            'nan_counts':{k:int(np.isnan(values[:,i]).sum()) for i,k in enumerate(feature_names)},
            'output_hashes':{p.name:sha256(p) for p in [path,output/'feature_schema.json']},
            'elapsed_seconds':round(time.time()-started,2)}
    write_json(output/'manifest.json',result)
    (output/'_SUCCESS').write_text(sha256(output/'manifest.json')+'\n')
    print(f'PASS: {len(rows):,} pairs, {len(feature_names)} features',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args()
    build(a.pilot,a.retrieval,a.output,a.device,a.batch_size)
