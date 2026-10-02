"""Compare lexical and multilingual routes on a checksummed sampled diagnostic."""
import argparse
from collections import defaultdict
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from ber.data.build_normalized import sha256, write_json

MODEL_ID = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
LEXICAL = ['name_char', 'address_char', 'name_word', 'address_word']
DENSE = ['dense_name', 'dense_name_address']


def merge_ranked(lists, limit):
    """Round-robin route quotas, deduplicating IDs without comparing score scales."""
    out, seen = [], set()
    for rank in range(max(map(len, lists), default=0)):
        for items in lists:
            if rank < len(items) and items[rank] not in seen:
                seen.add(items[rank])
                out.append(items[rank])
                if len(out) == limit:
                    return out
    return out


def evaluate(predictions, truth):
    links = hits = complete = 0
    recalls = []
    singletons = []
    for anchor, correct in truth.items():
        correct = set(correct)
        predicted = set(predictions.get(anchor, []))
        if not correct:
            singletons.append(len(predicted))
            continue
        found = len(correct & predicted)
        hits += found
        links += len(correct)
        recalls.append(found / len(correct))
        complete += found == len(correct)
    return {'positive_links': links, 'retrieved_positive_links': hits,
            'micro_recall': hits / links if links else None,
            'macro_recall_nonsingletons': float(np.mean(recalls)) if recalls else None,
            'complete_group_recall_nonsingletons': complete / len(recalls) if recalls else None,
            'nonsingleton_queries': len(recalls), 'singleton_queries': len(singletons),
            'mean_singleton_candidates': float(np.mean(singletons)) if singletons else None}


def lexical_search(queries, targets, fit, field, analyzer, grams, k):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=grams, lowercase=False,
                          dtype=np.float32, min_df=1, max_features=250000,
                          token_pattern=r'(?u)\b\w+\b')
    try:
        vec.fit([r[field] for r in fit])
    except ValueError as e:
        if 'empty vocabulary' in str(e):
            return [[] for _ in queries], {'vocabulary': 0}
        raise
    target_matrix = vec.transform([r[field] for r in targets]).tocsr()
    query_matrix = vec.transform([r[field] for r in queries]).tocsr()
    scores = sp_matmul_topn(query_matrix, target_matrix.T.tocsr(),
                            top_n=min(k, len(targets)), threshold=0.0, sort=True, n_threads=4)
    result = []
    for i in range(len(queries)):
        lo, hi = scores.indptr[i:i+2]
        found = [(int(j), float(s)) for j, s in zip(scores.indices[lo:hi], scores.data[lo:hi])]
        result.append(sorted(found, key=lambda x: (-x[1], targets[x[0]]['entity_id'])))
    return result, {'vocabulary': len(vec.vocabulary_), 'max_features': 250000,
                    'fit_records': len(fit), 'target_nonzeros': target_matrix.nnz,
                    'queries_with_zero_features': int(np.sum(np.diff(query_matrix.indptr) == 0))}


def text_view(row, combined):
    name = row['name_normalized']
    address = row['address_normalized']
    if not combined:
        return name
    return ' | '.join(x for x in (name, address) if x)


def dense_search(model, queries, targets, combined, k, output, batch_size):
    target_texts = [text_view(r, combined) for r in targets]
    query_texts = [text_view(r, combined) for r in queries]
    valid = np.array([i for i, text in enumerate(target_texts) if text], dtype=np.int64)
    if not len(valid):
        return [[] for _ in queries], {'valid_targets': 0}
    vectors = model.encode(['query: ' + target_texts[i] for i in valid],
                           batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    qvectors = model.encode(['query: ' + t for t in query_texts],
                            batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
    np.save(output, vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    values, indices = index.search(qvectors.astype(np.float32), min(k, len(valid)))
    found = [[(int(valid[j]), float(s)) for j, s in zip(js, ss)] if text else []
             for text, js, ss in zip(query_texts, indices, values)]
    # Counts before tokenizer truncation, so a short runtime setting is visible.
    lengths = [len(ids) for ids in model.tokenizer(
        ['query: ' + t for t in target_texts], truncation=False, add_special_tokens=True)['input_ids']]
    return found, {'index': 'IndexFlatIP; exact dense search, no ANN loss',
                   'valid_targets': len(valid), 'embedding_dimension': vectors.shape[1],
                   'target_sequences_exceeding_max_length': sum(n > model.max_seq_length for n in lengths),
                   'max_seq_length': model.max_seq_length}


def run(pilot, output, device='cuda', batch_size=64, model_cache='cache/huggingface'):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    import torch
    manifest = json.loads((pilot / 'manifest.json').read_text())
    for name, checksum in manifest['files'].items():
        if sha256(pilot / name) != checksum:
            raise ValueError(f'Pilot checksum mismatch: {name}')
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260925)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    queries, targets, fit = [pq.read_table(pilot / f'{name}.parquet').to_pylist()
                             for name in ('queries', 'targets', 'lexical_fit')]
    truth = json.loads((pilot / 'truth.json').read_text())
    qids = [r['entity_id'] for r in queries]
    tids = [r['entity_id'] for r in targets]
    routes, details = {}, {}
    for name in LEXICAL:
        field = 'name_normalized' if name.startswith('name') else 'address_normalized'
        char = name.endswith('char')
        started = time.time()
        routes[name], details[name] = lexical_search(queries, targets, fit, field,
                                                     'char_wb' if char else 'word',
                                                     (3, 5) if char else (1, 2), 200)
        details[name]['seconds'] = round(time.time() - started, 3)
        print(f'COMPLETE {name}: {details[name]}', flush=True)
    exact = defaultdict(list)
    for i, r in enumerate(targets):
        if r['name_normalized'] and r['address_normalized']:
            exact[(r['name_normalized'], r['address_normalized'])].append(i)
    routes['exact_name_address'] = [[(i, 1.0) for i in exact.get(
        (r['name_normalized'], r['address_normalized']), [])[:200]] for r in queries]
    patterns = ['config.json', 'model.safetensors', 'modules.json', '1_Pooling/config.json',
                'sentence_bert_config.json', 'sentencepiece.bpe.model', 'special_tokens_map.json',
                'tokenizer.json', 'tokenizer_config.json', 'README.md']
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, cache_dir=model_cache, allow_patterns=patterns)
    model = SentenceTransformer(snapshot, device=device, local_files_only=True)
    model.max_seq_length = 128
    for name in DENSE:
        started = time.time()
        routes[name], details[name] = dense_search(model, queries, targets, name.endswith('address'),
                                                    200, output / f'{name}_target_embeddings.npy', batch_size)
        details[name]['seconds'] = round(time.time() - started, 3)
        print(f'COMPLETE {name}: {details[name]}', flush=True)
    # Long-form score/rank provenance; scores across different routes are not calibrated.
    provenance = []
    ranked = {}
    for name, results in routes.items():
        ranked[name] = {}
        for anchor, found in zip(qids, results):
            ranked[name][anchor] = [tids[j] for j, _ in found]
            provenance.extend({'source1_entity_id': anchor, 'candidate_id': tids[j],
                               'route': name, 'route_rank': rank, 'score': score}
                              for rank, (j, score) in enumerate(found, 1))
    pq.write_table(pa.Table.from_pylist(provenance), output / 'route_candidates.parquet', compression='zstd')

    def combined(anchor, excluded=()):
        lexical = merge_ranked([ranked[r][anchor] for r in LEXICAL if r not in excluded], 200)
        dense = merge_ranked([ranked[r][anchor] for r in DENSE if r not in excluded], 200)
        return merge_ranked([ranked['exact_name_address'][anchor], lexical, dense], 200)

    ranked['lexical_union'] = {q: merge_ranked([ranked[r][q] for r in LEXICAL], 200) for q in qids}
    ranked['dense_union'] = {q: merge_ranked([ranked[r][q] for r in DENSE], 200) for q in qids}
    ranked['hybrid_union'] = {q: combined(q) for q in qids}
    ranked['hybrid_without_dense'] = {q: combined(q, DENSE) for q in qids}
    ranked['hybrid_without_lexical'] = {q: combined(q, LEXICAL) for q in qids}
    for route in LEXICAL + DENSE:
        ranked[f'hybrid_without_{route}'] = {q: combined(q, [route]) for q in qids}
    targets_by_id = {r['entity_id']: r for r in targets}
    queries_by_id = {r['entity_id']: r for r in queries}
    sliced_truth = {
        'all': truth,
        'India': {q: t for q, t in truth.items() if queries_by_id[q]['country'] == 'India'},
        'US': {q: t for q, t in truth.items() if queries_by_id[q]['country'] == 'US'},
        'cross_script_links': {},
        'missing_target_address_links': {},
    }
    for q, values in truth.items():
        for target in values:
            r = targets_by_id[target]
            a, b = set(queries_by_id[q]['name_scripts']), set(r['name_scripts'])
            if a and b and a.isdisjoint(b):
                sliced_truth['cross_script_links'].setdefault(q, []).append(target)
            if r['address_missing']:
                sliced_truth['missing_target_address_links'].setdefault(q, []).append(target)
    metrics = {name: {str(k): {s: evaluate({q: values[:k] for q, values in predictions.items()}, t)
                              for s, t in sliced_truth.items()} for k in (20, 50, 100, 200)}
               for name, predictions in ranked.items()}
    candidates = [{'source1_entity_id': q, 'candidate_id': t, 'candidate_rank': rank}
                  for q, values in ranked['hybrid_union'].items() for rank, t in enumerate(values, 1)]
    pq.write_table(pa.Table.from_pylist(candidates), output / 'hybrid_candidates.parquet', compression='zstd')
    misses = [{'source1_entity_id': q, 'target_id': t,
               'lexical_retrieved_at_200': t in ranked['lexical_union'][q],
               'dense_retrieved_at_200': t in ranked['dense_union'][q],
               'hybrid_retrieved_at_200': t in ranked['hybrid_union'][q]}
              for q, values in truth.items() for t in values
              if t not in ranked['hybrid_union'][q] or t not in ranked['lexical_union'][q]]
    write_json(output / 'misses.json', misses)
    report = {'status': 'PASS', 'scope': manifest['kind'], 'limitations': manifest['limitations'],
              'pilot_manifest_sha256': sha256(pilot / 'manifest.json'), 'model': MODEL_ID,
              'revision': REVISION, 'device': device, 'batch_size': batch_size,
              'prefix': 'query: for both sides', 'max_seq_length': 128,
              'script_sha256': sha256(Path(__file__)), 'details': details, 'metrics': metrics,
              'versions': {n: metadata.version(n) for n in ('numpy', 'scikit-learn', 'sparse-dot-topn',
                           'faiss-cpu', 'torch', 'sentence-transformers', 'transformers')}}
    write_json(output / 'report.json', report)
    (output / '_SUCCESS').write_text(sha256(output / 'report.json') + '\n')
    print('PASS: sampled-corpus diagnostic completed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--model-cache', default='cache/huggingface')
    a = p.parse_args()
    run(a.pilot, a.output, a.device, a.batch_size, a.model_cache)
