"""Train-only alias and field-balanced rescue for difficult India queries.

This stage never reads ground truth at inference. It preserves the original
candidate set and emits additional target IDs for a separate reranker.
"""
import argparse
import ctypes as C
import ctypes.util
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from ber.features.comparisons import tokens
from ber.retrieval.full_fuzzy import keys


LEGAL = {'private', 'limited', 'pvt', 'ltd', 'llp', 'llc', 'inc',
         'corporation', 'corp', 'co', 'company', 'pra', 'li', 'com', 'm', 's'}


def romanizer():
    path = ctypes.util.find_library('icui18n')
    if not path:
        raise RuntimeError('System ICU is required for India rescue')
    lib = C.CDLL(path)
    version = re.search(r'\.so\.(\d+)', path).group(1)
    op = getattr(lib, 'utrans_openU_' + version)
    op.restype = C.c_void_p
    op.argtypes = [C.POINTER(C.c_uint16), C.c_int32, C.c_int,
                   C.c_void_p, C.c_int32, C.c_void_p, C.POINTER(C.c_int32)]
    trans = getattr(lib, 'utrans_transUChars_' + version)
    trans.argtypes = [C.c_void_p, C.POINTER(C.c_uint16), C.POINTER(C.c_int32),
                      C.c_int32, C.c_int32, C.POINTER(C.c_int32), C.POINTER(C.c_int32)]
    ident = 'Any-Latin; Latin-ASCII; Lower()'.encode('utf-16-le')
    arr = (C.c_uint16 * (len(ident) // 2)).from_buffer_copy(ident)
    err = C.c_int32(0)
    handle = op(arr, len(arr), 0, None, 0, None, C.byref(err))
    if not handle or err.value > 0:
        raise RuntimeError(f'ICU transliterator initialization failed: {err.value}')

    @lru_cache(maxsize=200_000)
    def roman(text):
        if text.isascii():
            return text
        raw = text.encode('utf-16-le')
        size = max(128, len(raw) * 8)
        buf = (C.c_uint16 * size)()
        C.memmove(buf, raw, len(raw))
        length = C.c_int32(len(raw) // 2)
        limit = C.c_int32(length.value)
        error = C.c_int32(0)
        trans(handle, buf, C.byref(length), size, 0, C.byref(limit), C.byref(error))
        if error.value > 0:
            raise RuntimeError(f'ICU transliteration failed: {error.value}')
        return C.string_at(buf, length.value * 2).decode('utf-16-le')
    return roman


class Rescue:
    def __init__(self, index, aliases, max_frequency=50_000, preselect=2_000):
        self.db = sqlite3.connect(f'file:{index}?mode=ro', uri=True)
        self.db.execute('PRAGMA query_only=ON')
        self.aliases = json.loads(Path(aliases).read_text())
        self.roman = romanizer()
        self.canonical = lru_cache(maxsize=200_000)(self._canonical)
        self.counts = {}
        self.max_frequency = max_frequency
        self.preselect = preselect
        self.target_count = self.db.execute('SELECT count(*) FROM targets').fetchone()[0]

    def _canonical(self, value):
        vals = [self.aliases['aliases'][t]['latin'] if t in self.aliases['aliases']
                else self.roman(t) for t in tokens(value)]
        return ' '.join(sorted(t for t in vals if t not in LEGAL))

    def search(self, query, excluded=(), cap=50):
        groups = []
        for field in ('name_normalized', 'address_normalized'):
            view = {'name_normalized': '', 'address_normalized': ''}
            view[field] = query[field]
            group = set(keys(view))
            if field == 'name_normalized':
                words = re.findall(r'[^\W_]+', query[field])
                for phrase in (''.join(t for t in words if t not in LEGAL), ''.join(words)):
                    if phrase:
                        group.add('word:' + phrase)
            groups.append(group)
        extra = set()
        for token in tokens(query['name_normalized']):
            equivalent = self.aliases['aliases'].get(token, {}).get('latin', token)
            if equivalent in LEGAL:
                continue
            if equivalent != token:
                extra.add('word:' + equivalent)
            extra.update('word:' + v for v in self.aliases['reverse'].get(equivalent, []))
        groups.append(extra)
        selected = {}
        for group, quota in zip(groups, (10, 12, 12)):
            valid = []
            for key in group:
                if key not in self.counts:
                    row = self.db.execute('SELECT count FROM key_counts WHERE key=?', (key,)).fetchone()
                    self.counts[key] = row[0] if row else 0
                count = self.counts[key]
                if 0 < count <= self.max_frequency:
                    valid.append((count, key))
            for count, key in sorted(valid)[:quota]:
                selected[key] = count
        if not selected:
            return []
        values = ','.join('(?,?)' for _ in selected)
        params = []
        for key, count in selected.items():
            params.extend((key, math.log1p(self.target_count / count)))
        sql = (f'WITH q(key,weight) AS (VALUES {values}), shortlist AS '
               '(SELECT e.target_id, SUM(q.weight) evidence FROM q CROSS JOIN entries e '
               'INDEXED BY entries_lookup WHERE e.key=q.key GROUP BY e.target_id '
               f'ORDER BY evidence DESC,e.target_id LIMIT {self.preselect}) '
               'SELECT s.target_id,s.evidence,t.name,t.address FROM shortlist s '
               'JOIN targets t ON t.target_id=s.target_id')
        qname, qaddr = query['name_normalized'], query['address_normalized']
        qcanon = self.canonical(qname)
        hits = []
        for tid, evidence, name, address in self.db.execute(sql, params):
            tc = self.canonical(name)
            ns = max(fuzz.ratio(qname, name), fuzz.token_set_ratio(qname, name),
                     fuzz.ratio(qcanon, tc), fuzz.token_set_ratio(qcanon, tc),
                     fuzz.ratio(qcanon.replace(' ', ''), tc.replace(' ', '')))
            ads = max(fuzz.ratio(qaddr, address), fuzz.token_set_ratio(qaddr, address)) \
                if qaddr and address else 0.0
            hits.append((tid, float(ns), float(ads), float(evidence), name, address))
        name_order = sorted(hits, key=lambda x: (-x[1], -x[3], x[0]))
        addr_order = sorted(hits, key=lambda x: (-x[2], -x[3], x[0]))
        idf_order = sorted(hits, key=lambda x: (-x[3], x[0]))
        seen = set(excluded)
        result = []
        for i in range(len(hits)):
            for order in (name_order, addr_order, idf_order):
                row = order[i]
                if row[0] not in seen:
                    seen.add(row[0])
                    result.append(row)
                    if len(result) >= cap:
                        return result
        return result


SCHEMA = pa.schema([
    ('source1_entity_id', pa.string()), ('candidate_id', pa.string()),
    ('rescue_rank', pa.int32()), ('name_similarity', pa.float32()),
    ('address_similarity', pa.float32()), ('idf_evidence', pa.float32()),
    ('query_name', pa.string()), ('query_address', pa.string()),
    ('target_name', pa.string()), ('target_address', pa.string())])


def score_counts(path):
    result = Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=['source1_entity_id']):
        result.update(batch.column(0).to_pylist())
    return result


def run_shard(index, aliases, release, union, scores, output, shard, min_old_matches=0,
              max_old_matches=0,
              cap=50, max_frequency=50_000, preselect=2_000):
    path = release / 'test' / 'source1' / f'part-{shard:05d}.parquet'
    name = path.name
    destination = output / name
    if destination.exists():
        print(f'Already complete: {name}', flush=True)
        return
    counts = score_counts(scores / name)
    queries = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=['entity_id', 'country',
                                                              'name_normalized', 'address_normalized']):
        for row in batch.to_pylist():
            if (row['country'] == 'India' and
                    min_old_matches <= counts[row['entity_id']] <= max_old_matches):
                queries[row['entity_id']] = row
    excluded = defaultdict(set)
    if queries:
        for batch in pq.ParquetFile(union / name).iter_batches(columns=['source1_entity_id', 'candidate_id']):
            for row in batch.to_pylist():
                if row['source1_entity_id'] in queries:
                    excluded[row['source1_entity_id']].add(row['candidate_id'])
    rescue = Rescue(index, aliases, max_frequency, preselect)
    rows = []
    for i, (anchor, query) in enumerate(queries.items(), 1):
        for rank, (tid, ns, ads, evidence, tname, taddr) in enumerate(
                rescue.search(query, excluded[anchor], cap), 1):
            rows.append({'source1_entity_id': anchor, 'candidate_id': tid,
                         'rescue_rank': rank, 'name_similarity': ns,
                         'address_similarity': ads, 'idf_evidence': evidence,
                         'query_name': query['name_normalized'],
                         'query_address': query['address_normalized'],
                         'target_name': tname, 'target_address': taddr})
        if i % 500 == 0:
            print(f'{name}: {i}/{len(queries)} India queries; {len(rows)} new pairs', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression='zstd')
    tmp.replace(destination)
    print(f'{name}: PASS {len(queries)} queries, {len(rows)} pairs', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--aliases', type=Path, required=True)
    p.add_argument('--release', type=Path, default=Path('data/processed/conservative-v1-py311-u14'))
    p.add_argument('--union', type=Path, default=Path('candidates/full-union-v1/test'))
    p.add_argument('--scores', type=Path, default=Path('artifacts/a_tijil/full-test-scores-v1'))
    p.add_argument('--output', type=Path, default=Path('candidates/india-rescue-v1/test'))
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--min-old-matches', type=int, default=0)
    p.add_argument('--max-old-matches', type=int, default=0)
    p.add_argument('--cap', type=int, default=50)
    p.add_argument('--max-frequency', type=int, default=50_000)
    p.add_argument('--preselect', type=int, default=2_000)
    args = p.parse_args()
    run_shard(args.index, args.aliases, args.release, args.union, args.scores,
              args.output, args.shard, args.min_old_matches,
              args.max_old_matches, args.cap,
              args.max_frequency, args.preselect)
