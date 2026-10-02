"""Build model input for target IDs claimed by multiple Source-1 entities."""
import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def build(baseline, release, lookup, output):
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((baseline / 'manifest.json').read_text())
    if report['status'] != 'PASS':
        raise ValueError('Baseline submission is incomplete')
    output.mkdir(parents=True)
    con = duckdb.connect(str(output / 'audit.duckdb'))
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=4')
    con.execute(f"ATTACH {quote(lookup)} AS lookup (READ_ONLY)")
    matching = quote(baseline / 'matching_results.tsv')
    source = quote(release / 'test' / 'source1' / 'part-*.parquet')
    con.execute(f'''CREATE TABLE matches AS SELECT source1_entity_id,
        unnest(string_split(matched_entity_ids, ',')) AS candidate_id
        FROM read_csv({matching}, delim='\\t', header=true, all_varchar=true)
        WHERE matched_entity_ids != '' ''')
    con.execute('''CREATE TABLE conflicts AS SELECT candidate_id,count(*) AS owners
        FROM matches GROUP BY candidate_id HAVING count(*)>1''')
    con.execute('''CREATE TABLE conflict_edges AS
        SELECT m.source1_entity_id,m.candidate_id
        FROM matches m JOIN conflicts c USING(candidate_id)''')
    dest = output / 'conflict_pairs.parquet'
    con.execute(f'''COPY (SELECT e.source1_entity_id,e.candidate_id,
        q.name_normalized AS query_name,q.address_normalized AS query_address,
        q.country AS query_country,t.name_normalized AS target_name,
        t.address_normalized AS target_address,t.country AS target_country
        FROM conflict_edges e
        JOIN read_parquet({source}) q ON q.entity_id=e.source1_entity_id
        JOIN lookup.targets t ON t.entity_id=e.candidate_id)
        TO {quote(dest)} (FORMAT PARQUET, COMPRESSION ZSTD)''')
    matches, conflicts, edges = [con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                                 for table in ('matches', 'conflicts', 'conflict_edges')]
    con.close()
    if pq.ParquetFile(dest).metadata.num_rows != edges:
        raise ValueError('Missing conflict pair details')
    result = {'status': 'PASS', 'matches': matches, 'conflicting_targets': conflicts,
              'conflict_edges': edges,
              'baseline_manifest_sha256': sha256(baseline / 'manifest.json'),
              'conflict_pairs_sha256': sha256(dest)}
    write_json(output / 'manifest.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--release', type=Path,
                   default=Path('data/processed/conservative-v1-py311-u14'))
    p.add_argument('--lookup', type=Path,
                   default=Path('artifacts/a_tijil/test-lookup-v1.duckdb'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.baseline, a.release, a.lookup, a.output), indent=2))
