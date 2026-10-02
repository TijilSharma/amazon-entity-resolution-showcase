"""Replace prefiltered India top-10 B scores with specialist probabilities."""
import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ber.data.build_normalized import sha256, write_json


def merge(base, specialist, output, prefilter):
    if json.loads((specialist / 'manifest.json').read_text())['status'] != 'PASS':
        raise ValueError('India specialist scoring is incomplete')
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute('SET threads=4')
    con.execute('CREATE TABLE specialist AS SELECT * FROM read_parquet(?)',
                [str(specialist / 'part-*.parquet')])
    count, keys = con.execute('''SELECT count(*),count(DISTINCT
        (source1_entity_id,candidate_id)) FROM specialist''').fetchone()
    expected = json.loads((specialist / 'manifest.json').read_text())['rows']
    if count != expected or keys != count:
        raise ValueError('Specialist scores have duplicate or missing pairs')
    parts = sorted(base.glob('part-*.parquet'))
    if len(parts) != 87:
        raise ValueError(f'Expected 87 base top-10 shards, got {len(parts)}')
    replaced = total = 0
    for index, part in enumerate(parts, 1):
        dest = output / part.name
        report_path = output / (part.stem + '.json')
        base_report = json.loads((base / (part.stem + '.json')).read_text())
        if dest.exists() and report_path.exists():
            report = json.loads(report_path.read_text())
            if (report['input_sha256'] != base_report['input_sha256'] or
                    report['prefilter_min_B'] != prefilter or
                    report['rows'] != pq.ParquetFile(dest).metadata.num_rows):
                raise ValueError(f'Stale hybrid top-10 shard: {dest}')
        else:
            temp = dest.with_suffix('.parquet.tmp')
            input_sql = "'" + str(part).replace("'", "''") + "'"
            output_sql = "'" + str(temp).replace("'", "''") + "'"
            con.execute(f'''COPY (SELECT b.source1_entity_id,b.candidate_id,
                b.candidate_rank,coalesce(s.probability,b.probability) AS probability
                FROM read_parquet({input_sql}) b LEFT JOIN specialist s
                USING(source1_entity_id,candidate_id)) TO {output_sql}
                (FORMAT PARQUET, COMPRESSION ZSTD)''')
            rows = pq.ParquetFile(temp).metadata.num_rows
            if rows != base_report['rows']:
                temp.unlink(missing_ok=True)
                raise ValueError(f'Hybrid row count mismatch: {part.name}')
            temp.replace(dest)
            selected = con.execute('''SELECT count(*) FROM read_parquet(?) b
                JOIN specialist s USING(source1_entity_id,candidate_id)''',
                [str(part)]).fetchone()[0]
            report = {'part': part.name, 'rows': rows,
                      'india_specialist_rows': selected,
                      'prefilter_min_B': prefilter,
                      'input_sha256': base_report['input_sha256']}
            write_json(report_path, report)
        replaced += report['india_specialist_rows']
        total += report['rows']
        print(f'Hybrid top-10 {index}/87: {report["india_specialist_rows"]} India pairs',
              flush=True)
    con.close()
    if replaced != expected:
        raise ValueError(f'Specialist merge lost pairs: {replaced} != {expected}')
    manifest = {'status': 'PASS', 'rows': total, 'parts': len(parts),
                'india_specialist_rows': replaced,
                'prefilter_min_B': prefilter,
                'specialist_manifest_sha256': sha256(specialist / 'manifest.json')}
    write_json(output / 'manifest.json', manifest)
    (output / '_SUCCESS').write_text(sha256(output / 'manifest.json') + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path,
                   default=Path('artifacts/b_anushka/old-top10-test-probabilities-v1'))
    p.add_argument('--specialist', type=Path,
                   default=Path('artifacts/b_anushka/india-top10-specialist-test-v1'))
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/b_anushka/old-top10-hybrid-specialist-v1'))
    p.add_argument('--prefilter', type=float, default=0.0005)
    a = p.parse_args()
    print(json.dumps(merge(a.base, a.specialist, a.output, a.prefilter), indent=2))
