"""Apply a validated rescue stacker to frozen B test probabilities."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

from ber.data.build_normalized import sha256, write_json
from ber.models.b_crossencoder.rescue_stacker import matrix


SCHEMA = pa.schema([('source1_entity_id', pa.string()), ('candidate_id', pa.string()),
                    ('probability', pa.float32())])


def score_part(part, B_scores, model_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    result = output / part.name
    report_path = output / (part.stem + '.json')
    if result.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if report['input_sha256'] == sha256(part) and report['B_input_sha256'] == sha256(B_scores):
            return report
        raise ValueError(f'Existing stacker score differs from input: {part.name}')
    report = json.loads((model_dir / 'report.json').read_text())
    if report['status'] != 'PASS':
        raise ValueError('Rescue stacker is incomplete')
    names = json.loads((model_dir / 'feature_names.json').read_text())
    model = CatBoostClassifier()
    model.load_model(str(model_dir / 'model.cbm'))
    probabilities = {}
    for batch in pq.ParquetFile(B_scores).iter_batches():
        for row in batch.to_pylist():
            key = (row['source1_entity_id'], row['candidate_id'])
            if key in probabilities:
                raise ValueError(f'Duplicate B probability: {key}')
            probabilities[key] = row['probability']
    began = time.time()
    seen = 0
    temp = result.with_suffix('.parquet.tmp')
    with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer:
        for batch in pq.ParquetFile(part).iter_batches(batch_size=20_000):
            rows = batch.to_pylist()
            for row in rows:
                row['B_probability'] = probabilities.pop((row['source1_entity_id'],
                                                          row['candidate_id']))
            x, _ = matrix(rows, names)
            values = model.predict_proba(x)[:, 1]
            out = [{'source1_entity_id': row['source1_entity_id'],
                    'candidate_id': row['candidate_id'],
                    'probability': float(p)} for row, p in zip(rows, values)]
            writer.write_table(pa.Table.from_pylist(out, schema=SCHEMA))
            seen += len(rows)
    if probabilities:
        temp.unlink(missing_ok=True)
        raise ValueError(f'B scores contain {len(probabilities)} surplus pairs')
    if seen != pq.ParquetFile(part).metadata.num_rows:
        temp.unlink(missing_ok=True)
        raise ValueError('Stacker scoring row count mismatch')
    temp.replace(result)
    result_report = {'part': part.name, 'rows_with_probabilities': seen,
                     'input_sha256': sha256(part), 'B_input_sha256': sha256(B_scores),
                     'model_report_sha256': sha256(model_dir / 'report.json'),
                     'elapsed_seconds': round(time.time()-began, 2)}
    write_json(report_path, result_report)
    print(json.dumps(result_report), flush=True)
    return result_report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',type=Path,required=True)
    p.add_argument('--B-scores',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('artifacts/b_anushka/india-rescue-stacker-v1'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    score_part(a.part,a.B_scores,a.model,a.output)
