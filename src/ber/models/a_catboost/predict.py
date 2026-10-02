"""Apply a saved Pipeline A model and frozen threshold, without reading labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier,Pool
from ber.data.build_normalized import sha256,write_json
from ber.evaluation.entity_score import decisions
from .train import export_lists


def predict(model_root,features,queries,output):
    report=json.loads((model_root/'report.json').read_text())
    if report['status']!='PASS' or (model_root/'_SUCCESS').read_text().strip()!=sha256(model_root/'report.json'):
        raise ValueError('Incomplete model run')
    for name in ('model.cbm','threshold.json','feature_names.json'):
        if sha256(model_root/name)!=report['output_hashes'][name]:
            raise ValueError('Model/threshold checksum mismatch')
    manifest=json.loads((features/'manifest.json').read_text())
    if manifest['status']!='PASS' or (features/'_SUCCESS').read_text().strip()!=sha256(features/'manifest.json'):
        raise ValueError('Incomplete feature release')
    for name,h in manifest['output_hashes'].items():
        if sha256(features/name)!=h:
            raise ValueError('Feature checksum mismatch')
    names=json.loads((model_root/'feature_names.json').read_text())
    schema=json.loads((features/'feature_schema.json').read_text())
    if schema['features']!=names:
        raise ValueError('Feature schema/order differs from trained model')
    anchors=pq.read_table(queries,columns=['entity_id'])['entity_id'].to_pylist()
    if len(anchors)!=len(set(anchors)):
        raise ValueError('Duplicate query IDs')
    table=pq.read_table(features/'pair_features.parquet')
    pairs=list(zip(table['source1_entity_id'].to_pylist(),table['candidate_id'].to_pylist()))
    X=np.column_stack([table[name].to_numpy() for name in names]).astype(np.float32)
    if np.isinf(X).any():
        raise ValueError('Infinite feature value')
    model=CatBoostClassifier()
    model.load_model(str(model_root/'model.cbm'))
    if model.feature_names_!=names:
        raise ValueError('Model feature names differ')
    probabilities=model.predict_proba(Pool(X,feature_names=names))[:,1] if len(pairs) else np.array([])
    threshold=json.loads((model_root/'threshold.json').read_text())['threshold']
    matches=decisions(anchors,pairs,probabilities,threshold)
    candidates=decisions(anchors,pairs,probabilities,0.)
    output.mkdir(parents=True,exist_ok=False)
    export_lists(output/'matching_results.tsv',anchors,matches,'matched_entity_ids')
    export_lists(output/'candidate_pairs.tsv',anchors,candidates,'candidate_entity_ids')
    prediction_schema=pa.schema([('source1_entity_id',pa.string()),('candidate_id',pa.string()),('probability',pa.float64())])
    pq.write_table(pa.Table.from_pylist([
        {'source1_entity_id':q,'candidate_id':t,'probability':float(p)}
        for (q,t),p in zip(pairs,probabilities)],schema=prediction_schema),output/'pair_predictions.parquet',compression='zstd')
    result={'status':'PASS','scope':'predictions for supplied queries; not automatically a competition-test submission',
            'anchors':len(anchors),'pairs':len(pairs),'threshold':threshold,
            'input_hashes':{str(p):sha256(p) for p in [model_root/'report.json',features/'manifest.json',queries]},
            'output_hashes':{p.name:sha256(p) for p in output.iterdir()}}
    write_json(output/'manifest.json',result)
    (output/'_SUCCESS').write_text(sha256(output/'manifest.json')+'\n')
    print(f'PASS: {len(anchors)} anchors, {len(pairs)} pairs; no labels read')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    predict(a.model,a.features,a.queries,a.output)
