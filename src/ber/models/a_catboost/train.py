"""Pipeline A diagnostic: train / early-stop / threshold-calibration / holdout."""
import argparse
import csv
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from ber.data.build_normalized import sha256,write_json
from ber.evaluation.entity_score import score,decisions,choose_threshold
from .data import load,make_splits,groups,validate_splits,ROLES


def export_lists(path,anchors,predictions,column):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id',column])
        for q in sorted(anchors):
            writer.writerow([q,','.join(sorted(predictions.get(q,[])))])


def run(features,pilot,output,config_path,splits_path=None):
    start=time.time()
    config=json.loads(config_path.read_text())
    X,y,names,pairs,truth,records=load(features,pilot)
    if splits_path:
        assignments=json.loads(splits_path.read_text())
        group_map=groups(truth,records)
    else:
        assignments,group_map=make_splits(truth,records,config['seed'])
    validate_splits(assignments,truth,group_map)
    masks={role:np.array([assignments[q]==role for q,t in pairs]) for role in ROLES}
    for role in ('train','early_stop'):
        if len(np.unique(y[masks[role]]))!=2:
            raise ValueError(f'{role} needs both label classes')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'splits.json',assignments)
    write_json(output/'config.json',config)
    # No negative undersampling or class weighting in the first baseline.
    model=CatBoostClassifier(iterations=config['iterations'],depth=config['depth'],
        learning_rate=config['learning_rate'],l2_leaf_reg=config['l2_leaf_reg'],
        random_seed=config['seed'],thread_count=config['thread_count'],task_type='CPU',
        loss_function='Logloss',eval_metric='Logloss',nan_mode='Min',
        allow_writing_files=False,verbose=100)
    model.fit(Pool(X[masks['train']],y[masks['train']],feature_names=names),
              eval_set=Pool(X[masks['early_stop']],y[masks['early_stop']],feature_names=names),
              early_stopping_rounds=config['early_stopping_rounds'],use_best_model=True)
    model.save_model(str(output/'model.cbm'))
    write_json(output/'feature_names.json',names)
    probabilities=model.predict_proba(Pool(X,feature_names=names))[:,1]
    reloaded=CatBoostClassifier()
    reloaded.load_model(str(output/'model.cbm'))
    np.testing.assert_allclose(reloaded.predict_proba(Pool(X[:1000],feature_names=names))[:,1],
                               probabilities[:1000],rtol=0,atol=1e-12)
    def partition(role):
        indices=np.flatnonzero(masks[role])
        return ({q:v for q,v in truth.items() if assignments[q]==role},
                [pairs[i] for i in indices],probabilities[indices])
    calibration_truth,calibration_pairs,calibration_p=partition('calibration')
    selected,curve=choose_threshold(calibration_truth,calibration_pairs,calibration_p)
    write_json(output/'threshold.json',{**selected,'selected_on':'calibration ONLY',
                                       'decision':'probability >= threshold',
                                       'probability_calibration':'none; raw CatBoost score'})
    pq.write_table(pa.Table.from_pylist(curve),output/'threshold_curve.parquet',compression='zstd')
    reports={}
    for role in ROLES:
        actual,role_pairs,probs=partition(role)
        predicted=decisions(actual,role_pairs,probs,selected['threshold'])
        candidates=decisions(actual,role_pairs,probs,0.0)
        result=score(actual,predicted)
        oracle={q:sorted(set(candidates[q])&set(actual[q])) for q in actual}
        result['candidate_oracle_macro_f05']=score(actual,oracle)['macro_f05']
        result['missing_positive_links_before_model']=sum(len(set(actual[q])-set(candidates[q])) for q in actual)
        result['candidate_pairs']=len(role_pairs)
        result['positive_pairs']=int(y[masks[role]].sum())
        result['negative_pairs']=int((1-y[masks[role]]).sum())
        result['at_threshold_0_5']=score(actual,decisions(actual,role_pairs,probs,.5))['macro_f05']
        result['country_slices']={}
        for country in sorted({records[q]['country'] for q in actual}):
            subset={q:v for q,v in actual.items() if records[q]['country']==country}
            result['country_slices'][country]=score(subset,{q:predicted[q] for q in subset})['macro_f05']
        reports[role]=result
        export_lists(output/f'{role}_matching_results.tsv',actual,predicted,'matched_entity_ids')
        export_lists(output/f'{role}_candidate_pairs.tsv',actual,candidates,'candidate_entity_ids')
    pq.write_table(pa.Table.from_pylist([
        {'source1_entity_id':q,'candidate_id':t,'split':assignments[q],
         'label':int(label),'probability':float(p)}
        for (q,t),label,p in zip(pairs,y,probabilities)]),output/'pair_predictions.parquet',compression='zstd')
    write_json(output/'feature_importance.json',
               sorted([{'feature':name,'importance':float(value)}
                       for name,value in zip(names,model.feature_importances_)],
                      key=lambda r:-r['importance']))
    report={'status':'PASS','scope':'500-anchor sampled-corpus diagnostic; NOT a leaderboard forecast',
            'limitations':['Sample target pool contains deliberately included positives.',
                          'Known shared true targets and exact query name/address fingerprints are grouped; near-duplicate leakage remains unaudited.',
                          'Holdout is small. Do not repeatedly tune against its reported score.',
                          'Features and matrices are loaded in memory; this command is not a full-scale training job.'],
            'split_origin':str(splits_path) if splits_path else 'local diagnostic only; does not replace the team split',
            'anchor_counts':{r:sum(v==r for v in assignments.values()) for r in ROLES},
            'group_count':len(set(group_map.values())),'features':len(names),
            'best_iteration':model.get_best_iteration(),'tree_count':model.tree_count_,
            'threshold':selected,'metrics':reports,'config':config,
            'versions':{n:metadata.version(n) for n in ['catboost','numpy','pyarrow']},
            'input_hashes':{str(p):sha256(p) for p in [features/'manifest.json',pilot/'manifest.json',config_path]},
            'elapsed_seconds':round(time.time()-start,2)}
    from . import data
    from ber.evaluation import entity_score
    report['code_hashes']={str(p):sha256(p) for p in [Path(__file__),Path(data.__file__),Path(entity_score.__file__)]}
    if splits_path:
        report['input_hashes'][str(splits_path)]=sha256(splits_path)
    report['output_hashes']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'report.json',report)
    (output/'_SUCCESS').write_text(sha256(output/'report.json')+'\n')
    print(json.dumps({k:report[k] for k in ['anchor_counts','best_iteration','tree_count','threshold']},indent=2))
    print(f"Holdout diagnostic macro F0.5: {reports['holdout']['macro_f05']:.6f}",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/a_pilot.json'))
    p.add_argument('--splits',type=Path,help='JSON anchor -> train/early_stop/calibration/holdout')
    a=p.parse_args()
    run(a.features,a.pilot,a.output,a.config,a.splits)
