import csv
import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from ber.data.build_normalized import sha256, write_json
from ber.data.build_split import build, signature, choose_split
from ber.data.normalize import normalize_record
from ber.data.verify_split import verify


class SplitTests(unittest.TestCase):
    def test_signature_preserves_indic_marks_and_handles_formatting(self):
        self.assertEqual(signature('India','traders ltd','12, main road'),
                         signature('India','ltd traders','road main 12'))
        self.assertNotEqual(signature('India','कला','12'), signature('India','कल','12'))
        self.assertIsNone(signature('India','traders',''))

    def test_linked_duplicate_groups_and_pilot_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); release=root/'release'; release.mkdir()
            val=next(f'S1-v{i}' for i in range(1000) if choose_split(f'S1-v{i}',42)=='validation')
            hold=next(f'S1-h{i}' for i in range(1000) if choose_split(f'S1-h{i}',42)=='holdout')
            def record(i,n,a):
                return normalize_record(dict(entity_id=i,business_name=n,business_address=a,country='India'))
            allrows={1:[record('S1-a','Alpha','1'),record('S1-b','Beta','2'),
                        record(val,'Gamma','3'),record(hold,'Delta','4')],
                     2:[record('S2-p','Trádérs Ltd','12 main'),record('S2-q','Ltd traders','main 12')],
                     3:[record('S3-x','Other','9')]}
            m={'status':'PASS','sources':{}}
            for s, rows in allrows.items():
                folder=release/'train'/f'source{s}';folder.mkdir(parents=True)
                file=folder/'part-00000.parquet'
                pq.write_table(pa.Table.from_pylist(rows),file)
                m['sources'][f'train_source{s}.tsv']={'rows':len(rows),'shards':[{'file':file.name,'sha256':sha256(file)}]}
            labels=root/'labels.tsv'
            with labels.open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','matched_entity_ids'])
                w.writerows([['S1-a','S2-p'],['S1-b','S2-q'],[val,''],[hold,'']])
            m['ground_truth']={'sha256':sha256(labels)}
            write_json(release/'manifest.json',m)
            (release/'_SUCCESS').write_text(sha256(release/'manifest.json'))
            pilot=root/'pilot';pilot.mkdir()
            write_json(pilot/'truth.json',{'S1-a':['S2-p']})
            write_json(pilot/'manifest.json',{'files':{'truth.json':sha256(pilot/'truth.json')}})
            out=root/'split'
            result=build(release,labels,pilot,out,42)
            self.assertEqual(result['counts'],{'train':2,'validation':1,'holdout':1})
            self.assertEqual(result['forced_train_anchors'],2)
            self.assertEqual(result['cross_partition_signature_collisions'],0)
            membership={r['entity_id']:r for r in pq.read_table(out/'record_membership.parquet').to_pylist()}
            self.assertEqual(membership['S2-p']['group_id'],membership['S2-q']['group_id'])
            self.assertEqual(membership['S2-q']['split'],'train')
            self.assertNotIn('S3-x',membership) # unlabeled distractor, not a false singleton label
            self.assertEqual((out/'_SUCCESS').read_text().strip(),sha256(out/'manifest.json'))
            self.assertEqual(result['owned_records'],6)
            self.assertEqual(verify(out,release)['status'],'PASS')
            with (out/'validation_ground_truth.tsv').open('a') as f:
                f.write('corruption')
            with self.assertRaisesRegex(ValueError,'file changed'):
                verify(out,release)
