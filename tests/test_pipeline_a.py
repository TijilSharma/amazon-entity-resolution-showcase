import math
import unittest
import numpy as np
from ber.evaluation.entity_score import score,decisions,choose_threshold
from ber.models.a_catboost.data import groups,make_splits,validate_splits

class ScorerTests(unittest.TestCase):
    def test_organizer_example(self):
        self.assertAlmostEqual(score({'q':['a','c']},{'q':['a','b','c']})['macro_f05'],5/7)

    def test_singletons_and_unretrieved_positive(self):
        actual={'a':[],'b':['x'],'c':['z']}
        result=score(actual,{'a':[],'b':['x']})
        self.assertAlmostEqual(result['macro_f05'],2/3)
        self.assertEqual(result['fn'],1)
        self.assertEqual(score({'q':[]},{'q':['wrong']})['macro_f05'],0)

    def test_macro_is_not_micro(self):
        self.assertAlmostEqual(score({'a':['1','2','3'],'b':['4']},
                                     {'a':['1','2','3'],'b':[]})['macro_f05'],.5)

    def test_threshold_sweep_matches_brute_force_including_ties(self):
        rng=np.random.default_rng(7)
        truth={'a':['x','missing'],'b':[],'c':['z']}
        pairs=[('a','x'),('a','y'),('b','x'),('b','y'),('c','z'),('c','y')]
        for _ in range(30):
            probs=rng.choice([0,.1,.5,.9,1.],size=len(pairs))
            chosen,_=choose_threshold(truth,pairs,probs)
            thresholds=[math.nextafter(float(max(probs)),math.inf)]+sorted(set(probs),reverse=True)
            brute=[(score(truth,decisions(truth,pairs,probs,t))['macro_f05'],t) for t in thresholds]
            expected=max(brute)
            self.assertAlmostEqual(chosen['macro_f05'],expected[0])
            self.assertEqual(chosen['threshold'],expected[1])

    def test_reject_all_and_empty_candidate_pool(self):
        chosen,_=choose_threshold({'q':[]},[('q','x')],[1.])
        self.assertGreater(chosen['threshold'],1.)
        chosen,_=choose_threshold({'q':['x']},[],[])
        self.assertEqual(chosen['macro_f05'],0)

    def test_invalid_scores_and_duplicate_pairs_rejected(self):
        for probs in ([float('nan')],[1.2]):
            with self.assertRaises(ValueError):
                choose_threshold({'q':[]},[('q','x')],probs)
        with self.assertRaises(ValueError):
            decisions(['q'],[('q','x'),('q','x')],[.1,.2],.5)

class GroupTests(unittest.TestCase):
    def fixture(self):
        truth={f'q{i}':[f't{i}'] for i in range(20)}
        records={q:{'name_normalized':q,'address_normalized':'12 road','country':'India'} for q in truth}
        truth['q1']=['t0']
        records['q3']=dict(records['q2'])
        return truth,records

    def test_shared_positives_and_exact_fingerprints_stay_together(self):
        truth,records=self.fixture()
        assignments,g=make_splits(truth,records,1)
        self.assertEqual(g['q0'],g['q1'])
        self.assertEqual(g['q2'],g['q3'])
        validate_splits(assignments,truth,g)
        self.assertEqual(assignments,make_splits(truth,records,1)[0])
        assignments['q1']=next(x for x in ['train','early_stop','calibration','holdout'] if x!=assignments['q0'])
        with self.assertRaises(ValueError):
            validate_splits(assignments,truth,g)

    def test_zero_candidate_anchor_remains_in_split(self):
        truth,records=self.fixture()
        truth['q0']=[]
        assignments,g=make_splits(truth,records,1)
        self.assertEqual(set(assignments),set(truth))
