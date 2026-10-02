import math
import unittest
import numpy as np
from ber.features.comparisons import compare_field, tokens, numbers, pair_features
from ber.features.build_features import aligned_cosines, validate_candidates

class FeatureTests(unittest.TestCase):
    def test_missing_is_not_agreement(self):
        f=compare_field('','','name')
        self.assertTrue(f['name_both_missing'])
        for key in ['exact','edit_similarity','token_jaccard','number_sets_equal']:
            self.assertTrue(math.isnan(f['name_'+key]))

    def test_accents_only_change_auxiliary_comparison(self):
        f=compare_field('café','cafe','name')
        self.assertEqual(f['name_exact'],0)
        self.assertEqual(f['name_accent_folded_exact'],1)

    def test_indic_marks_and_numeric_leading_zeros(self):
        self.assertEqual(tokens('शर्मा ट्रेडर्स'),{'शर्मा','ट्रेडर्स'})
        self.assertEqual(numbers('१२ 0012'),{'12','0012'})
        f=compare_field('12 Main Street','99 Main Street','address')
        self.assertEqual(f['address_numbers_disjoint'],1)
        self.assertEqual(f['address_number_sets_equal'],0)

    def test_subset_ratio_is_not_exact(self):
        f=compare_field('sharma','sharma traders','name')
        self.assertEqual(f['name_token_set_ratio'],1)
        self.assertEqual(f['name_exact'],0)
        self.assertEqual(f['name_token_jaccard'],.5)

    def test_embedding_alignment_and_missing_masks(self):
        q=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
        t=np.array([[0.,1.],[1.,0.],[0.,0.]],dtype=np.float32)
        result=aligned_cosines(q,t,np.array([1,0,0,1]),np.array([0,1,0,2]),
                               np.array([True,True]),np.array([True,True,False]))
        np.testing.assert_allclose(result[:3],[1,1,0])
        self.assertTrue(np.isnan(result[3]))

    def test_candidate_reference_and_duplicate_checks(self):
        c={'source1_entity_id':'q','candidate_id':'t','candidate_rank':1}
        validate_candidates([c],{'q':0},{'t':0})
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_candidates([c,c],{'q':0},{'t':0})
        with self.assertRaisesRegex(ValueError,'unknown'):
            validate_candidates([c],{'q':0},{})

    def test_no_labels_or_ids_enter_comparisons(self):
        r={'name_normalized':'abc','address_normalized':'12 road',
           'name_scripts':['Latin'],'address_scripts':['Latin'],'country':'India',
           'entity_id':'arbitrary','label':1}
        first=pair_features(r,r)
        other={**r,'entity_id':'different','label':0}
        self.assertEqual(first,pair_features(other,other))
        self.assertNotIn('label',first)
