import unittest
from ber.retrieval.run_pilot import evaluate, merge_ranked, lexical_search, text_view


class RetrievalTests(unittest.TestCase):
    def test_union_retains_complementary_routes_and_deduplicates(self):
        self.assertEqual(merge_ranked([['a','b','c'], ['a','d','e']], 4), ['a','b','d','c'])
        self.assertEqual(merge_ranked([[], ['x']], 20), ['x'])
        self.assertEqual(merge_ranked([], 20), [])

    def test_metrics_keep_singletons_out_of_positive_recall(self):
        result = evaluate({'q1':['a','wrong'], 'q2':['c'], 'q3':['wrong']},
                          {'q1':['a','b'], 'q2':['c'], 'q3':[]})
        self.assertAlmostEqual(result['micro_recall'], 2/3)
        self.assertEqual(result['macro_recall_nonsingletons'], .75)
        self.assertEqual(result['complete_group_recall_nonsingletons'], .5)
        self.assertEqual(result['mean_singleton_candidates'], 1)

    def test_lexical_keeps_near_spelling_and_empty_queries_empty(self):
        def row(i, text): return {'entity_id':i, 'name_normalized':text}
        fit = [row('fit1','sharma traders'), row('fit2','sunny bakery')]
        queries = [row('q','sharma trader'), row('empty','')]
        targets = [row('a','sharma traders'), row('b','sunny bakery')]
        found, info = lexical_search(queries, targets, fit, 'name_normalized', 'char_wb', (3,5), 2)
        self.assertEqual(found[0][0][0], 0)
        self.assertEqual(found[1], [])
        self.assertEqual(info['queries_with_zero_features'], 1)

    def test_missing_views_do_not_invent_content(self):
        self.assertEqual(text_view({'name_normalized':'', 'address_normalized':'12 street'}, True), '12 street')
        self.assertEqual(text_view({'name_normalized':'', 'address_normalized':''}, True), '')
