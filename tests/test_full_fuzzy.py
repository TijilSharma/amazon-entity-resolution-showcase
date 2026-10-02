import unittest

from ber.retrieval.full_fuzzy import fuzzy_score, keys


class FullFuzzyTests(unittest.TestCase):
    def test_prefix_and_unicode_numeric_keys_are_retained(self):
        values = keys({"name_normalized": "sharma corporation", "address_normalized": "१२ market road"})
        self.assertIn("word:sharma", values)
        self.assertIn("prefix:corp", values)
        self.assertIn("number:12", values)

    def test_name_similarity_survives_abbreviation(self):
        query = {"name_normalized": "abc corporation", "address_normalized": "12 market road"}
        target = {"name_normalized": "abc corp", "address_normalized": "market rd 12"}
        self.assertGreater(fuzzy_score(query, target), .7)

    def test_missing_address_does_not_create_address_agreement(self):
        query = {"name_normalized": "sharma traders", "address_normalized": ""}
        target = {"name_normalized": "sharma traders", "address_normalized": "different address"}
        self.assertEqual(fuzzy_score(query, target), 1.0)
