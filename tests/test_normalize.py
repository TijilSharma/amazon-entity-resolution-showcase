"""Behavioral checks for identity evidence that normalization must preserve."""

import csv
import tempfile
import unicodedata
import unittest
from pathlib import Path

from ber.data.normalize import normalize_record, normalize_text, fold_latin_accents
from ber.data.preview_normalization import iter_preview


def record(name="Example LLP", address="AF-0684, Unit 11", country="France"):
    return dict(entity_id="S1-00001", business_name=name,
                business_address=address, country=country)


class NormalizationTests(unittest.TestCase):
    def test_raw_fields_and_input_are_preserved(self):
        source = record("  ÉCOLE  SARL ", "  5 bis Rue du Marché  ")
        original = source.copy()
        result = normalize_record(source)
        self.assertEqual(source, original)
        for key, value in original.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["business_name_raw"], source["business_name"])
        self.assertEqual(result["business_address_raw"], source["business_address"])

    def test_unicode_case_and_whitespace(self):
        result = normalize_record(record("  ＡＢＣ\tEngineering\nPvt Ltd  "))
        self.assertEqual(result["name_normalized"], "abc engineering pvt ltd")

    def test_latin_accents_only(self):
        self.assertEqual(fold_latin_accents("école du marché"), "ecole du marche")
        for text in ["ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்", "एसएस फ़ूड", "ಕರ್ನಾಟಕ", "తెలుగు"]:
            self.assertEqual(fold_latin_accents(text), unicodedata.normalize("NFC", text))
        mixed = normalize_record(record("École ராஜ்"))
        self.assertEqual(mixed["name_accent_folded"], "ecole ராஜ்")
        self.assertEqual(mixed["name_scripts"], ["Latin", "Tamil"])

    def test_numbers_punctuation_and_suffix_survive(self):
        source = record("A&B Pvt. Ltd / SARL", "AF-0684, 19 1/2, 6(29), Unit 11, 5 bis")
        result = normalize_record(source)
        self.assertEqual(result["name_normalized"], "a&b pvt. ltd / sarl")
        self.assertEqual(result["address_normalized"], "af-0684, 19 1/2, 6(29), unit 11, 5 bis")
        self.assertNotEqual(normalize_text("AF-0684"), normalize_text("AF-684"))

    def test_blank_and_placeholder_are_distinct(self):
        result = normalize_record(record(" \t", " NULL "))
        self.assertTrue(result["name_blank"])
        self.assertFalse(result["name_placeholder"])
        self.assertTrue(result["address_placeholder"])
        self.assertFalse(result["address_blank"])
        self.assertTrue(result["address_missing"])
        self.assertEqual(result["address_normalized"], "")
        self.assertEqual(result["business_address_raw"], " NULL ")

    def test_placeholder_words_inside_names_are_retained(self):
        result = normalize_record(record("Null & Sons", "None Such Road"))
        self.assertFalse(result["name_missing"])
        self.assertFalse(result["address_missing"])
        self.assertEqual(result["name_normalized"], "null & sons")

    def test_optional_null_cleanup_does_not_change_main_view(self):
        result = normalize_record(record(address="630 45ND TERRACE, null, KANSAS CITY, MO"))
        self.assertIn(", null,", result["address_normalized"])
        self.assertEqual(result["address_without_null_components"],
                         "630 45nd terrace, kansas city, mo")
        self.assertTrue(result["address_null_component_removed"])
        other = normalize_record(record(address="Nullarbor Road, null street"))
        self.assertFalse(other["address_null_component_removed"])

    def test_idempotence(self):
        for text in [" ＡＢＣ  ", "ÉCOLE", "İstanbul", "Straße", "ராஜ்", "फ़ूड", "\u0301A"]:
            self.assertEqual(normalize_text(normalize_text(text)), normalize_text(text))
            self.assertEqual(fold_latin_accents(fold_latin_accents(text)), fold_latin_accents(text))
        result = normalize_record(record("École ராஜ்", "19 1/2, null, Town"))
        self.assertEqual(normalize_record(result), result)

    def test_unknown_country_and_joiners_are_preserved(self):
        source = record("क्\u200dष", country="Unseen Country")
        result = normalize_record(source)
        self.assertEqual(result["country"], "Unseen Country")
        self.assertIn("\u200d", result["name_normalized"])

    def test_missing_or_nonstring_inputs_fail(self):
        with self.assertRaises(ValueError):
            normalize_record({})
        with self.assertRaises(TypeError):
            normalize_record(record(address=None))

    def test_preview_limit_order_and_file_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tsv"
            rows = [record("École"), record("ராஜ்"), record("Third")]
            for i, row in enumerate(rows):
                row["entity_id"] = f"S1-{i}"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            original = path.read_bytes()
            output = list(iter_preview(path, 2))
            self.assertEqual([row["entity_id"] for row in output], ["S1-0", "S1-1"])
            self.assertEqual(len(list(iter_preview(path, 10))), 3)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                list(iter_preview(path, 0))

    def test_preview_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tsv"
            path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tName\n")
            with self.assertRaises(ValueError):
                list(iter_preview(path, 1))


if __name__ == "__main__":
    unittest.main()
