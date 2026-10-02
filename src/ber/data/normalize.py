"""Pure, conservative record normalization. No file I/O or external data.

Original fields remain unchanged. Derived fields are comparison views, never
replacements for the original evidence. Unicode behavior is versioned because
Python releases may bundle different Unicode databases.
"""

import unicodedata
from collections.abc import Mapping
from functools import lru_cache


NORMALIZATION_VERSION = "conservative-v1"
UNICODE_VERSION = unicodedata.unidata_version
PLACEHOLDERS = frozenset({"null", "none", "nan", "n/a", "na"})
SOURCE_FIELDS = ("entity_id", "business_name", "business_address", "country")
SCRIPT_NAMES = (
    "LATIN", "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
    "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "GREEK",
    "CYRILLIC", "ARABIC", "HEBREW", "THAI", "HIRAGANA", "KATAKANA",
    "HANGUL",
)


def normalize_text(text: str) -> str:
    """Normalize compatibility forms, case and whitespace; retain punctuation.

    The final NFKC pass restores normalization after casefolding. Meaningful
    marks, digits, joiners and punctuation are not explicitly stripped.
    """
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFKC", text.casefold())
    return " ".join(text.split())


def fold_latin_accents(text: str) -> str:
    """Remove combining marks attached to Latin bases only, in a separate view.

    This is not transliteration: letters such as ø and ł remain unchanged.
    Marks belonging to non-Latin bases survive. NFC recomposes retained text.
    """
    if text.isascii():
        return text
    output = []
    latin_base = False
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.category(char).startswith("M"):
            if not latin_base:
                output.append(char)
        else:
            latin_base = "LATIN" in unicodedata.name(char, "")
            output.append(char)
    return unicodedata.normalize("NFC", "".join(output))


@lru_cache(maxsize=8192)
def _letter_script(char: str) -> str | None:
    if not unicodedata.category(char).startswith("L"):
        return None
    name = unicodedata.name(char, "")
    if "CJK" in name or "IDEOGRAPH" in name:
        return "Han"
    for script in SCRIPT_NAMES:
        if script in name:
            return script.title()
    return "Other"


def detect_scripts(text: str) -> list[str]:
    """Return sorted, heuristic letter-script indicators, not languages.

    Digits, punctuation and marks do not establish a script on their own.
    Unrecognized letters are labelled Other rather than silently discarded.
    """
    if text.isascii():
        return ["Latin"] if any(char.isalpha() for char in text) else []
    return sorted({script for char in text if (script := _letter_script(char))})


def address_without_null_components(text: str) -> tuple[str, bool]:
    """Remove only whole comma-separated 'null' components in an extra view.

    Do not remove substrings (e.g. Nullarbor) or infer missing address details.
    Other placeholder words inside an address are retained.
    """
    components = text.split(",")
    kept = [component for component in components if component.strip() != "null"]
    removed = len(kept) != len(components)
    return (normalize_text(",".join(kept)) if removed else text), removed


def normalize_record(record: Mapping[str, str]) -> dict:
    """Return original fields plus deterministic derived views and flags.

    Missingness includes blanks and exact whole-field placeholders. Raw fields
    preserve the distinction even when both produce an empty normalized view.
    Country and entity_id are preserved byte-for-byte at the string level.
    No input mutation, suffix stripping, translation or spelling correction.
    """
    for field in SOURCE_FIELDS:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(record[field], str):
            raise TypeError(f"{field} must be a string; got {type(record[field]).__name__}")

    result = dict(record)
    result["normalization_version"] = NORMALIZATION_VERSION
    result["unicode_version"] = UNICODE_VERSION

    for prefix, field in (("name", "business_name"), ("address", "business_address")):
        raw = record[field]
        normalized = normalize_text(raw)
        blank = normalized == ""
        placeholder = normalized in PLACEHOLDERS
        result[f"{field}_raw"] = raw
        result[f"{prefix}_blank"] = blank
        result[f"{prefix}_placeholder"] = placeholder
        result[f"{prefix}_missing"] = blank or placeholder
        result[f"{prefix}_scripts"] = detect_scripts(raw)
        # The raw value and placeholder flag retain anything suppressed here.
        view = "" if blank or placeholder else normalized
        result[f"{prefix}_normalized"] = view
        result[f"{prefix}_accent_folded"] = fold_latin_accents(view)

    cleaned, removed = address_without_null_components(result["address_normalized"])
    result["address_without_null_components"] = cleaned
    result["address_null_component_removed"] = removed
    return result
