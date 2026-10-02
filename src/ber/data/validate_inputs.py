"""Inspect one source TSV without modifying it or loading it all into memory.

Example, from the business_entity_resolution directory:
    python src/ber/data/validate_inputs.py /path/to/train_source1.tsv --prefix S1-

Blank names and addresses are reported, not rejected. Literal strings such as
"null" remain ordinary text. Duplicate IDs and ground truth are not checked yet.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def inspect_source(path: Path, expected_prefix: str) -> dict:
    """Return counts for one source file; fail on an unexpected header.

    total_records includes malformed records. Field-level counts and country
    counts include only structurally valid records. CSV records are not always
    physical lines: a quoted field may contain a newline.
    """
    total_records = 0
    malformed_records = 0
    empty_names = 0
    empty_addresses = 0
    invalid_id_prefixes = 0
    countries = Counter()
    malformed_examples = []

    with path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file, delimiter="\t", strict=True)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected header in {path}: {reader.fieldnames!r}. "
                f"Expected {EXPECTED_COLUMNS!r}."
            )

        for row in reader:
            total_records += 1

            # DictReader stores extra columns under the None key and fills
            # missing columns with None. An empty string is a valid field.
            if None in row or any(row[column] is None for column in EXPECTED_COLUMNS):
                malformed_records += 1
                if len(malformed_examples) < 5:
                    malformed_examples.append(
                        {
                            "record_number": total_records,
                            "ending_file_line": reader.line_num,
                        }
                    )
                continue

            # strip() is only used for the blank check; stored text is unchanged.
            if not row["business_name"].strip():
                empty_names += 1
            if not row["business_address"].strip():
                empty_addresses += 1
            if not row["entity_id"].startswith(expected_prefix):
                invalid_id_prefixes += 1

            # Keep every country label, including unfamiliar or empty labels.
            countries[row["country"]] += 1

    return {
        "path": str(path.resolve()),
        "expected_prefix": expected_prefix,
        "total_records": total_records,
        "structurally_valid_records": total_records - malformed_records,
        "malformed_records": malformed_records,
        "malformed_examples": malformed_examples,
        "empty_business_names": empty_names,
        "empty_business_addresses": empty_addresses,
        "invalid_id_prefixes": invalid_id_prefixes,
        "records_per_country": dict(sorted(countries.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to one source TSV")
    parser.add_argument("--prefix", required=True, choices=["S1-", "S2-", "S3-"])
    args = parser.parse_args()

    try:
        summary = inspect_source(args.path, args.prefix)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(status=1, message=f"Input inspection failed: {error}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Empty descriptive fields alone do not fail this structural check.
    return 1 if summary["malformed_records"] or summary["invalid_id_prefixes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
