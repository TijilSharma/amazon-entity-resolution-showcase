"""Print a bounded normalization preview as JSON lines; never write the input.

From the project root:
    PYTHONPATH=src python3 -m ber.data.preview_normalization FILE.tsv --limit 5

This previews the first records, not a representative random sample.
"""

import argparse
import csv
import json
import sys
from itertools import islice
from pathlib import Path

from .normalize import NORMALIZATION_VERSION, UNICODE_VERSION, normalize_record
from .validate_inputs import EXPECTED_COLUMNS


def iter_preview(path: Path, limit: int):
    """Yield at most limit normalized records in input order, bounded in memory."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected source header: {reader.fieldnames!r}")
        for row in islice(reader, limit):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed row ending at line {reader.line_num}")
            yield normalize_record(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    print(
        f"Preview only: {NORMALIZATION_VERSION}; Unicode {UNICODE_VERSION}; "
        f"Python {sys.version.split()[0]}", file=sys.stderr,
    )
    try:
        for row in iter_preview(args.path, args.limit):
            print(json.dumps(row, ensure_ascii=False))
    except (OSError, UnicodeError, csv.Error, ValueError, TypeError) as error:
        parser.exit(1, f"Preview failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
