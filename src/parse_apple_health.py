"""Safely summarize and extract selected records from an Apple Health export."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from lxml import etree


CSV_COLUMNS = [
    "type",
    "sourceName",
    "unit",
    "creationDate",
    "startDate",
    "endDate",
    "value",
]
DATE_COLUMNS = ["creationDate", "startDate", "endDate"]
APPLE_HEALTH_DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DEFAULT_BATCH_SIZE = 50_000

TARGET_RECORDS = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate.csv",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate.csv",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv.csv",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep_analysis.csv",
    "HKQuantityTypeIdentifierStepCount": "steps.csv",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy.csv",
    "HKQuantityTypeIdentifierAppleExerciseTime": "exercise_time.csv",
}


class AppleHealthParseError(RuntimeError):
    """Raised when an Apple Health export cannot be parsed safely."""


def validate_paths(input_path: Path, output_dir: Path) -> None:
    """Validate the input file and create the output directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input XML does not exist: {input_path}")
    if not input_path.is_file():
        raise AppleHealthParseError(f"Input path is not a file: {input_path}")
    if not os.access(input_path, os.R_OK):
        raise PermissionError(f"Input XML is not readable: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise AppleHealthParseError(
            f"Output path is not a directory: {output_dir}"
        )


def record_attributes(element: etree._Element) -> dict[str, str | None]:
    """Return only the approved attributes from one Record element."""
    return {column: element.get(column) for column in CSV_COLUMNS}


def records_to_dataframe(
    records: list[dict[str, str | None]],
) -> tuple[pd.DataFrame, Counter[str]]:
    """Convert a record batch to a DataFrame and normalize its dates to UTC."""
    frame = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    invalid_dates: Counter[str] = Counter()

    for column in DATE_COLUMNS:
        original = frame[column]
        converted = pd.to_datetime(
            original,
            format=APPLE_HEALTH_DATE_FORMAT,
            errors="coerce",
            utc=True,
        )
        invalid_dates[column] = int((original.notna() & converted.isna()).sum())
        frame[column] = converted

    return frame, invalid_dates


def write_record_batch(
    records: list[dict[str, str | None]],
    output_path: Path,
    write_header: bool,
) -> Counter[str]:
    """Append one normalized record batch to a CSV file."""
    frame, invalid_dates = records_to_dataframe(records)
    frame.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )
    return invalid_dates


def initialize_empty_csvs(output_dir: Path) -> None:
    """Create all expected extraction files with headers."""
    empty_frame = pd.DataFrame(columns=CSV_COLUMNS)
    for filename in TARGET_RECORDS.values():
        output_path = output_dir / filename
        if not output_path.exists():
            empty_frame.to_csv(output_path, index=False)


def parse_apple_health_export(
    input_path: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Stream an Apple Health XML export and write aggregate/extracted CSVs.

    Returns counters for all Record types, extracted rows by type, and invalid
    date values by date column. No individual health record is logged.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    validate_paths(input_path, output_dir)
    record_type_counts: Counter[str] = Counter()
    extracted_counts: Counter[str] = Counter()
    invalid_date_counts: Counter[str] = Counter()
    buffers: dict[str, list[dict[str, str | None]]] = {
        record_type: [] for record_type in TARGET_RECORDS
    }
    headers_written = {record_type: False for record_type in TARGET_RECORDS}

    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".apple_health_parse_"
    ) as temporary_directory:
        temporary_output = Path(temporary_directory)

        try:
            context: Iterable[tuple[str, etree._Element]] = etree.iterparse(
                str(input_path),
                events=("end",),
                tag="Record",
                load_dtd=False,
                no_network=True,
                resolve_entities=False,
                huge_tree=True,
            )

            for _, element in context:
                record_type = element.get("type")
                if record_type:
                    record_type_counts[record_type] += 1

                    if record_type in TARGET_RECORDS:
                        buffers[record_type].append(record_attributes(element))
                        extracted_counts[record_type] += 1

                        if len(buffers[record_type]) >= batch_size:
                            filename = TARGET_RECORDS[record_type]
                            invalid_date_counts.update(
                                write_record_batch(
                                    buffers[record_type],
                                    temporary_output / filename,
                                    write_header=not headers_written[record_type],
                                )
                            )
                            buffers[record_type].clear()
                            headers_written[record_type] = True

                element.clear()
                parent = element.getparent()
                if parent is not None:
                    while element.getprevious() is not None:
                        del parent[0]

        except etree.XMLSyntaxError as exc:
            raise AppleHealthParseError(
                f"Invalid or incomplete Apple Health XML: {exc}"
            ) from exc

        for record_type, records in buffers.items():
            if records:
                filename = TARGET_RECORDS[record_type]
                invalid_date_counts.update(
                    write_record_batch(
                        records,
                        temporary_output / filename,
                        write_header=not headers_written[record_type],
                    )
                )

        initialize_empty_csvs(temporary_output)

        summary = pd.DataFrame(
            sorted(record_type_counts.items()),
            columns=["record_type", "count"],
        )
        summary.to_csv(
            temporary_output / "record_type_summary.csv",
            index=False,
        )

        output_filenames = [
            "record_type_summary.csv",
            *TARGET_RECORDS.values(),
        ]
        for filename in output_filenames:
            os.replace(temporary_output / filename, output_dir / filename)

    return record_type_counts, extracted_counts, invalid_date_counts


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Apple Health Record types and extract approved metrics. "
            "The command prints counts only, never individual records."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the local Apple Health export.xml file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=argparse.SUPPRESS,
    )
    return parser


def print_summary(
    record_type_counts: Counter[str],
    extracted_counts: Counter[str],
    invalid_date_counts: Counter[str],
) -> None:
    """Print aggregate counts without exposing individual health records."""
    print("Apple Health XML parsing completed.")
    print(f"Record types found: {len(record_type_counts)}")
    print(f"Total Record elements: {sum(record_type_counts.values())}")
    print("Selected record counts:")
    for record_type, filename in TARGET_RECORDS.items():
        print(f"  {filename}: {extracted_counts[record_type]}")

    invalid_total = sum(invalid_date_counts.values())
    print(f"Invalid date fields converted to empty values: {invalid_total}")


def main() -> int:
    """Run the command-line parser."""
    args = build_argument_parser().parse_args()

    try:
        counts, extracted, invalid_dates = parse_apple_health_export(
            input_path=args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
        )
    except (AppleHealthParseError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(counts, extracted, invalid_dates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
