#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/pennylane_factorial_analysis.py

Purpose
-------
Validate and analyze the canonical 5 x 5 PennyLane factorial experiment
(five training partitions x five initialization seeds). Reports descriptive
statistics only; they do not establish quantum computational advantage.

Research design: partition seeds 42-46, initialization seeds 43-47, 12
training and 4 held-out examples, 200 optimization steps, learning rate 0.1,
ridge parameter 1.0. Historical CSVs do not encode every hyperparameter, so
the canonical manifest below names the five validated 200-step datasets.

Security
--------
Standard library only; no network access. Never modifies input CSV files or
source code and performs no Git operations. Output files are created with
exclusive mode ("x"), so an existing file is never overwritten. Dry-run
validates and prints statistics without writing any export or log file.

Usage
-----
    python labs/pennylane_factorial_analysis.py --help
    python labs/pennylane_factorial_analysis.py --dry-run
    python labs/pennylane_factorial_analysis.py
    python labs/pennylane_factorial_analysis.py \
        --output-dir results/pennylane/factorial_analysis

Log files
---------
logs/pennylane_factorial_analysis/analysis_<timestamp>.log (plain text,
DEBUG and above; none is written in --dry-run mode).

Exit Codes
----------
0   Validation (and export, unless --dry-run) completed successfully.
1   Missing or invalid input data, or an output operation failed.
2   Invalid command-line arguments (argparse).
130 Interrupted by user.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


# Resolve paths from this file so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "pennylane"

# Canonical manifest: one validated 200-step CSV per partition seed. Listing the
# files explicitly keeps stray or exploratory runs out of the statistics.
CANONICAL_FILES = {
    42: "disjoint_runs_20261015_135700_901540.csv",
    43: "disjoint_runs_20261015_143226_802847.csv",
    44: "disjoint_runs_20261015_194041_859006.csv",
    45: "disjoint_runs_20261015_203356_702327.csv",
    46: "disjoint_runs_20261015_204308_944440.csv",
}

# The factorial grid is fully determined by these two seed sets (5 x 5 = 25).
EXPECTED_SPLITS = set(CANONICAL_FILES)
EXPECTED_INITS = {43, 44, 45, 46, 47}

# Columns every canonical CSV must carry; baselines are optional (see below).
REQUIRED_COLUMNS = {
    "mode",
    "split_seed",
    "init_seed",
    "quantum_train_mse",
    "quantum_train_accuracy_pct",
    "quantum_test_mse",
    "quantum_test_accuracy_pct",
}

# Older CSVs predate the classical baselines, so these may be blank.
BASELINE_COLUMNS = (
    "majority_test_accuracy_pct",
    "linear_test_accuracy_pct",
    "polynomial_test_accuracy_pct",
    "nearest_neighbour_test_accuracy_pct",
)


class ColorFormatter(logging.Formatter):
    """Colorize terminal output without altering persistent log records."""

    COLORS = {
        logging.DEBUG: "\033[2m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[31;1m",
    }

    def __init__(self, *, enabled: bool):
        """Create the formatter; ``enabled`` switches ANSI colors on or off."""
        super().__init__(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self.enabled = enabled

    def format(self, record):
        """Return the formatted record, wrapped in its level color if enabled."""
        formatted = super().format(record)

        if not self.enabled:
            return formatted

        color = self.COLORS.get(record.levelno, "")

        return f"{color}{formatted}\033[0m" if color else formatted


def configure_logging(log_path: Path | None) -> logging.Logger:
    """Build the logger: colored console plus an optional plain-text file."""
    logger = logging.getLogger("pennylane.factorial")
    # The logger passes everything; each handler applies its own threshold.
    logger.setLevel(logging.DEBUG)
    # Clearing handlers avoids duplicate lines if this is called twice.
    logger.handlers.clear()
    # Do not bubble up to the root logger, which would print each line twice.
    logger.propagate = False

    terminal = logging.StreamHandler(sys.stdout)
    terminal.setLevel(logging.INFO)

    # Color only for real terminals; honor NO_COLOR and TERM=dumb so redirected
    # output and CI logs stay free of escape codes.
    use_color = (
        sys.stdout.isatty()
        and os.environ.get("TERM", "") != "dumb"
        and "NO_COLOR" not in os.environ
    )

    terminal.setFormatter(ColorFormatter(enabled=use_color))
    logger.addHandler(terminal)

    if log_path is not None:
        # Create the per-script log directory on first use.
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        # The file keeps DEBUG detail and never contains ANSI codes.
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            ColorFormatter(enabled=False)
        )

        logger.addHandler(file_handler)

    return logger


def parse_arguments():
    """Parse the command line (argparse exits with code 2 on bad input)."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze the canonical 25-configuration "
            "PennyLane factorial experiment."
        )
    )

    # Inputs are read-only; outputs always go to a separate, new directory.
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing canonical PennyLane CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "factorial_analysis",
        help="Directory for new analysis exports.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print statistics without exporting.",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in terminal output.",
    )

    return parser.parse_args()


def read_canonical_data(results_dir: Path):
    """Load and strictly validate the 25 canonical configurations.

    Returns (records keyed by (split_seed, init_seed), source file paths).
    Raises FileNotFoundError or ValueError on any structural or numeric fault.
    """
    records = {}
    source_files = {}

    for expected_split, filename in sorted(CANONICAL_FILES.items()):
        # Each file must exist, have the required columns and hold exactly one
        # row per initialization seed; anything else is a data-integrity fault.
        path = results_dir / filename

        # Fail fast with the exact missing path rather than a vague error later.
        if not path.is_file():
            raise FileNotFoundError(
                f"Canonical dataset missing: {path}"
            )

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)

            # utf-8-sig above strips a spreadsheet BOM that would corrupt the
            # first column name.
            if not reader.fieldnames:
                raise ValueError(f"Empty CSV: {path}")

            missing = REQUIRED_COLUMNS - set(reader.fieldnames)

            if missing:
                raise ValueError(
                    f"{path.name}: missing columns {sorted(missing)}"
                )

            rows = list(reader)

        # Five initialization seeds per partition: any other count is a
        # truncated or appended file.
        if len(rows) != 5:
            raise ValueError(
                f"{path.name}: expected 5 rows; found {len(rows)}"
            )

        found_inits = set()

        for row in rows:
            # Only initialization-mode runs belong to this factorial design.
            if row["mode"] != "initialization":
                raise ValueError(
                    f"{path.name}: unexpected mode {row['mode']}"
                )

            split = int(row["split_seed"])
            init = int(row["init_seed"])

            # The partition seed must match the file it is listed under.
            if split != expected_split:
                raise ValueError(
                    f"{path.name}: expected split "
                    f"{expected_split}; found {split}"
                )

            if init not in EXPECTED_INITS:
                raise ValueError(
                    f"{path.name}: unexpected init seed {init}"
                )

            # A repeated init seed would double-count one cell of the grid.
            if init in found_inits:
                raise ValueError(
                    f"{path.name}: duplicate init seed {init}"
                )

            found_inits.add(init)

            key = (split, init)

            # The same cell appearing in two files would bias the statistics.
            if key in records:
                raise ValueError(
                    f"Duplicate canonical configuration: {key}"
                )

            # Reject NaN/inf and out-of-range values before they reach the
            # statistics, where they would silently distort means.
            numeric_fields = (
                "quantum_train_mse",
                "quantum_train_accuracy_pct",
                "quantum_test_mse",
                "quantum_test_accuracy_pct",
            )

            # Keep the provenance (source file) with every record.
            record = {
                "split_seed": split,
                "init_seed": init,
                "source_file": filename,
            }

            for field in numeric_fields:
                value = float(row[field])

                if not math.isfinite(value):
                    raise ValueError(
                        f"{key}: nonfinite value in {field}"
                    )

                # Accuracies are percentages; MSE only needs to be non-negative.
                if field.endswith("_accuracy_pct"):
                    if not 0 <= value <= 100:
                        raise ValueError(
                            f"{key}: invalid accuracy {value}"
                        )
                elif value < 0:
                    raise ValueError(
                        f"{key}: negative MSE {value}"
                    )

                record[field] = value

            # Baselines are optional: blank means "not recorded", not zero.
            for field in BASELINE_COLUMNS:
                raw = row.get(field)

                if raw is None or raw == "":
                    record[field] = None
                    continue

                value = float(raw)

                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(
                        f"{key}: invalid baseline {field}"
                    )

                record[field] = value

            records[key] = record

        # With 5 unique, in-range seeds this proves full coverage per file.
        if found_inits != EXPECTED_INITS:
            raise ValueError(
                f"{path.name}: incomplete initialization coverage"
            )

        source_files[expected_split] = str(path)

    # Final completeness check: the full 5 x 5 grid, no gaps, no extras.
    expected_keys = {
        (split, init)
        for split in EXPECTED_SPLITS
        for init in EXPECTED_INITS
    }

    if set(records) != expected_keys:
        raise ValueError(
            "Canonical factorial coverage is incomplete"
        )

    return records, source_files


def descriptive_stats(values):
    """Return count, mean, sample SD (n-1), minimum and maximum of ``values``."""
    values = list(values)

    if not values:
        raise ValueError("Cannot summarize an empty dataset")

    return {
        "count": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def analyze(records):
    """Summarize all configurations overall, per split and per initialization."""
    ordered = [
        records[key]
        for key in sorted(records)
    ]

    metrics = (
        "quantum_train_accuracy_pct",
        "quantum_test_accuracy_pct",
        "quantum_train_mse",
        "quantum_test_mse",
    )

    # Pooled statistics across all 25 configurations.
    overall = {
        metric: descriptive_stats(
            row[metric] for row in ordered
        )
        for metric in metrics
    }

    # Marginal summaries: averaging over inits (per split) shows partition
    # sensitivity; averaging over splits (per init) shows seed sensitivity.
    by_split = {}
    by_init = {}

    for split in sorted(EXPECTED_SPLITS):
        subset = [
            records[(split, init)]
            for init in sorted(EXPECTED_INITS)
        ]

        by_split[str(split)] = {
            metric: descriptive_stats(
                row[metric] for row in subset
            )
            for metric in metrics
        }

    for init in sorted(EXPECTED_INITS):
        subset = [
            records[(split, init)]
            for split in sorted(EXPECTED_SPLITS)
        ]

        by_init[str(init)] = {
            metric: descriptive_stats(
                row[metric] for row in subset
            )
            for metric in metrics
        }

    # Configurations below 100% held-out accuracy are reported individually.
    failures = [
        row for row in ordered
        if row["quantum_test_accuracy_pct"] < 100
    ]

    # Baselines are summarized only over rows that actually recorded them.
    baseline_stats = {}

    for field in BASELINE_COLUMNS:
        values = [
            row[field]
            for row in ordered
            if row[field] is not None
        ]

        if values:
            baseline_stats[field] = descriptive_stats(values)

    return {
        "design": {
            "partition_seeds": sorted(EXPECTED_SPLITS),
            "initialization_seeds": sorted(EXPECTED_INITS),
            "configurations": len(ordered),
            "training_examples": 12,
            "held_out_examples": 4,
            "optimization_steps": 200,
            "learning_rate": 0.1,
            "ridge_parameter": 1.0,
            "hyperparameter_provenance": (
                "Validated execution logs and experiment manifest; "
                "not independently encoded in detailed CSV rows."
            ),
        },
        "overall": overall,
        "by_split": by_split,
        "by_initialization": by_init,
        "baselines": baseline_stats,
        "perfect_held_out_configurations": sum(
            row["quantum_test_accuracy_pct"] == 100
            for row in ordered
        ),
        "lower_performing_configurations": failures,
    }


def inspect_historical_files(results_dir: Path):
    """
    Report repeated seed combinations.

    These are occurrences, not necessarily equivalent experiments.
    """

    # Maps (split, init) -> files that contain it, to expose repeated runs.
    occurrences = defaultdict(list)

    for path in sorted(results_dir.glob("disjoint_runs_*.csv")):
        try:
            with path.open(
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                reader = csv.DictReader(handle)

                for row in reader:
                    key = (
                        int(row["split_seed"]),
                        int(row["init_seed"]),
                    )

                    if (
                        key[0] in EXPECTED_SPLITS
                        and key[1] in EXPECTED_INITS
                    ):
                        occurrences[key].append(path.name)

        except (KeyError, ValueError, UnicodeError, csv.Error):
            # Unrelated or malformed historical files are skipped on purpose;
            # this scan is informational and must never abort the analysis.
            continue

    return {
        f"{split}:{init}": filenames
        for (split, init), filenames in sorted(occurrences.items())
        if len(filenames) > 1
    }


def export_results(output_dir, records, report, duplicates):
    """
    Create timestamped files without overwriting existing artifacts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Microsecond timestamps make collisions between runs practically impossible.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    summary_path = output_dir / f"factorial_summary_{stamp}.json"
    data_path = output_dir / f"factorial_records_{stamp}.csv"
    duplicates_path = output_dir / f"historical_occurrences_{stamp}.json"

    # Check all three names first so a failure cannot leave a partial export.
    paths = (summary_path, data_path, duplicates_path)

    if any(path.exists() for path in paths):
        raise FileExistsError(
            "One or more output filenames already exist"
        )

    # Exclusive creation prevents accidental overwrite.
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    with data_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(next(iter(records.values())).keys()),
        )

        writer.writeheader()

        for key in sorted(records):
            writer.writerow(records[key])

    with duplicates_path.open("x", encoding="utf-8") as handle:
        json.dump(duplicates, handle, indent=2)
        handle.write("\n")

    return paths


def main():
    """Run validation and analysis; return the process exit code."""
    args = parse_arguments()

    # --no-color is applied through NO_COLOR, which configure_logging honors.
    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Dry-run writes nothing, including the log file.
    log_path = None if args.dry_run else (
        PROJECT_ROOT
        / "logs"
        / "pennylane_factorial_analysis"
        / f"analysis_{stamp}.log"
    )

    logger = configure_logging(log_path)

    try:
        logger.info("PennyLane factorial analysis started")

        records, sources = read_canonical_data(args.results_dir)

        logger.info(
            "Validated %d unique configurations",
            len(records),
        )

        # Analysis is pure computation; exports happen only after it succeeds.
        report = analyze(records)
        report["canonical_sources"] = sources

        duplicates = inspect_historical_files(args.results_dir)

        # Headline numbers go to the console and the log for quick review.
        test_stats = report["overall"]["quantum_test_accuracy_pct"]

        logger.info(
            "Mean held-out accuracy: %.2f%%",
            test_stats["mean"],
        )

        logger.info(
            "Held-out sample SD: %.2f percentage points",
            test_stats["sample_sd"],
        )

        logger.info(
            "Perfect held-out configurations: %d/%d",
            report["perfect_held_out_configurations"],
            len(records),
        )

        for row in report["lower_performing_configurations"]:
            logger.warning(
                "Split=%d init=%d train=%.1f%% test=%.1f%%",
                row["split_seed"],
                row["init_seed"],
                row["quantum_train_accuracy_pct"],
                row["quantum_test_accuracy_pct"],
            )

        logger.info(
            "Historical seed combinations with multiple "
            "occurrences: %d",
            len(duplicates),
        )

        if args.dry_run:
            logger.info(
                "DRY RUN PASS: No analysis exports created"
            )
            return 0

        paths = export_results(
            args.output_dir,
            records,
            report,
            duplicates,
        )

        for path in paths:
            logger.info("Created: %s", path)

        logger.info("PASS: Factorial analysis completed")

        return 0

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        csv.Error,
    ) as exc:
        logger.error("FAIL: %s", exc)
        return 1

    except KeyboardInterrupt:
        # Conventional shell exit status for SIGINT (128 + 2).
        logger.error("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
