#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/pennylane_phase2_robustness.py

Purpose
-------
PennyLane Phase II: a partition x initialization robustness study. It reuses
the existing four-qubit parity classifier (labs/PennyLane.py) and the
classical baselines in labs/pennylane_repeated_eval.py without modifying
Phase I source or results, so Phase II numbers are directly comparable.

Every split seed must yield a valid, unique train/test partition of the 16
four-bit inputs; the full grid is validated before any training starts.

Scientific limitations: partitions may share held-out states despite
distinct signatures. The bootstrap is descriptive over the selected
partitions, not a population-generalization confidence interval. The
polynomial baseline's held-out score uses half credit for near-zero
predictions.

Security
--------
Handles no secrets and touches no network. Dry-run creates no files,
directories or logs. Real runs create a unique
results/pennylane/phase2/<run-id>/ directory with exclusive mkdir, and every
result file is exclusively created, so completed runs survive an interrupted
experiment and Phase I artifacts are never overwritten.

Usage
-----
    python labs/pennylane_phase2_robustness.py --dry-run
    python labs/pennylane_phase2_robustness.py --split-seeds 100 \\
        --init-seeds 200,201 --steps 5
    python labs/pennylane_phase2_robustness.py --split-seeds 100-109 \\
        --init-seeds 200-204 --steps 200

Prerequisites: Python 3.10+, NumPy, PennyLane, and the repository files
labs/PennyLane.py and labs/pennylane_repeated_eval.py.

Log files
---------
results/pennylane/phase2/<run-id>/execution.log (real runs only).

Exit Codes
----------
0   All configurations and summary exports completed (or dry-run passed).
1   Validation, training, or output failure.
2   Invalid command-line arguments.
130 Interrupted by user.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Make labs/ importable when run from the repo root, so Phase I code is
# reused verbatim instead of copied (copies would silently drift).
ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "labs"
if str(LABS) not in sys.path:
    sys.path.insert(0, str(LABS))

import numpy as np
from PennyLane import majority_baseline, make_disjoint_dataset
from pennylane_repeated_eval import (
    linear_baseline,
    nearest_neighbour_baseline,
    polynomial_baseline,
    run_quantum,
)

# Every per-configuration row must carry exactly these metrics; summarize()
# and validate_metric() both iterate over this tuple.
METRICS = (
    "quantum_train_mse",
    "quantum_train_accuracy_pct",
    "quantum_test_mse",
    "quantum_test_accuracy_pct",
    "majority_train_accuracy_pct",
    "majority_test_accuracy_pct",
    "linear_train_accuracy_pct",
    "linear_test_accuracy_pct",
    "polynomial_train_accuracy_pct",
    "polynomial_test_abstention_adjusted_pct",
    "nearest_neighbour_train_accuracy_pct",
    "nearest_neighbour_test_accuracy_pct",
)


def seed_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ranges such as 100-109."""
    values = []
    for part in value.split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match:
            lo, hi = map(int, match.groups())
            # Cap range width so a typo like 1-1000000 cannot queue a
            # multi-day run.
            if hi < lo or hi - lo > 9999:
                raise argparse.ArgumentTypeError("invalid or oversized seed range")
            values.extend(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", part):
            values.append(int(part))
        else:
            raise argparse.ArgumentTypeError(f"invalid seed entry: {part!r}")
    # Duplicate seeds would double-count a configuration in the statistics.
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("seed list must be nonempty and unique")
    if len(values) > 10000:
        raise argparse.ArgumentTypeError("too many seeds")
    return tuple(values)


def positive_int(value: str) -> int:
    """argparse type: an integer strictly greater than zero."""
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_finite(value: str) -> float:
    """argparse type: a finite float strictly greater than zero."""
    result = float(value)
    # float() accepts "nan" and "inf"; reject them explicitly.
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def arguments(argv=None):
    """Parse command-line options; defaults reproduce the Phase II design."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split-seeds", type=seed_spec, default=tuple(range(100, 110)))
    parser.add_argument("--init-seeds", type=seed_spec, default=tuple(range(200, 205)))
    parser.add_argument("--train-unique", type=positive_int, default=12)
    parser.add_argument("--steps", type=positive_int, default=200)
    parser.add_argument("--learning-rate", type=positive_finite, default=0.1)
    parser.add_argument("--ridge", type=positive_finite, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=positive_int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261015)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all partitions; create no files or logs",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    """Hash a file in 1 MiB chunks so large files never load fully in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits(angles):
    """Decode angle-encoded samples (0 or pi per qubit) back to 4-bit tuples."""
    # Round rather than truncate: encoded angles carry float error near pi.
    values = np.rint(np.asarray(angles) / np.pi).astype(int)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected four-bit encoded samples")
    return tuple(tuple(map(int, row)) for row in values)


def partitions_for(args):
    """Validate every split seed's partition and return the run manifest.

    Each split must be disjoint, cover all 16 inputs, carry correct parity
    labels, and hold out a set of inputs no other seed already produced.
    """
    if not 1 <= args.train_unique <= 15:
        raise ValueError("--train-unique must be in 1..15")
    if len(args.split_seeds) * len(args.init_seeds) > 10000:
        raise ValueError("at most 10000 configurations are supported")
    signatures = {}
    manifest = []
    # All 16 four-bit inputs, most-significant bit first, matching bits().
    universe = {tuple((n >> shift) & 1 for shift in (3, 2, 1, 0)) for n in range(16)}
    for seed in args.split_seeds:
        tx, ty, vx, vy = make_disjoint_dataset(args.train_unique, seed)
        train, test = bits(tx), bits(vx)
        train_set, test_set = set(train), set(test)
        if (
            len(train_set) != args.train_unique
            or len(test_set) != 16 - args.train_unique
            or train_set & test_set
            or train_set | test_set != universe
        ):
            raise ValueError(f"split {seed}: overlap, duplicate or missing inputs")
        # Recompute parity independently so a dataset bug cannot pass silently.
        if any(
            float(y) != (1.0 if sum(x) % 2 == 0 else -1.0)
            for x, y in list(zip(train, ty)) + list(zip(test, vy))
        ):
            raise ValueError(f"split {seed}: parity target mismatch")
        # Two seeds yielding the same held-out set would be pseudo-replicates.
        signature = tuple(sorted(test_set))
        if signature in signatures:
            raise ValueError(
                f"duplicate partition: seeds {signatures[signature]} and {seed}"
            )
        signatures[signature] = seed
        manifest.append(
            {
                "split_seed": seed,
                "train_bits": ["".join(map(str, x)) for x in sorted(train_set)],
                "test_bits": ["".join(map(str, x)) for x in signature],
                "train_count": len(train),
                "test_count": len(test),
            }
        )
    return manifest


class ConsoleFormatter(logging.Formatter):
    """Severity-colored console formatter; plain when colored is False."""

    def __init__(self, colored):
        """Use the shared timestamp layout; colored toggles ANSI output."""
        super().__init__("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        self.colored = colored

    def format(self, record):
        """Format the record, wrapping it in its severity color if enabled."""
        message = super().format(record)
        if not self.colored:
            return message
        color = {
            "INFO": "\033[36m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
        }.get(record.levelname, "")
        return f"{color}{message}\033[0m" if color else message


def logger_for(log_path=None):
    """Return the Phase II logger: console always, plain file if log_path.

    Deliberately not applog: applog always writes under logs/, which would
    break the guarantee that --dry-run creates no files.
    """
    logger = logging.getLogger("pennylane_phase2")
    # Clearing handlers lets main() rebuild the logger once the run directory
    # exists without duplicating console output.
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler(sys.stdout)
    colored = sys.stdout.isatty() and not os.getenv("NO_COLOR")
    console.setFormatter(ConsoleFormatter(colored))
    logger.addHandler(console)
    if log_path is not None:
        # mode="x": never append to or clobber an earlier run's log.
        handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        handler.setFormatter(ConsoleFormatter(False))
        logger.addHandler(handler)
    return logger


def exclusive_json(path, data):
    """Create path exclusively and write data as JSON, fsynced to disk."""
    # allow_nan=False: NaN/Infinity are not valid JSON and would break readers.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_csv(path, records):
    """Create path exclusively and write records as CSV, fsynced to disk."""
    if not records:
        raise ValueError("cannot export empty records")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())


def validate_metric(value, name):
    """Return value as a float, rejecting non-finite or out-of-range metrics."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite metric: {name}")
    # The suffix encodes the metric's valid range, so check by name.
    if name.endswith("_pct") and not 0 <= value <= 100:
        raise ValueError(f"invalid accuracy metric: {name}={value}")
    if name.endswith("_mse") and value < 0:
        raise ValueError(f"invalid MSE metric: {name}={value}")
    return value


def classical_scores(split_seed, args):
    """Score all classical baselines on one split, as percentages."""
    tx, ty, vx, vy = make_disjoint_dataset(args.train_unique, split_seed)
    # majority_baseline returns fractions; the other baselines return percent.
    _, mt, mv = majority_baseline(ty, vy)
    lt, lv = linear_baseline(tx, ty, vx, vy, args.ridge)
    pt, pv = polynomial_baseline(tx, ty, vx, vy, args.ridge)
    nt, nv = nearest_neighbour_baseline(tx, ty, vx, vy)
    return {
        "majority_train_accuracy_pct": 100 * mt,
        "majority_test_accuracy_pct": 100 * mv,
        "linear_train_accuracy_pct": lt,
        "linear_test_accuracy_pct": lv,
        "polynomial_train_accuracy_pct": pt,
        "polynomial_test_abstention_adjusted_pct": pv,
        "nearest_neighbour_train_accuracy_pct": nt,
        "nearest_neighbour_test_accuracy_pct": nv,
    }


def summarize(records, args):
    """Aggregate per-configuration rows into partition rows and a summary.

    Initializations are averaged within each partition first, so each
    partition counts once regardless of how many init seeds it ran.
    """
    grouped = {}
    for row in records:
        grouped.setdefault(row["split_seed"], []).append(row)
    partition_rows = []
    for seed in args.split_seeds:
        group = grouped[seed]
        if len(group) != len(args.init_seeds):
            raise ValueError(f"incomplete initialization grid for split {seed}")
        partition_rows.append(
            {
                "split_seed": seed,
                **{
                    f"{metric}_mean": statistics.mean(row[metric] for row in group)
                    for metric in METRICS
                },
                # stdev needs two points; None marks a single-init run.
                "quantum_test_accuracy_pct_sd_init": (
                    statistics.stdev(row["quantum_test_accuracy_pct"] for row in group)
                    if len(group) > 1
                    else None
                ),
            }
        )
    metrics = {}
    for metric in METRICS:
        values = [row[f"{metric}_mean"] for row in partition_rows]
        metrics[metric] = {
            "partition_mean": statistics.mean(values),
            "partition_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min_partition_mean": min(values),
            "max_partition_mean": max(values),
        }
    # Resample partition-level means as clusters, keeping all initializations
    # together. This describes the chosen seed-based partition collection only.
    rng = np.random.default_rng(args.bootstrap_seed)
    vals = np.array([row["quantum_test_accuracy_pct_mean"] for row in partition_rows])
    sampled = rng.integers(0, len(vals), size=(args.bootstrap_replicates, len(vals)))
    bootstrap = np.mean(vals[sampled], axis=1)
    return partition_rows, {
        "configuration_count": len(records),
        "unique_partition_count": len(partition_rows),
        "initializations_per_partition": len(args.init_seeds),
        "metrics": metrics,
        "quantum_test_accuracy_partition_bootstrap_percentile_interval_95": [
            float(x) for x in np.percentile(bootstrap, [2.5, 97.5])
        ],
        "bootstrap_interpretation": (
            "Descriptive cluster bootstrap over selected, overlapping finite-space "
            "partitions; NOT a confidence interval for new-population generalization."
        ),
        "polynomial_test_scoring": (
            "Half credit for abs(score) < 1e-9 (abstention-adjusted)"
        ),
        # Limitations travel with the numbers so summary.json cannot be quoted
        # without its caveats.
        "limitations": [
            (
                "Only 16 possible inputs; each test partition has four examples "
                "at train_unique=12."
            ),
            (
                "Unique partitions can share held-out examples and are not "
                "population-independent."
            ),
            (
                "Initialization repetitions within a partition are correlated "
                "by shared data."
            ),
            ("No claim of quantum advantage or out-of-distribution " "generalization."),
        ],
    }


def main(argv=None):
    """Validate the grid, train every configuration, and export results."""
    args = arguments(argv)
    # Console-only until the run directory exists; see logger_for().
    log = logger_for()
    try:
        source_files = (LABS / "PennyLane.py", LABS / "pennylane_repeated_eval.py")
        for source in source_files:
            if not source.is_file():
                raise FileNotFoundError(source)
        manifest = partitions_for(args)
        total = len(manifest) * len(args.init_seeds)
        log.info(
            "Validated %d unique partitions × %d initialization seeds = %d runs",
            len(manifest),
            len(args.init_seeds),
            total,
        )
        if args.dry_run:
            for row in manifest:
                log.info(
                    "Split %d: held out %s",
                    row["split_seed"],
                    ",".join(row["test_bits"]),
                )
            log.info("DRY RUN PASS: no training, files, directories, or logs created")
            return 0
        base = ROOT / "results" / "pennylane" / "phase2"
        base.mkdir(parents=True, exist_ok=True)
        # UTC microseconds plus PID keeps run IDs unique even for two runs
        # started in the same instant; exist_ok=False enforces it.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_id = f"{stamp}_{os.getpid()}"
        run_dir = base / run_id
        run_dir.mkdir(exist_ok=False)
        log = logger_for(run_dir / "execution.log")
        # Record everything needed to reproduce the run, including hashes of
        # the Phase I sources this study depends on.
        config = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "split_seeds": list(args.split_seeds),
            "init_seeds": list(args.init_seeds),
            "train_unique": args.train_unique,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "ridge": args.ridge,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "source_sha256": {
                str(p.relative_to(ROOT)): sha256(p) for p in source_files
            },
        }
        # The version lookup is provenance only; never fail a run over it.
        try:
            from importlib.metadata import version

            config["pennylane"] = version("pennylane")
        except Exception:
            config["pennylane"] = "unknown"
        exclusive_json(run_dir / "configuration.json", config)
        exclusive_json(run_dir / "partition_manifest.json", manifest)
        log.info("Run directory: %s", run_dir)
        log.info(
            "Starting %d configurations; results checkpointed per configuration",
            total,
        )
        records = []
        for partition in manifest:
            seed = partition["split_seed"]
            # Classical baselines are deterministic per split; compute once.
            baseline = classical_scores(seed, args)
            for init in args.init_seeds:
                # run_quantum is the existing verified subprocess interface.
                qmse_train, qacc_train, qmse_test, qacc_test = run_quantum(
                    LABS / "PennyLane.py", args, seed, init
                )
                row = {
                    "split_seed": seed,
                    "init_seed": init,
                    "quantum_train_mse": qmse_train,
                    "quantum_train_accuracy_pct": qacc_train,
                    "quantum_test_mse": qmse_test,
                    "quantum_test_accuracy_pct": qacc_test,
                    **baseline,
                }
                # Validate before checkpointing so a bad value never reaches
                # disk looking like a completed configuration.
                for metric in METRICS:
                    row[metric] = validate_metric(row[metric], metric)
                # Checkpoint before moving on so an interruption loses at
                # most the configuration in progress.
                exclusive_json(run_dir / f"split_{seed}_init_{init}.json", row)
                records.append(row)
                log.info(
                    "PASS split=%d init=%d quantum test=%.2f%% (%d/%d)",
                    seed,
                    init,
                    qacc_test,
                    len(records),
                    total,
                )
        # Summaries are written only after every configuration succeeded, so
        # their presence certifies a complete grid.
        partition_rows, summary = summarize(records, args)
        exclusive_csv(run_dir / "detailed_results.csv", records)
        exclusive_csv(run_dir / "partition_statistics.csv", partition_rows)
        exclusive_json(run_dir / "summary.json", summary)
        log.info(
            "PASS complete: mean quantum test accuracy %.2f%%; interval %s",
            summary["metrics"]["quantum_test_accuracy_pct"]["partition_mean"],
            summary["quantum_test_accuracy_partition_bootstrap_percentile_interval_95"],
        )
        log.info("Verified %d result JSON files and all summary exports", len(records))
        return 0
    except Exception as exc:
        log.error("FAIL: %s", exc)
        log.error("%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
