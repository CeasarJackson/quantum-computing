#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/pennylane_repeated_eval.py

Purpose
-------
Run repeated, reproducible disjoint-input evaluations of labs/PennyLane.py.
Compare the quantum model with majority-class and ridge-regularized linear
classifiers on identical train/test splits. Save individual and summary CSVs.

Security
--------
Runs only the local classifier with the active Python interpreter. Does not
install dependencies, modify the classifier, or access the network. Dry-run
performs no model training and writes no CSV output.

Usage
-----
    python labs/pennylane_repeated_eval.py --dry-run
    python labs/pennylane_repeated_eval.py --seeds 42,43,44 --steps 5
    python labs/pennylane_repeated_eval.py --seeds 42,43,44,45,46 --steps 50
    python labs/pennylane_repeated_eval.py --mode initialization --fixed-split-seed 42 --seeds 43,44,45,46,47 --steps 50
    python labs/pennylane_repeated_eval.py --mode partition --fixed-init-seed 43 --seeds 42,43,44,45,46 --steps 50

Exit Codes
----------
0   All requested runs and CSV exports completed successfully.
1   A model run, dependency, or output operation failed.
2   Invalid command-line arguments.
130 Interrupted by user.
===============================================================================
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from applog import get_logger
from PennyLane import make_disjoint_dataset, majority_baseline

# Parse only explicit PASS metric lines, not intermediate training progress.
TRAIN_RE = re.compile(r"Training complete: MSE=([\d.]+), accuracy=([\d.]+)%")
TEST_RE = re.compile(r"Held-out test: MSE=([\d.]+), accuracy=([\d.]+)%")
SPLIT_RE = re.compile(
    r"Disjoint input split verified: (\d+) train, (\d+) test, overlap=0"
)


def seed_list(value):
    """Parse a nonempty, comma-separated sequence of distinct integer seeds."""
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    return seeds


def positive_int(value):
    """Require a positive integer."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments(argv=None):
    """Return the command-line configuration."""
    parser = argparse.ArgumentParser(
        description="Repeated disjoint PennyLane evaluation"
    )
    parser.add_argument("--seeds", type=seed_list, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--mode", choices=("paired", "initialization", "partition"), default="paired"
    )
    parser.add_argument("--fixed-split-seed", type=int, default=42)
    parser.add_argument("--fixed-init-seed", type=int, default=43)
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--train-unique", type=positive_int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def linear_baseline(train_x, train_y, test_x, test_y, strength):
    """Fit an intercept plus four linear bit coefficients using ridge regression."""
    # Convert encoded RX angles back to the binary features they represent.
    x_train = np.column_stack((np.ones(len(train_x)), train_x / np.pi))
    x_test = np.column_stack((np.ones(len(test_x)), test_x / np.pi))
    # The linear baseline cannot represent parity perfectly in raw bit space.
    # It is a more informative comparator than a constant majority predictor.
    # Do not penalize the intercept; regularize only the four bit weights.
    penalty = np.diag([0.0, strength, strength, strength, strength])
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    train_pred = np.where(x_train @ weights >= 0, 1.0, -1.0)
    test_pred = np.where(x_test @ weights >= 0, 1.0, -1.0)
    return 100 * np.mean(train_pred == train_y), 100 * np.mean(test_pred == test_y)


def run_quantum(script, args, split_seed, init_seed):
    """Execute one PennyLane training run with independent random seeds."""
    command = [
        sys.executable,
        str(script),
        "--split",
        "disjoint",
        "--train-unique",
        str(args.train_unique),
        "--steps",
        str(args.steps),
        "--learning-rate",
        str(args.learning_rate),
        "--split-seed",
        str(split_seed),
        "--init-seed",
        str(init_seed),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout + "\n" + result.stderr
    train, test, split = (
        TRAIN_RE.search(output),
        TEST_RE.search(output),
        SPLIT_RE.search(output),
    )
    if result.returncode or not all((train, test, split)):
        raise RuntimeError(
            f"Split {split_seed}, init {init_seed}: classifier failed or output changed:\n{output[-3000:]}"
        )
    if (int(split.group(1)), int(split.group(2))) != (
        args.train_unique,
        16 - args.train_unique,
    ):
        raise RuntimeError(
            f"Split {split_seed}, init {init_seed}: unexpected split sizes"
        )
    return tuple(
        map(float, (train.group(1), train.group(2), test.group(1), test.group(2)))
    )


def export_csv(path, fieldnames, rows):
    """Write CSV atomically, replacing only this run's timestamped destination."""
    temp = path.with_suffix(".tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    """Run each seed on identical splits and report cross-seed statistics."""
    args = arguments(argv)
    log = get_logger("pennylane_repeated_eval")
    if not 1 <= args.train_unique <= 15:
        log.fail("--train-unique must be in 1..15")
        return 1
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        log.fail("--learning-rate must be finite and positive")
        return 1
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        log.fail("--ridge must be finite and positive")
        return 1
    root = Path(__file__).resolve().parent.parent
    classifier = root / "labs" / "PennyLane.py"
    if not classifier.is_file():
        log.fail("Classifier not found: %s", classifier)
        return 1
    log.info(
        "Mode=%s; seeds=%s; steps=%d; training combinations=%d",
        args.mode,
        args.seeds,
        args.steps,
        args.train_unique,
    )
    if args.dry_run:
        log.ok("Dry-run: classifier found; no training or CSV exports performed")
        return 0

    # Each seed defines one deterministic partition of the 16 input states.
    # We compare all three methods on precisely the same partition.
    # The quantum model is retrained independently for each partition.
    # This avoids interpreting one favorable four-example test as decisive.
    rows = []
    for seed in args.seeds:
        # Vary one randomness source at a time for controlled comparisons.
        if args.mode == "initialization":
            split_seed, init_seed = args.fixed_split_seed, seed
        elif args.mode == "partition":
            split_seed, init_seed = seed, args.fixed_init_seed
        else:
            split_seed, init_seed = seed, seed + 1
        # The existing classifier owns the split implementation; reuse it.
        train_x, train_y, test_x, test_y = make_disjoint_dataset(
            args.train_unique, split_seed
        )
        train_keys = {tuple(row) for row in train_x}
        test_keys = {tuple(row) for row in test_x}
        if train_keys & test_keys:
            raise RuntimeError(f"Seed {seed}: train/test leakage")
        _, majority_train, majority_test = majority_baseline(train_y, test_y)
        linear_train, linear_test = linear_baseline(
            train_x, train_y, test_x, test_y, args.ridge
        )
        quantum_train_mse, quantum_train_acc, quantum_test_mse, quantum_test_acc = (
            run_quantum(classifier, args, split_seed, init_seed)
        )
        rows.append(
            {
                "seed": seed,
                "mode": args.mode,
                "split_seed": split_seed,
                "init_seed": init_seed,
                "quantum_train_mse": quantum_train_mse,
                "quantum_train_accuracy_pct": quantum_train_acc,
                "quantum_test_mse": quantum_test_mse,
                "quantum_test_accuracy_pct": quantum_test_acc,
                "majority_train_accuracy_pct": round(100 * majority_train, 6),
                "majority_test_accuracy_pct": round(100 * majority_test, 6),
                "linear_train_accuracy_pct": round(float(linear_train), 6),
                "linear_test_accuracy_pct": round(float(linear_test), 6),
            }
        )
        log.ok(
            "Split=%d init=%d: quantum=%.1f%%, majority=%.1f%%, ridge-linear=%.1f%% (test)",
            split_seed,
            init_seed,
            quantum_test_acc,
            100 * majority_test,
            linear_test,
        )

    # Report sample standard deviation when there are at least two seeds.
    # Aggregate accuracy across seeds; report spread, not just the mean.
    # Sample standard deviation is undefined for a single run.
    # Each split still contains only a handful of unseen configurations.
    metrics = [
        name
        for name in rows[0]
        if name not in ("seed", "mode", "split_seed", "init_seed")
    ]
    summary = []
    for name in metrics:
        values = np.array([row[name] for row in rows], dtype=float)
        summary.append(
            {
                "metric": name,
                "count": len(values),
                "mean": round(float(np.mean(values)), 6),
                "std_sample": (
                    round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else "NA"
                ),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
            }
        )
    # Keep experiment results separate from source and package pins.
    # Timestamped CSV files make runs independently inspectable.
    out_dir = root / "results" / "pennylane"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    detail_path = out_dir / f"disjoint_runs_{stamp}.csv"
    summary_path = out_dir / f"disjoint_summary_{stamp}.csv"
    export_csv(detail_path, list(rows[0]), rows)
    export_csv(summary_path, list(summary[0]), summary)
    for name in (
        "quantum_test_accuracy_pct",
        "majority_test_accuracy_pct",
        "linear_test_accuracy_pct",
    ):
        entry = next(item for item in summary if item["metric"] == name)
        log.info(
            "%s: mean=%.2f%%, sample SD=%s", name, entry["mean"], entry["std_sample"]
        )
    log.ok("Detailed results: %s", detail_path)
    log.ok("Summary results: %s", summary_path)
    log.info("These small parity splits do not demonstrate quantum advantage")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        get_logger("pennylane_repeated_eval").fail("Evaluation failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
