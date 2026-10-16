#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/run_all.py

Purpose
-------
Runs the whole course in one command and prints a single summary table: the
code-standards audit, every lab (lab*.py) and every optional exercise
(ex[0-9]*.py). Each script runs in its own process with the same Python that
runs this file, so activate the virtual environment first. Labs 11, 15 and 17
run in their default DRY-RUN mode: nothing here contacts IBM or uses a key.

For each script the table shows PASS/FAIL, the elapsed time and the number of
checks it reported ("All N checks passed"), so a regression anywhere in the
course shows up immediately.

Security
--------
Handles no secrets and makes no network calls. Child processes inherit the
environment, but none of the scripts is started in live mode.

Usage
-----
    python3 labs/run_all.py                  # everything
    python3 labs/run_all.py --only lab1      # names containing "lab1"
    python3 labs/run_all.py --skip-slow      # skip scripts above --slow secs
    python3 labs/run_all.py --timeout 600

Log files
---------
logs/run_all/run_all_<YYYYmmdd_HHMMSS>.log (summary) and, per script, its own
log under logs/<script>/ as usual.

Exit Codes
----------
0   Every script passed.
1   At least one script failed or timed out.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from applog import get_logger

log = get_logger("run_all")

HERE = Path(__file__).resolve().parent
# Scripts known to take a minute or more; --skip-slow leaves them out.
SLOW = {
    "ex10_capstone.py": 60,
    "lab20_vqe_scaling_and_noise.py": 30,
    "ex06_shor.py": 30,
    "lab23_sparse_heisenberg.py": 30,
    "lab24_mps_heisenberg.py": 60,
    "lab25_dmrg.py": 60,
    "lab26_hva_vs_dmrg.py": 90,
    "lab27_mps_adjoint.py": 90,
    "lab28_deeper_hva.py": 150,
    "lab29_noisy_mps.py": 60,
    "lab30_zero_noise_extrapolation.py": 90,
    "lab31_symmetry_verification.py": 60,
    "lab32_pennylane_classifier.py": 90,
    "lab33_damping_and_coherent_errors.py": 90,
    "lab34_pauli_frames.py": 90,
    "lab35_probabilistic_error_cancellation.py": 90,
    "lab36_total_spin_verification.py": 90,
    "lab37_damping_and_extrapolation.py": 90,
    "lab38_randomised_compiling.py": 120,
    "lab39_spin_extrapolation.py": 60,
    "lab40_ansatz_symmetry.py": 90,
    "lab41_clifford_bond_gate.py": 150,
    "lab42_dmrg_excited.py": 150,
    "lab44_noise_aware_on_qpu.py": 150,
    "lab45_perbond_mps.py": 400,
}
SUMMARY = re.compile(r"All (\d+) (?:check|file)")


def discover():
    """Return the audit plus every lab and exercise script, in order."""
    # Sort so the order is stable: audit first, then labs, then exercises.
    labs = sorted(HERE.glob("lab[0-9]*.py"))
    exes = sorted(HERE.glob("ex[0-9]*.py"))
    return [HERE / "check_standards.py"] + labs + exes


def run_one(path, timeout):
    """Run a script; return (status, seconds, checks) without raising."""
    # Wall-clock start, so the table can show how long each script took.
    start = time.time()
    try:
        # Capture output so it can be parsed; the script keeps its own log file.
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=HERE.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start, None
    secs = time.time() - start
    # The scripts colour their output; strip ANSI codes before matching.
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    # The last 'All N checks passed' line carries the script's check count.
    found = SUMMARY.findall(text)
    checks = int(found[-1]) if found else None
    return ("PASS" if proc.returncode == 0 else "FAIL"), secs, checks


def main(argv=None):
    """Run the selected scripts and print the summary; return the exit code."""
    ap = argparse.ArgumentParser(description="Run every lab and exercise")
    ap.add_argument("--only", help="run scripts whose name contains this text")
    ap.add_argument(
        "--skip-slow", action="store_true", help="skip the scripts listed as slow"
    )
    ap.add_argument(
        "--timeout", type=int, default=900, help="seconds allowed per script"
    )
    args = ap.parse_args(argv)
    # Guard against a timeout so small that every script would fail.
    if args.timeout < 10:
        ap.error(f"--timeout must be >= 10. Got {args.timeout}.")
    # Apply --only and --skip-slow to the discovered scripts.
    plan = [
        p
        for p in discover()
        if (not args.only or args.only in p.name)
        and not (args.skip_slow and p.name in SLOW)
    ]
    if not plan:
        ap.error("no script matches the selection.")
    log.info("running %d script(s) with %s", len(plan), sys.executable)
    # rows collects (name, status, seconds, checks) for the final summary.
    rows = []
    for path in plan:
        status, secs, checks = run_one(path, args.timeout)
        rows.append((path.name, status, secs, checks))
        # One line per script: the level (PASS/FAIL) already shows the result.
        label = path.name + ("" if status == "PASS" else f" [{status}]")
        (log.ok if status == "PASS" else log.fail)(
            "%-40s %6.1f s  %s",
            label,
            secs,
            f"{checks} checks" if checks is not None else "",
        )
    # Anything that is not a clean PASS (failure or timeout) counts as failed.
    failed = [r for r in rows if r[1] != "PASS"]
    # Total checks across labs and exercises (the audit counts files instead).
    total = sum(r[3] or 0 for r in rows if r[0] != "check_standards.py")
    log.info(
        "total time %.0f s; %d checks reported by labs and exercises",
        sum(r[2] for r in rows),
        total,
    )
    if failed:
        log.fail(
            "%d of %d scripts failed: %s",
            len(failed),
            len(rows),
            ", ".join(r[0] for r in failed),
        )
        return 1
    log.ok("All %d scripts passed", len(rows))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
