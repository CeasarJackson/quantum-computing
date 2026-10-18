#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab60_pipeline_null_control.py

Purpose
-------
Labs 55 to 59 found singlet-preparation loss about 3 to 4 times what the
calibration implies, repeating from run to run (Lab 57). Every one of those
numbers comes out of OUR pipeline: the Lab 55 circuits, the readout correction,
the implied-loss formula and the persistence statistics. Before the excess is
believed, the pipeline needs a NULL CONTROL: run exactly the same measurement and
analysis on a device whose true noise is known and equals the calibration, and see
whether the pipeline reports an excess or a persistence that is not there.

This lab does that at NO QPU time and NO network. For each of three simulated
backends (FakeFez, FakeKingston, FakeMarrakesh, whose noise models are built from
their own calibration snapshots) it chooses 8 links exactly as Lab 55 does, runs
the Lab 55 measurement twice (40 circuits x 2000 shots each), and analyses the
two runs with the Lab 55 estimator and the Lab 57 statistics. Optionally
(--data) it applies the same persistence statistic to the REAL Lab 57 table.

WHAT THE NULL DOES AND DOES NOT CONTAIN: the simulator noise is built from the
calibration (gate depolarizing error, thermal relaxation from T1 and T2, readout
error). It has no crosstalk, no drift between runs, no two-level-system defects, no
coherent error and no leakage, and its readout error equals the one the pipeline
measures. So a pipeline bias found here is a bias of the estimator or of the
implied-loss formula; a clean result here does NOT show that the real excess has
a physical cause, only that the pipeline does not manufacture it.

THE RULES, fixed here before the control is run. Links with both readout
attenuations at least 0.95 (the Lab 56 / 57 cut), all runs pooled.
  RULE 1 BIAS. B = mean(loss - implied) over the null links; sigma_B =
         sqrt(sum err^2) / n (shot noise only, a lower bound).
           UNBIASED   |B| < 0.01 : the pipeline does not create the real excess.
           INFLATES   B >= +0.02 : part of the real excess may be an artefact;
                                   the lab prints the real excess minus B.
           DEFLATES   B <= -0.02.
           SMALL BIAS otherwise.
  RULE 2 SPURIOUS HIGH. The fraction of null links classed HIGH (Lab 55 class) is
         printed with no rule of its own (an unbiased pipeline gives about 0).
  RULE 3 PERSISTENCE OF THE EXCESS. The Lab 57 statistic (Spearman rho of the
         repeat against run 1, permutation p) on the EXCESS loss - implied, once
         for the null runs (rho_null, p_null) and, with --data, once for the real
         Lab 57 table (rho_real, p_real).
           NOT SPECIFIC            rho_null >= 0.30 and p_null < 0.05 : the
                                   statistic finds persistence on a device with no
                                   hidden excess, so Lab 57's verdict is weakened.
           EXCESS PERSISTENCE REAL rho_real >= 0.70 and p_real < 0.01 and the
                                   null is not NOT SPECIFIC.
           NOT SHOWN               otherwise (the real excess does not persist
                                   beyond what the null allows).
         Without --data only the null side is printed.

LIMITATIONS: the fake backends are fixed old calibration snapshots, not the live
devices; only 2 runs per backend and 8 links each, so the null bias has an error of
about 0.005 (it is printed); the simulator readout is perfectly stationary, which
real readout was not (Lab 55 saw reported readout error wrong at run time);
a correlation of excess between two runs is a statistic, not a cause.

FIRST RUN (2026-10-18, full run, 3 fake backends x 8 links x 2 runs): RULE 1
B = -0.0005 +/- 0.0031 (n = 36), UNBIASED: the pipeline does not create an excess
of the size seen on hardware (real Lab 55 excess about +0.059). RULE 2: 0 of 36
null measurements classed HIGH. RULE 3: null excess persistence rho -0.40
(p 0.10, n = 18), real excess persistence rho +0.94 (p < 0.0001, n = 13) from the
Lab 57 CSV: EXCESS PERSISTENCE REAL. The real-data figure (loss minus implied,
not the loss alone) was computed by this rule for the first time here.
Limitation: the Aer noise model contains only what the calibration reports, so
this rules out an artefact of the analysis, not of the real device.

Modes
-----
  (default)  Full null control on the three simulated backends (several minutes).
  --quick    One backend, 500 shots, 1 pair of runs (for a fast self-test).
  --data     Also read the real Lab 57 table (newest logs/lab57/pairs_*.csv, or
             --lab57-csv) and apply Rule 3 to it. No network, no key.

Security
--------
No key, no network and no QPU: simulated backends only; the optional real table
is a local CSV with no secrets.

Usage
-----
    python3 labs/lab60_pipeline_null_control.py --quick
    python3 labs/lab60_pipeline_null_control.py --data

Log files
---------
logs/lab60/lab60_<YYYYmmdd_HHMMSS>.log (plain text)
logs/lab60/null_<YYYYmmdd_HHMMSS>.csv (one row per link: both null runs)

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (missing package, file, bad flag).
130 User interrupted execution.
===============================================================================
"""

import argparse
import csv
import glob
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from applog import get_logger

log = get_logger("lab60")

# Everything shared is imported so each piece of logic exists once.
try:
    from lab17_mirror_benchmark import make_sampler
    from lab51_herald_and_isolate import check
    from lab52_gate_count_orientation_idle import summarise
    from lab55_other_pairs_other_device import (
        build_jobs,
        eligible_links,
        implied_link_loss,
        select_links,
        transpile_for,
    )
    from lab56_calibration_vs_loss import GOOD_ALPHA, N_PERM, perm_p, spearman
    from lab57_persistence import (
        good_pairs,
        pair_rows,
        persistence_stats,
        persistence_verdict,
        write_csv,
    )
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeKingston, FakeMarrakesh
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

OUT_DIR = Path("logs/lab60")
FAKES = (
    ("fake_fez", FakeFez),
    ("fake_kingston", FakeKingston),
    ("fake_marrakesh", FakeMarrakesh),
)
B_SMALL, B_LARGE = 0.01, 0.02
RHO_NULL, P_NULL = 0.30, 0.05
RHO_REAL, P_REAL = 0.70, 0.01


# ----------------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------------
def excess_rows(rows):
    """Copies of pair rows with the implied loss subtracted from both losses."""
    out = []
    for r in rows:
        q = dict(r)
        q["loss1"] = r["loss1"] - r["implied"]
        q["loss2"] = r["loss2"] - r["implied"]
        out.append(q)
    return out


def bias_stats(rows):
    """(B, sigma_B, n) over every measurement (both runs) of the good links."""
    vals, errs = [], []
    for r in good_pairs(rows):
        vals += [r["loss1"] - r["implied"], r["loss2"] - r["implied"]]
        errs += [r["err1"], r["err2"]]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    return sum(vals) / n, math.sqrt(sum(e * e for e in errs)) / n, n


def bias_verdict(b):
    """Rule 1."""
    if not math.isfinite(b):
        return "no usable links"
    if abs(b) < B_SMALL:
        return "UNBIASED: the pipeline does not create an excess of this size"
    if b >= B_LARGE:
        return "INFLATES: part of the real excess may be an artefact of the pipeline"
    if b <= -B_LARGE:
        return "DEFLATES: the pipeline reads less loss than the calibration implies"
    return "SMALL BIAS: between 0.01 and 0.02 in size"


def persistence_rule(rho_null, p_null, rho_real=None, p_real=None):
    """Rule 3: compare the persistence of the excess on the null and the real data."""
    if (
        math.isfinite(rho_null)
        and math.isfinite(p_null)
        and rho_null >= RHO_NULL
        and p_null < P_NULL
    ):
        return "NOT SPECIFIC: the statistic finds persistence on a device with no hidden excess"
    if rho_real is None:
        return "null side only: no persistence found on the null (no real data given)"
    if rho_real >= RHO_REAL and p_real < P_REAL:
        return (
            "EXCESS PERSISTENCE REAL: the null does not persist, the real excess does"
        )
    return "NOT SHOWN: the real excess does not persist beyond what the null allows"


# ----------------------------------------------------------------------------
# The simulated runs
# ----------------------------------------------------------------------------
class _Args:
    """Stand-in for the argparse namespace (the sampler needs these fields)."""

    def __init__(self, shots):
        """Fields the sampler factory reads: shots, live, sampler."""
        self.shots, self.live, self.sampler, self.max_seconds = shots, False, "v2", 60


def run_fake(name, cls, shots, n_runs=2):
    """Run the Lab 55 measurement n_runs times on a simulated backend."""
    backend = cls()
    target = backend.target
    chosen = select_links(eligible_links(target))
    keys, circuits = build_jobs(chosen)
    isa = transpile_for(backend, chosen, keys, circuits)
    sx = target["sx"]
    implied = [
        implied_link_loss(e, (sx[(a,)].error, sx[(b,)].error)) for (a, b), e in chosen
    ]
    sampler = make_sampler(_Args(shots), backend)
    runs = []
    for i in range(n_runs):
        result = sampler.run(isa).result()
        runs.append({k: result[j].data.meas.get_counts() for j, k in enumerate(keys)})
        log.info("%s: run %d of %d done (%d circuits)", name, i + 1, n_runs, len(keys))
    return chosen, implied, runs


def null_rows(shots, fakes, n_runs=2):
    """Pair rows for every link of every simulated backend (first two runs)."""
    rows = []
    for name, cls in fakes:
        chosen, implied, runs = run_fake(name, cls, shots, n_runs)
        rows += pair_rows(name, chosen, implied, runs[0], runs[1])
    return rows


# ----------------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------------
def read_real(path):
    """The Lab 57 pairs CSV as pair rows (floats; classes kept as text)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            q = {
                k: float(v)
                for k, v in r.items()
                if k not in ("device", "link", "class1", "class2")
            }
            q.update({k: r[k] for k in ("device", "link", "class1", "class2")})
            rows.append(q)
    return rows


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def report(rows, real=None):
    """Print the null table, the three rules and, with real data, the comparison."""
    log.info("  device          link       implied   run 1               run 2")
    for r in rows:
        log.info(
            "  %-15s %-8s  %.4f   %+.3f +/- %.3f %-10s %+.3f +/- %.3f %-10s",
            r["device"],
            r["link"],
            r["implied"],
            r["loss1"],
            r["err1"],
            r["class1"],
            r["loss2"],
            r["err2"],
            r["class2"],
        )
    b, sb, n = bias_stats(rows)
    log.info(
        "RULE 1: B = mean(loss - implied) = %+.4f +/- %.4f (n = %d measurements)",
        b,
        sb,
        n,
    )
    log.info("verdict (bias): %s", bias_verdict(b))
    if real:
        log.info(
            "  real Lab 55 excess on well-read links was about +0.059; minus B: %+.3f",
            0.059 - b,
        )
    good = good_pairs(rows)
    hi = sum(r["class1"] == "HIGH" for r in good) + sum(
        r["class2"] == "HIGH" for r in good
    )
    log.info("RULE 2: %d of %d null measurements classed HIGH", hi, 2 * len(good))
    st = persistence_stats(good_pairs(excess_rows(rows)))
    log.info(
        "RULE 3 null: rho of the excess, run 2 against run 1 = %+.2f (p %.3f, n = %d)",
        st["rho"],
        st["p"],
        st["n"],
    )
    rho_real = p_real = None
    if real:
        sr = persistence_stats(good_pairs(excess_rows(real)))
        rho_real, p_real = sr["rho"], sr["p"]
        log.info(
            "RULE 3 real: rho of the excess, repeat against run 1 = %+.2f (p %.4f, n = %d)",
            rho_real,
            p_real,
            sr["n"],
        )
        log.info(
            "  (Lab 57 rule on the loss itself: %s)",
            persistence_verdict(*_loss_rho(real))[:40],
        )
    log.info(
        "verdict (persistence): %s",
        persistence_rule(st["rho"], st["p"], rho_real, p_real),
    )


def _loss_rho(real):
    """(rho, p) of the real Lab 57 table on the loss itself (for comparison)."""
    st = persistence_stats(good_pairs(real))
    return st["rho"], st["p"]


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def synthetic(n, offset, seed, stable=0.0):
    """Pair rows whose loss is implied + offset (+ a stable per-link part) + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        imp = 0.01 + 0.03 * float(rng.random())
        s = stable * float(rng.normal())
        base = imp + offset + s
        e = 0.015
        l1, l2 = base + e * float(rng.normal()), base + e * float(rng.normal())
        rows.append(
            {
                "device": "sim",
                "link": f"{i}-{i + 1}",
                "implied": imp,
                "loss1": l1,
                "err1": e,
                "class1": "CONSISTENT",
                "alpha1": 0.98,
                "loss2": l2,
                "err2": e,
                "class2": "CONSISTENT",
                "alpha2": 0.98,
            }
        )
    return rows


def check_pure(r):
    """The three rules and the helpers on cases with known answers."""
    log.info("=== Check 1: the rules on cases with known answers ===")
    b0 = bias_stats(synthetic(60, 0.0, 1))
    b1 = bias_stats(synthetic(60, 0.03, 2))
    r.append(
        check(
            bias_verdict(b0[0]).startswith("UNBIASED")
            and bias_verdict(b1[0]).startswith("INFLATES"),
            f"bias: planted 0 read {b0[0]:+.4f}, planted +0.03 read {b1[0]:+.4f}",
        )
    )
    r.append(
        check(
            bias_verdict(-0.03).startswith("DEFLATES")
            and bias_verdict(0.015).startswith("SMALL"),
            "rule 1: DEFLATES and SMALL BIAS",
        )
    )
    st_null = persistence_stats(good_pairs(excess_rows(synthetic(40, 0.0, 3))))
    st_stable = persistence_stats(
        good_pairs(excess_rows(synthetic(40, 0.0, 4, stable=0.05)))
    )
    r.append(
        check(
            st_null["rho"] < 0.5 and st_stable["rho"] > 0.7,
            f"persistence of the excess: none {st_null['rho']:+.2f}, planted stable part {st_stable['rho']:+.2f}",
        )
    )
    r.append(
        check(
            persistence_rule(0.5, 0.01, 0.9, 0.001).startswith("NOT SPECIFIC")
            and persistence_rule(0.0, 0.5, 0.9, 0.001).startswith(
                "EXCESS PERSISTENCE REAL"
            )
            and persistence_rule(0.0, 0.5, 0.5, 0.02).startswith("NOT SHOWN")
            and persistence_rule(0.0, 0.5).startswith("null side only"),
            "rule 3: all four outcomes",
        )
    )
    rows = synthetic(5, 0.0, 5)
    rows[0]["alpha2"] = 0.5
    r.append(
        check(
            bias_stats(rows)[2] == 8,
            "a link with poor readout in either run is left out",
        )
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 60: pipeline null control")
    ap.add_argument("--quick", action="store_true", help="one backend, 500 shots")
    ap.add_argument(
        "--data", action="store_true", help="also analyse the real Lab 57 table"
    )
    ap.add_argument("--lab57-csv", default=None, help="Lab 57 pairs CSV")
    ap.add_argument("--shots", type=int, default=2000, help="shots per circuit")
    args = ap.parse_args(argv)
    if not 100 <= args.shots <= 8192:
        ap.error(f"--shots must be 100..8192. Got {args.shots}.")
    r = []
    check_pure(r)

    real = None
    if args.data:
        path = (
            args.lab57_csv
            or (sorted(glob.glob("logs/lab57/pairs_*.csv")) or [None])[-1]
        )
        r.append(
            check(
                bool(path) and Path(path).exists(), f"real Lab 57 table found: {path}"
            )
        )
        if not (path and Path(path).exists()):
            return 2
        real = read_real(path)

    log.info("=== Check 2: the simulated null runs ===")
    fakes = FAKES[:1] if args.quick else FAKES
    shots = 500 if args.quick else args.shots
    rows = null_rows(shots, fakes)
    r.append(
        check(len(rows) == 8 * len(fakes), f"{len(rows)} null links measured twice")
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(rows, OUT_DIR / f"null_{stamp}.csv")
    report(rows, real)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
