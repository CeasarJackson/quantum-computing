#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab18_faulty_measurements.py

Purpose
-------
Lab 12 assumed the error-correction machinery itself was perfect: only data
qubits suffered errors and the syndrome was read out flawlessly. Real syndrome
measurements fail too. Lab 18 adds that, with the PHENOMENOLOGICAL noise model:

  * each round, every data qubit gets an X error with probability p;
  * each round, every syndrome bit is read wrongly with probability q;
  * the code is measured for R = d rounds, followed by one perfect final
    readout of the data qubits (the standard "memory experiment").

A single wrong syndrome bit cannot be told from a data error by looking at one
round, so the decoder works on the history: it takes DETECTION EVENTS (a
syndrome bit that changed between rounds) and matches them in SPACE-TIME with
minimum-weight perfect matching (pymatching). Data errors are space-like
edges; measurement errors are time-like edges.

The lab verifies:
  1. Every single fault (data error or measurement error, at any round) is
     corrected on the distance-3 code, while some pairs of faults are not.
  2. The space-time decoder beats a decoder that sees only the last round.
  3. Threshold: the d=3, 5, 7 curves cross near 3 percent (literature ~2.9
     percent for p = q), far below the ~10 percent of Lab 12.
  4. Below threshold a bigger code still wins, by a constant factor per +2 in d.
  5. Measurement noise matters: raising q at fixed p raises the logical error.

What this model leaves out: faults INSIDE the syndrome-extraction circuit
(CNOT errors that spread, "hook" errors), which lower the threshold to roughly
1 percent. Only X errors are simulated; Z errors are symmetric.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab18_faulty_measurements.py
    python3 labs/lab18_faulty_measurements.py --shots 40000 --seed 3
    python3 labs/lab18_faulty_measurements.py --quick

Log files
---------
logs/lab18/lab18_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (pymatching).
130 User interrupted execution.
===============================================================================
"""

import argparse
import itertools
import math
import sys

import numpy as np

from applog import get_logger

log = get_logger("lab18")

try:
    import pymatching
except ImportError:  # pragma: no cover
    log.fail("pymatching is not installed: pip install pymatching")
    sys.exit(2)

DISTANCES = (3, 5, 7)
P_GRID = (0.01, 0.02, 0.025, 0.03, 0.04, 0.05)


# ----------------------------------------------------------------------------
# Code construction (same rotated surface code as Lab 12)
# ----------------------------------------------------------------------------
def build_code(d):
    """Rotated surface code: (HX, HZ) check matrices over d*d data qubits."""
    # Plaquettes sit on the corners of the data grid, (i, j) in 0..d. Their
    # type alternates like a checkerboard. Bulk plaquettes touch four data
    # qubits; the boundary keeps only every other one (weight 2), with Z-type
    # on the top/bottom edges and X-type on the left/right edges.
    xs, zs = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            qubits = [
                a * d + b
                for a, b in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j))
                if 0 <= a < d and 0 <= b < d
            ]
            kind = "X" if (i + j) % 2 == 0 else "Z"
            bulk = 1 <= i <= d - 1 and 1 <= j <= d - 1
            if bulk:
                keep = True
            elif i in (0, d) and 1 <= j <= d - 1:
                keep = kind == "Z"
            elif j in (0, d) and 1 <= i <= d - 1:
                keep = kind == "X"
            else:
                keep = False  # corners host no check
            if keep:
                (xs if kind == "X" else zs).append(qubits)
    return to_matrix(xs, d * d), to_matrix(zs, d * d)


def to_matrix(checks, n):
    """List of qubit-index lists -> binary check matrix (one row per check)."""
    h = np.zeros((len(checks), n), dtype=np.uint8)
    for row, qubits in enumerate(checks):
        h[row, qubits] = 1
    return h


def logicals(d):
    """Logical X (a row of X's) and logical Z (a column of Z's), as bit vectors."""
    lx = np.zeros(d * d, dtype=np.uint8)
    lz = np.zeros(d * d, dtype=np.uint8)
    # A logical X runs along a full row; any row is equivalent.
    lx[0:d] = 1  # top row
    lz[0::d] = 1  # left column
    return lx, lz


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Space-time simulation and decoding
# ----------------------------------------------------------------------------
def make_decoder(hz, rounds, p, q):
    """Space-time matching graph: `rounds` noisy layers plus a perfect one.

    Layer t holds the detection events of round t. Edges between layers
    (time-like) are measurement errors; edges inside a layer (space-like) are
    data errors, and only those carry a fault id (the data qubit).
    """
    # Weights are log-likelihood ratios, so a likelier fault is cheaper.
    w_space = math.log((1 - p) / p)
    w_time = math.log((1 - q) / q) if q > 0 else 20.0
    return pymatching.Matching.from_check_matrix(
        hz, weights=w_space, repetitions=rounds + 1, timelike_weights=w_time
    )


def run_memory(hz, lz, data_err, meas_err, matching, last_only=None):
    """Decode a batch of fault histories; return (failed, final_syndrome).

    data_err  shape (shots, R, n): new X errors in each round.
    meas_err  shape (shots, R, c): flipped syndrome bits in each round.
    The final round is a perfect readout, so the decoder sees R noisy
    syndromes plus one clean one.
    """
    shots, rounds, n = data_err.shape
    # Total data error present after each round (errors accumulate).
    total = np.cumsum(data_err, axis=1) % 2
    # Noisy syndrome of every round, then the perfect final one.
    syn = (total @ hz.T) % 2 ^ meas_err
    syn = np.concatenate([syn, ((total[:, -1] @ hz.T) % 2)[:, None, :]], axis=1)
    # Detection events = change between consecutive rounds (first vs zeros).
    events = syn.copy()
    events[:, 1:] ^= syn[:, :-1]
    final = total[:, -1]
    if last_only is not None:
        # Baseline: ignore history, decode only the clean final syndrome.
        corr = last_only.decode_batch(syn[:, -1])
    else:
        corr = matching.decode_batch(events.reshape(shots, -1).astype(np.uint8))
    residual = (final ^ corr) % 2
    return ((residual @ lz) % 2).astype(bool)


def sample(shots, rounds, n, c, p, q, rng):
    """Draw random data and measurement faults for a batch of experiments."""
    data = (rng.random((shots, rounds, n)) < p).astype(np.uint8)
    meas = (rng.random((shots, rounds, c)) < q).astype(np.uint8)
    return data, meas


def logical_rate(d, p, q, shots, rng, baseline=False):
    """Logical failure rate of a distance-d memory experiment of d rounds."""
    hz = build_code(d)[1]
    lz = logicals(d)[1]
    data, meas = sample(shots, d, d * d, hz.shape[0], p, q, rng)
    if baseline:
        failed = run_memory(hz, lz, data, meas, None, last_only=pymatching.Matching(hz))
    else:
        failed = run_memory(hz, lz, data, meas, make_decoder(hz, d, p, max(q, 1e-6)))
    return float(failed.mean())


def stderr(x, shots):
    """Binomial standard error of a measured rate."""
    return math.sqrt(max(x, 1e-9) * (1 - x) / shots)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_faults(r):
    """Check 1: brute force over every single and double fault at d=3."""
    log.info("=== Check 1: every fault pattern on the d=3 code ===")
    d, rounds = 3, 3
    hz, lz = build_code(d)[1], logicals(d)[1]
    n, c = d * d, hz.shape[0]
    matching = make_decoder(hz, rounds, 0.01, 0.01)
    # A fault location is ("data", round, qubit) or ("meas", round, check).
    sites = [("d", t, i) for t in range(rounds) for i in range(n)] + [
        ("m", t, i) for t in range(rounds) for i in range(c)
    ]

    def failures(patterns):
        """Run each fault pattern (a tuple of sites); return failure flags."""
        data = np.zeros((len(patterns), rounds, n), dtype=np.uint8)
        meas = np.zeros((len(patterns), rounds, c), dtype=np.uint8)
        for row, pat in enumerate(patterns):
            for kind, t, i in pat:
                (data if kind == "d" else meas)[row, t, i] ^= 1
        return run_memory(hz, lz, data, meas, matching)

    singles = [(s,) for s in sites]
    bad1 = failures(singles).sum()
    pairs = list(itertools.combinations(sites, 2))
    bad2 = failures(pairs).sum()
    log.info(
        "%d single faults: %d fail; %d fault pairs: %d fail",
        len(singles),
        bad1,
        len(pairs),
        bad2,
    )
    r.append(
        check(
            bad1 == 0,
            "all single faults (data AND measurement, any "
            "round) are corrected: space-time distance is 3",
        )
    )
    r.append(
        check(
            bad2 > 0,
            f"{bad2} of {len(pairs)} fault pairs defeat the "
            "decoder: two faults can be too many",
        )
    )
    # A lone measurement error produces TWO detection events in time, which
    # is exactly why a single round of data cannot be trusted.
    one_meas = np.zeros((1, rounds, c), dtype=np.uint8)
    one_meas[0, 1, 0] = 1
    zero = np.zeros((1, rounds, n), dtype=np.uint8)
    total = np.cumsum(zero, axis=1)
    syn = (total @ hz.T) % 2 ^ one_meas
    syn = np.concatenate([syn, np.zeros((1, 1, c), dtype=np.uint8)], axis=1)
    ev = syn.copy()
    ev[:, 1:] ^= syn[:, :-1]
    r.append(
        check(
            int(ev.sum()) == 2,
            "one measurement error makes exactly 2 "
            "detection events (a time-like pair)",
        )
    )


def check_history(r, shots, rng):
    """Check 2: decoding with history versus only the last syndrome."""
    log.info("=== Check 2: space-time decoder versus last-round-only ===")
    p = 0.02
    full = logical_rate(5, p, p, shots, rng)
    last = logical_rate(5, p, p, shots, rng, baseline=True)
    log.info("d=5, p=q=%.2f: space-time %.4f, last-round-only %.4f", p, full, last)
    r.append(
        check(
            full < last - 3 * stderr(last, shots),
            "using the whole history cuts the logical error rate "
            f"({full:.4f} vs {last:.4f})",
        )
    )


def check_threshold(r, shots, rng):
    """Checks 3 and 4: the threshold and below-threshold scaling."""
    log.info("=== Check 3: threshold, p = q, %d shots per point ===", shots)
    # Failures are rare at the lowest p, so that column gets 4x the shots to
    # keep its counts large enough for the ratio checks below.
    rates = {
        (d, p): logical_rate(d, p, p, shots * (4 if p <= 0.01 else 1), rng)
        for d in DISTANCES
        for p in P_GRID
    }
    for p in P_GRID:
        log.info(
            "p=%.3f: d=3 %.4f   d=5 %.4f   d=7 %.4f",
            p,
            *(rates[d, p] for d in DISTANCES),
        )
    low, high = 0.01, 0.05
    r.append(
        check(
            rates[3, low] > rates[5, low] + 2 * stderr(rates[3, low], 4 * shots)
            and rates[5, low] > rates[7, low],
            f"p={low}: bigger code, fewer failures (d=3>5>7)",
        )
    )
    r.append(
        check(
            rates[3, high] < rates[5, high] < rates[7, high],
            f"p={high}: above threshold, bigger code is WORSE",
        )
    )
    # The d=3 and d=7 curves cross at the threshold (interpolate linearly).
    diff = [rates[3, p] - rates[7, p] for p in P_GRID]
    cross = None
    for k in range(len(P_GRID) - 1):
        if diff[k] > 0 >= diff[k + 1]:
            frac = diff[k] / (diff[k] - diff[k + 1])
            cross = P_GRID[k] + frac * (P_GRID[k + 1] - P_GRID[k])
    log.info(
        "d=3 and d=7 curves cross near p = %s (literature ~0.029 for "
        "this model for large d; small-d crossings read a bit low. Lab 12's "
        "data-only model gave ~0.094)",
        f"{cross:.3f}" if cross else "n/a",
    )
    r.append(
        check(
            cross is not None and 0.02 < cross < 0.04,
            "threshold estimate lies in 2%-4%, about 3x lower than "
            "with perfect measurements",
        )
    )
    log.info("=== Check 4: below-threshold suppression ===")
    ratios = (
        rates[3, low] / max(rates[5, low], 1e-9),
        rates[5, low] / max(rates[7, low], 1e-9),
    )
    log.info("suppression per +2 in distance at p=%.2f: x%.1f, x%.1f", low, *ratios)
    r.append(
        check(
            all(x > 1.5 for x in ratios),
            "each +2 in distance still cuts the failure rate by >1.5x",
        )
    )


def check_measurement_noise(r, shots, rng):
    """Check 5: logical error versus measurement-error rate q."""
    log.info("=== Check 5: effect of measurement noise (d=5, p=0.02) ===")
    qs = (0.0, 0.01, 0.02, 0.04)
    rates = [logical_rate(5, 0.02, q, shots, rng) for q in qs]
    for q, x in zip(qs, rates):
        log.info("q=%.2f: logical error %.4f", q, x)
    r.append(
        check(
            rates[-1] > rates[0] + 3 * stderr(rates[-1], shots),
            "worse measurements, worse logical qubit "
            f"({rates[0]:.4f} at q=0 -> {rates[-1]:.4f} at q=0.04)",
        )
    )


def main(argv=None):
    """Run every Lab 18 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 18: faulty measurements")
    ap.add_argument(
        "--shots", type=int, default=20000, help="experiments per data point"
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use 4000 shots")
    args = ap.parse_args(argv)
    if args.shots < 2000:
        ap.error(f"--shots must be >= 2000. Got {args.shots}.")
    shots = 4000 if args.quick else args.shots
    rng = np.random.default_rng(args.seed)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_faults(r)
    check_history(r, shots, rng)
    check_threshold(r, shots, rng)
    check_measurement_noise(r, shots, rng)
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
