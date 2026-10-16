#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab12_surface_code.py

Purpose
-------
Lab 12 builds the rotated SURFACE CODE and measures how well it protects a
qubit. Lab 7's 3-qubit code fixed only bit flips. The surface code arranges
d x d data qubits on a grid with two kinds of parity checks, X-type and
Z-type plaquettes, so it corrects both error types, and its protection grows
exponentially with the distance d as long as physical errors are rare enough.

The lab verifies, in order:
  1. Structure: (d^2-1)/2 X checks and as many Z checks, X and Z checks
     commute, exactly one logical qubit, logical operators of weight d.
  2. Distance 3 by brute force: no error of weight 1 or 2 is invisible AND
     harmful; every single-qubit error is corrected; some weight-2 are not.
  3. The exact logical error rate of d = 3 (all 512 error patterns), checked
     against a Monte Carlo simulation.
  4. The threshold: below p ~ 10% bigger codes fail less; above it, more.
  5. The same syndromes read out by actual Qiskit circuits with ancilla
     qubits (X errors via Z checks, Z errors via X checks).

Model: "code capacity" noise. Each data qubit gets an X error with
probability p, independently a Z error with probability p, and the parity
checks themselves are perfect. Real devices also have faulty checks, which
needs repeated rounds and gives a much lower threshold (about 1%). That is
NOT simulated here. Decoder: minimum-weight perfect matching (pymatching).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab12_surface_code.py
    python3 labs/lab12_surface_code.py --shots 100000 --seed 7

Log files
---------
logs/lab12/lab12_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing package (pip install pymatching).
130 User interrupted execution.
===============================================================================
"""

import argparse
import itertools
import math
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from applog import get_logger

log = get_logger("lab12")

# pymatching does the minimum-weight perfect matching decoding; fail early
# with the exact fix rather than a traceback.
try:
    import pymatching
except ImportError:  # pragma: no cover
    log.fail("pymatching is not installed: pip install pymatching")
    sys.exit(2)

# Odd distances only: d = 2t+1 corrects t errors, and 3, 5, 7 show the trend.
DISTANCES = (3, 5, 7)
# Physical error rates straddling the ~10% threshold.
P_GRID = (0.02, 0.05, 0.08, 0.10, 0.12, 0.15)


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


def gf2_rank(m):
    """Rank over GF(2) by Gaussian elimination (number of independent rows)."""
    m = m.copy() % 2
    rank = 0
    for col in range(m.shape[1]):
        pivots = [r for r in range(rank, m.shape[0]) if m[r, col]]
        if not pivots:
            continue
        m[[rank, pivots[0]]] = m[[pivots[0], rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]  # XOR = addition mod 2
        rank += 1
    return rank


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


def logical_failure(h, observable, errors, matching):
    """Fraction of error vectors whose decoded residual flips the logical."""
    # Syndrome = parity of the errors on each check's qubits.
    syndromes = (errors @ h.T) % 2
    # Minimum-weight matching pairs up syndrome defects and boundaries.
    corrections = matching.decode_batch(syndromes)
    # What remains after correction; harmless unless it is a logical operator.
    residual = (errors ^ corrections) % 2
    return ((residual @ observable) % 2).astype(bool)


def check_structure(r):
    """Check 1: the code has the right shape for every distance."""
    log.info("=== Check 1: structure of the rotated surface code ===")
    for d in DISTANCES:
        hx, hz = build_code(d)
        n = d * d
        lx, lz = logicals(d)
        # Logical qubits k = data qubits minus independent checks.
        k = n - gf2_rank(hx) - gf2_rank(hz)
        log.info(
            "d=%d: %d data qubits, %d X checks, %d Z checks, k=%d",
            d,
            n,
            len(hx),
            len(hz),
            k,
        )
        want = (n - 1) // 2
        r.append(
            check(
                len(hx) == want and len(hz) == want,
                f"d={d}: (d^2-1)/2 = {want} checks of each type",
            )
        )
        # X and Z checks must overlap on an even number of qubits, otherwise
        # they would not commute and could not be measured together.
        r.append(
            check(
                not ((hx @ hz.T) % 2).any(),
                f"d={d}: every X check commutes with every Z check",
            )
        )
        r.append(check(k == 1, f"d={d}: encodes exactly one logical qubit"))
        # A logical X must commute with all Z checks yet anticommute with
        # logical Z (odd overlap); otherwise it would be a stabilizer.
        r.append(
            check(
                not ((hz @ lx) % 2).any()
                and not ((hx @ lz) % 2).any()
                and int(lx @ lz) % 2 == 1,
                f"d={d}: logicals commute with checks, anticommute "
                f"with each other, weight {int(lx.sum())}",
            )
        )


def check_distance3(r):
    """Checks 2-3: brute force at d=3, including the exact logical rate."""
    log.info("=== Check 2: distance 3 by brute force ===")
    hx, hz = build_code(3)
    _, lz = logicals(3)
    matching = pymatching.Matching(hz)
    # All 2^9 X-error patterns, as rows of bits.
    # Exhaustive enumeration is only feasible because d=3 has 9 qubits.
    patterns = np.array(list(itertools.product((0, 1), repeat=9)), dtype=np.uint8)
    weight = patterns.sum(axis=1)
    syndrome = (patterns @ hz.T) % 2
    flips = ((patterns @ lz) % 2).astype(bool)
    invisible = ~syndrome.any(axis=1)
    # An undetected error that flips the logical is a logical operator: its
    # smallest weight IS the code distance.
    bad = invisible & flips
    r.append(
        check(
            weight[bad].min() == 3,
            f"smallest undetectable logical error has weight "
            f"{weight[bad].min()} (distance 3)",
        )
    )
    # Ties between equal-weight corrections are broken by pymatching,
    # so the weight-2 count depends on its tie-break rule.
    failed = logical_failure(hz, lz, patterns, matching)
    by_weight = {w: int(failed[weight == w].sum()) for w in range(4)}
    log.info(
        "decoder failures by error weight: %s (of %s patterns)",
        by_weight,
        {w: int((weight == w).sum()) for w in range(4)},
    )
    r.append(
        check(
            by_weight[0] == 0 and by_weight[1] == 0,
            "every weight-0 and weight-1 error is corrected",
        )
    )
    r.append(
        check(
            by_weight[2] > 0,
            f"{by_weight[2]} of 36 weight-2 patterns defeat the code: "
            "it corrects ONE error, not two",
        )
    )

    log.info("=== Check 3: exact logical error rate at d=3 vs Monte Carlo ===")
    return patterns, weight, failed, hz, lz, matching


def exact_rate(weight, failed, p):
    """Sum P(pattern) over failing patterns: the exact logical error rate."""
    # Probability of one exact pattern: p per flipped qubit, 1-p per intact one.
    prob = (p**weight) * ((1 - p) ** (9 - weight))
    return float(prob[failed].sum())


def check_exact_vs_mc(r, bundle, shots, rng):
    """Check 3: enumeration agrees with sampling; compare with Lab 7's code."""
    patterns, weight, failed, hz, lz, matching = bundle
    p = 0.05
    exact = exact_rate(weight, failed, p)
    # Each qubit flips independently with probability p.
    errors = (rng.random((shots, 9)) < p).astype(np.uint8)
    mc = float(logical_failure(hz, lz, errors, matching).mean())
    sigma = math.sqrt(exact * (1 - exact) / shots)
    log.info("p=%.2f: exact %.5f, Monte Carlo %.5f +/- %.5f", p, exact, mc, sigma)
    r.append(
        check(
            abs(mc - exact) < 5 * sigma,
            "Monte Carlo agrees with exact enumeration (5 sigma)",
        )
    )
    # Lab 7's closed form for the 3-qubit repetition code.
    rep = 3 * p**2 - 2 * p**3
    log.info(
        "Lab 7's 3-qubit code (X errors only): %.5f; surface d=3: %.5f", rep, exact
    )
    # Honest comparison: for X noise alone the repetition code is better at
    # equal distance, because it has fewer ways to fail. The surface code pays
    # for protecting BOTH error types with the same distance.
    r.append(
        check(
            exact > rep,
            "at d=3 the surface code is worse than Lab 7's repetition "
            "code against X noise alone (it also fixes Z, which Lab 7 cannot)",
        )
    )


def check_threshold(r, shots, rng):
    """Check 4: the logical error rate versus p for d = 3, 5, 7."""
    log.info("=== Check 4: threshold, %d shots per point ===", shots)
    # rates[(d, p)] holds the measured logical failure rate.
    rates = {}
    for d in DISTANCES:
        _, hz = build_code(d)
        _, lz = logicals(d)
        matching = pymatching.Matching(hz)
        for p in P_GRID:
            errors = (rng.random((shots, d * d)) < p).astype(np.uint8)
            rates[d, p] = float(logical_failure(hz, lz, errors, matching).mean())
    for p in P_GRID:
        log.info(
            "p=%.2f: d=3 %.4f   d=5 %.4f   d=7 %.4f",
            p,
            *(rates[d, p] for d in DISTANCES),
        )

    def sig(x):
        """Binomial standard error of a measured rate."""
        return math.sqrt(max(x, 1e-9) * (1 - x) / shots)

    # One point deep below threshold, one well above it.
    low, high = 0.02, 0.15
    r.append(
        check(
            rates[3, low] > rates[5, low] + 3 * sig(rates[3, low])
            and rates[5, low] > rates[7, low] + 3 * sig(rates[5, low]),
            f"p={low}: bigger code, fewer failures (d=3>5>7)",
        )
    )
    r.append(
        check(
            rates[3, high] < rates[5, high] - 3 * sig(rates[5, high])
            and rates[5, high] < rates[7, high] - 3 * sig(rates[7, high]),
            f"p={high}: above threshold, bigger code is WORSE (d=3<5<7)",
        )
    )
    # The d=3 and d=7 curves cross at the threshold; interpolate linearly.
    # Where the small and large codes perform equally is the threshold.
    diff = [rates[3, p] - rates[7, p] for p in P_GRID]
    cross = None
    for k in range(len(P_GRID) - 1):
        if diff[k] > 0 >= diff[k + 1]:
            frac = diff[k] / (diff[k] - diff[k + 1])
            cross = P_GRID[k] + frac * (P_GRID[k + 1] - P_GRID[k])
    log.info(
        "d=3 and d=7 curves cross near p = %s (literature: ~0.103 for "
        "this noise model)",
        f"{cross:.3f}" if cross else "n/a",
    )
    r.append(
        check(
            cross is not None and 0.08 < cross < 0.12,
            "threshold estimate lies in 8%-12%",
        )
    )
    # Below threshold the suppression is exponential in d: every +2 in d
    # multiplies the failure rate by a roughly constant factor < 1.
    # Ratios of consecutive distances show the exponential suppression.
    ratios = (
        rates[3, 0.02] / max(rates[5, 0.02], 1e-9),
        rates[5, 0.02] / max(rates[7, 0.02], 1e-9),
    )
    log.info("suppression per +2 in distance at p=0.02: x%.1f, x%.1f", *ratios)
    r.append(
        check(
            all(x > 2 for x in ratios),
            "each +2 in distance cuts the failure rate by more than 2x",
        )
    )


def syndrome_circuit(h, kind, qubit):
    """Qiskit circuit: prepare data, inject one error, measure all checks.

    kind "X": data in |0>, X error, Z checks (ancilla = parity of data bits).
    kind "Z": data in |+>, Z error, X checks (H, controlled-X, H on ancilla).
    Qubits 0..n-1 are data; n.. are one ancilla per check.
    """
    n = h.shape[1]
    qc = QuantumCircuit(n + h.shape[0])
    if kind == "Z":
        qc.h(range(n))  # |+> is the +1 eigenstate of X
    (qc.x if kind == "X" else qc.z)(qubit)  # the injected error
    for row in range(h.shape[0]):
        # One ancilla per check, stacked after the data qubits.
        anc = n + row
        support = np.nonzero(h[row])[0]
        if kind == "X":
            for q in support:
                qc.cx(int(q), anc)  # anc accumulates Z parity
        else:
            qc.h(anc)
            for q in support:
                qc.cx(anc, int(q))  # X parity via phase kickback
            qc.h(anc)
    return qc


def check_circuits(r):
    """Check 5: ancilla circuits reproduce the classical syndromes."""
    log.info("=== Check 5: Qiskit syndrome extraction (d=3, 17 qubits) ===")
    hx, hz = build_code(3)
    for kind, h in (("X", hz), ("Z", hx)):
        ok = True
        for q in range(9):
            qc = syndrome_circuit(h, kind, q)
            anc = list(range(9, 9 + h.shape[0]))
            # Ancillas are deterministic here, so one outcome should have probability 1.
            probs = Statevector.from_instruction(qc).probabilities_dict(anc)
            outcome, prob = max(probs.items(), key=lambda kv: kv[1])
            # Qiskit prints ancilla 0 rightmost; reverse to match row order.
            measured = np.array([int(b) for b in outcome[::-1]], dtype=np.uint8)
            ok &= bool(np.array_equal(measured, h[:, q])) and prob > 1 - 1e-9
        r.append(
            check(
                ok,
                f"{kind} errors: all 9 single-qubit syndromes from "
                f"the circuit equal column q of H, with certainty",
            )
        )
    qc = syndrome_circuit(hz, "X", 4)
    log.info(
        "example (X error on the centre qubit): %d gates, depth %d",
        sum(qc.count_ops().values()),
        qc.depth(),
    )


def main(argv=None):
    """Run every Lab 12 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 12: the surface code")
    ap.add_argument(
        "--shots",
        type=int,
        default=40000,
        help="Monte Carlo samples per (distance, p) point",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.shots < 5000:
        ap.error(f"--shots must be >= 5000. Got {args.shots}.")
    rng = np.random.default_rng(args.seed)
    r = []
    check_structure(r)
    bundle = check_distance3(r)
    check_exact_vs_mc(r, bundle, args.shots, rng)
    check_threshold(r, args.shots, rng)
    check_circuits(r)
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
