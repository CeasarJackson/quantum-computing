#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex05_qft_qpe.py

Purpose
-------
Lab 5 optional exercises, each as predict-then-verify:
  1. QFT peaks for period 2 and period 8 on N = 16.
  2. Period 3 on N = 16 (not a divisor of 16): blurred peaks, and how
     continued fractions still recovers r = 3.
  3. For phi = 0.7, the smallest t that guarantees error below 0.01.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex05_qft_qpe.py

Log files
---------
logs/ex05/ex05_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import sys
from fractions import Fraction

import numpy as np
from qiskit.quantum_info import Statevector

from applog import get_logger
from exlib import Tally, base_parser
from lab05_qft_qpe import qft_circuit, qpe_distribution

log = get_logger("ex05")

BIG_N = 16  # 4 qubits, as in Lab 5 Check 3


def comb_qft(period):
    """QFT probabilities of the comb state |0> + |r> + |2r> + ... on N = 16."""
    amp = np.zeros(BIG_N, dtype=complex)
    amp[::period] = 1  # the periodic superposition
    amp /= np.linalg.norm(amp)
    return Statevector(amp).evolve(qft_circuit(4)).probabilities()


def ex1(t):
    """Exercise 1: period 2 and period 8."""
    t.ask("Exercise 1: period 2 and period 8 on N = 16")
    t.predict(
        "peaks sit at multiples of N/r: r=2 -> {0, 8}; r=8 -> every"
        " multiple of 2 (0, 2, ..., 14)"
    )
    # r=2 is the coarsest comb and r=8 the finest that divides 16.
    for period in (2, 8):
        p = comb_qft(period)
        peaks = sorted(int(i) for i in np.nonzero(p > 1e-9)[0])
        want = list(range(0, BIG_N, BIG_N // period))
        t.note("r=%d: peaks %s, each P=%.4f", period, peaks, p[peaks[0]])
        t.check(peaks == want, f"r={period}: peaks at multiples of {BIG_N // period}")
        t.check(
            np.allclose(p[peaks], 1 / period),
            f"r={period}: each of the {period} peaks has probability 1/{period}",
        )
    # Shorter period means FEWER, WIDER-spaced peaks: period and peak spacing
    # are reciprocal, the uncertainty-principle flavor of the Fourier transform.


def ex2(t):
    """Exercise 2: period 3 does not divide 16."""
    t.ask("Exercise 2: period 3 on N = 16")
    t.predict(
        "peaks blur around 0, 16/3 = 5.33 and 10.67, so the top outcomes"
        " are 0, 5 and 11; 5/16 and 11/16 give r = 3 by continued"
        " fractions with denominator limit 4"
    )
    # Period 3 does not divide 16, the realistic situation for Shor.
    p = comb_qft(3)
    # argsort on -p picks the three most probable outcomes.
    top = sorted(int(i) for i in np.argsort(-p)[:3])
    t.note("top-3 outcomes %s with P = %s", top, np.round(p[top], 3))
    t.check(top == [0, 5, 11], "the three tallest peaks are at 0, 5, 11")
    t.check(
        p[top].sum() < 1 - 1e-3, "probability leaks to neighbours: peaks are NOT exact"
    )
    for m, want_num in ((5, 1), (11, 2)):
        # limit_denominator(4): we know r < 4, so no larger denominator can be
        # the period. That bound is the classical knowledge Shor also uses.
        frac = Fraction(m, BIG_N).limit_denominator(4)
        t.note("%d/16 = %.4f -> best fraction %s", m, m / 16, frac)
        t.check(
            frac == Fraction(want_num, 3) and frac.denominator == 3,
            f"{m}/16 recovers {frac} so r = 3",
        )
    # Guarantee: |m/N - s/r| < 1/(2 r^2) lets continued fractions succeed.
    t.check(
        abs(5 / 16 - 1 / 3) < 1 / (2 * 3**2),
        "5/16 is within 1/(2 r^2) of 1/3: recovery is guaranteed",
    )


def ex3(t):
    """Exercise 3: smallest t for error below 0.01 at phi = 0.7."""
    t.ask("Exercise 3: precision needed for phi = 0.7")
    t.predict(
        "worst-case error is 1/2^(t+1) < 0.01 needs t = 6; t = 5 gives"
        " a best estimate 22/32 = 0.6875 with error 0.0125"
    )
    # 0.7 has an infinite binary expansion, so no finite t is exact.
    phi = 0.7
    rows = {}
    # Sweep t to watch the error shrink by about half per added qubit.
    for tt in range(3, 9):
        dist = qpe_distribution(tt, phi)
        m = int(dist.argmax())
        err = abs(phi - m / 2**tt)
        # Probability that ONE run lands within 0.01 of phi.
        near = sum(dist[k] for k in range(2**tt) if abs(phi - k / 2**tt) < 0.01)
        rows[tt] = (m, err, float(near))
        t.note(
            "t=%d: best m=%d -> %.5f, error %.5f, bound %.5f, P(within" " 0.01)=%.3f",
            tt,
            m,
            m / 2**tt,
            err,
            1 / 2 ** (tt + 1),
            near,
        )
    # Worst-case bound versus what this particular phi achieves.
    guaranteed = min(tt for tt in rows if 1 / 2 ** (tt + 1) < 0.01)
    achieved = min(tt for tt in rows if rows[tt][1] < 0.01)
    t.check(guaranteed == 6, "worst-case guarantee first holds at t = 6")
    t.check(achieved == 6, "for phi = 0.7 the best estimate first beats 0.01 at t = 6")
    t.check(
        rows[5][0] == 22 and abs(rows[5][1] - 0.0125) < 1e-12,
        "t = 5 is not enough: m = 22 -> 0.6875, error 0.0125",
    )


def main(argv=None):
    """Run all three Lab 5 exercises; return the exit code."""
    base_parser("Lab 5 exercises").parse_args(argv)
    t = Tally(log)
    ex1(t)
    ex2(t)
    ex3(t)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
