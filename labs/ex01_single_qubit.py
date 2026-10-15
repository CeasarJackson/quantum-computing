#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex01_single_qubit.py

Purpose
-------
Lab 1 optional exercises, each as predict-then-verify:
  1. An Rz(theta) gate and an X-basis measurement (H before measuring).
  2. The gate sequence that puts the Bloch vector on +y.
  3. T^8 = I, and the smallest k with T^k = Z.
  4. Predict HSSH|0> and check it.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex01_single_qubit.py [--seed 42]

Log files
---------
logs/ex01/ex01_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import sys

import numpy as np

from applog import get_logger
from exlib import Tally, base_parser
from lab01_single_qubit import GATES, KET0, apply, bloch_vector, measure

log = get_logger("ex01")


def rz(theta):
    """Rotation about Z by theta: diag(e^{-i theta/2}, e^{+i theta/2})."""
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def measure_basis(state, basis, shots, rng):
    """Measure in the X, Y or Z basis by rotating that axis onto Z first."""
    # Measuring X means: H maps |+>,|-> to |0>,|1>, then measure normally.
    # Y needs S-dagger then H. Z needs nothing.
    prep = {"Z": [], "X": ["H"], "Y": ["S", "S", "S", "H"]}[basis]
    return measure(apply(state, *prep), shots, rng)


def ex1(t, rng):
    """Exercise 1: Rz and X-basis measurement."""
    t.ask("Exercise 1: Rz(theta) and measure_basis('X')")
    # Rz only changes the RELATIVE phase, so it is invisible in the Z basis;
    # |+> (equator of the sphere) is the state that makes it visible.
    t.predict(
        "|+> measured in X gives '0' 100%; Rz(theta) turns Bloch (1,0,0)"
        " into (cos t, sin t, 0)"
    )
    plus = apply(KET0, "H")
    counts = measure_basis(plus, "X", 1000, rng)
    t.note("|+> in X basis: %s", counts)
    t.check(counts == {"0": 1000, "1": 0}, "|+> is 100% '0' in the X basis")
    # Measuring |+> in Z contrasts with the deterministic X result above.
    # In the Z basis the same state is a coin flip: the basis choice matters.
    zc = measure_basis(plus, "Z", 4000, rng)
    t.note("|+> in Z basis: %s", zc)
    t.check(abs(zc["0"] / 4000 - 0.5) < 0.04, "|+> is 50/50 in the Z basis")
    # Sweep a few angles so a lucky coincidence cannot pass the check.
    for theta in (0.3, 1.0, np.pi / 2, 2.5):
        got = bloch_vector(rz(theta) @ plus)
        want = (np.cos(theta), np.sin(theta), 0.0)
        t.check(
            np.allclose(got, want, atol=1e-9),
            f"Rz({theta:.2f})|+> Bloch = (cos, sin, 0)",
        )
    # Rz(pi) is Z up to a global phase, so |+> becomes |-> (X-basis '1').
    flipped = measure_basis(rz(np.pi) @ plus, "X", 1000, rng)
    t.check(flipped["1"] == 1000, "Rz(pi)|+> = |-> : 100% '1' in the X basis")


def ex2(t):
    """Exercise 2: reach +y."""
    t.ask("Exercise 2: the gate sequence for +y")
    t.predict("H then S: |0> -> |+> -> (|0>+i|1>)/sqrt2, Bloch (0,1,0)")
    v = bloch_vector(apply(KET0, "H", "S"))
    t.note("Bloch vector after H,S = %s", np.round(v, 6))
    t.check(np.allclose(v, (0, 1, 0), atol=1e-9), "H then S lands on +y")
    # The reversed order cannot work: S on |0> is only a global phase.
    v2 = bloch_vector(apply(KET0, "S", "H"))
    t.note("S then H instead gives %s", np.round(v2, 6))
    t.check(np.allclose(v2, (1, 0, 0), atol=1e-9), "S then H gives +x, not +y")


def ex3(t):
    """Exercise 3: powers of T."""
    t.ask("Exercise 3: T^8 = I and the smallest k with T^k = Z")
    # Diagonal gates multiply, so powers just add phases: this is arithmetic
    # mod 8 on multiples of pi/4, which is why T is "the 8th root of unity".
    t.predict("T = diag(1, e^{i pi/4}), so T^k has phase k pi/4: T^8 = I, T^4 = Z")
    tg = GATES["T"]
    pw = lambda k: np.linalg.matrix_power(tg, k)  # noqa: E731
    t.check(np.allclose(pw(8), GATES["I"]), "T^8 = I")
    ks = [k for k in range(1, 9) if np.allclose(pw(k), GATES["Z"])]
    t.note("k with T^k = Z for k=1..8: %s", ks)
    t.check(ks == [4], "smallest (and only) k is 4")
    t.check(np.allclose(pw(2), GATES["S"]), "T^2 = S as a bonus")


def ex4(t):
    """Exercise 4: HSSH|0>."""
    t.ask("Exercise 4: predict HSSH|0>")
    # Conjugating by H swaps the X and Z axes, so H Z H = X: a standard
    # identity worth memorizing because it recurs in every basis change.
    t.predict("SS = Z and HZH = X, so HSSH|0> = X|0> = |1>")
    out = apply(KET0, "H", "S", "S", "H")
    t.note("amplitudes = %s", np.round(out, 6))
    t.check(np.allclose(out, [0, 1]), "HSSH|0> = |1> exactly")


def main(argv=None):
    """Run all four Lab 1 exercises; return the exit code."""
    args = base_parser("Lab 1 exercises").parse_args(argv)
    rng = np.random.default_rng(args.seed)
    t = Tally(log)
    ex1(t, rng)
    ex2(t)
    ex3(t)
    ex4(t)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
