#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab01_single_qubit.py

Purpose
-------
Lab 1 builds a single-qubit simulator from first principles using only NumPy.
A qubit is a unit vector in C^2; gates are 2x2 unitary matrices; measurement
is sampling from the Born-rule distribution |amplitude|^2. Writing this by
hand first means that when we move to Qiskit later, every abstraction
(circuit, backend, counts) maps to something you have already built.

The lab proves four facts numerically, each logged as PASS or FAIL:
  1. Every gate is unitary (U†U = I), so norm is preserved.
  2. H|0> gives a 50/50 superposition.
  3. H·H = I: the two paths into |1> cancel (interference, not randomness).
  4. Sampling counts converge to Born-rule probabilities as shots grow.

Security
--------
Touches no credentials, network, or secrets. Writes only a plain-text log
under logs/lab01/.

Usage
-----
    python3 labs/lab01_single_qubit.py
    python3 labs/lab01_single_qubit.py --shots 100000 --seed 7
    NO_COLOR=1 python3 labs/lab01_single_qubit.py

Log files
---------
logs/lab01/lab01_<YYYYmmdd_HHMMSS>.log  (plain text, no ANSI escapes)

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import sys

import numpy as np

from applog import get_logger

# Logging lives in applog.py (shared by every lab): colored console output and
# a plain-text file sink. Bind its helpers to the short names used below.
log = get_logger("lab01")
ok, warn_, fail = log.ok, log.warn_, log.fail


# ----------------------------------------------------------------------------
# Quantum model
# ----------------------------------------------------------------------------
# Computational basis. |0> = [1,0]^T, |1> = [0,1]^T. complex128 because
# amplitudes are complex in general (phase gates S and T need the imaginary part).
KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)

GATES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),  # bit flip
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),  # phase flip
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),  # sqrt(Z)
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}


def apply(state, *names):
    """Apply gates left-to-right (the order you would read a circuit)."""
    for n in names:
        state = GATES[n] @ state  # matrix acts on the LEFT of the ket
    return state


def probabilities(state):
    """Born rule: P(k) = |amplitude_k|^2."""
    return np.abs(state) ** 2


def measure(state, shots, rng):
    """Sample computational-basis outcomes; returns counts for '0' and '1'."""
    outcomes = rng.choice(2, size=shots, p=probabilities(state))
    return {"0": int(np.sum(outcomes == 0)), "1": int(np.sum(outcomes == 1))}


def bloch_vector(state):
    """(x, y, z) = (<X>, <Y>, <Z>). Pure states land on the unit sphere."""
    return tuple(float(np.real(np.conj(state) @ GATES[g] @ state)) for g in "XYZ")


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check(condition, label):
    """Log PASS/FAIL and return the boolean so callers can tally failures."""
    (ok if condition else fail)("%s", label)
    return bool(condition)


def main(argv=None):
    """Run every Lab 1 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 1: single-qubit simulator")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed; fixed by default so runs are reproducible",
    )
    args = ap.parse_args(argv)
    if args.shots < 1:
        ap.error(f"--shots must be >= 1. Got {args.shots}.")

    rng = np.random.default_rng(args.seed)
    results = []

    log.info("=== Check 1: every gate is unitary (U†U = I) ===")
    for name, U in GATES.items():
        results.append(
            check(np.allclose(U.conj().T @ U, np.eye(2)), f"{name} is unitary")
        )

    log.info("=== Check 2: H|0> is an equal superposition ===")
    plus = apply(KET0, "H")
    log.info("H|0> amplitudes = %s", np.round(plus, 4))
    results.append(
        check(
            np.allclose(probabilities(plus), [0.5, 0.5]), "P(0)=P(1)=0.5 after Hadamard"
        )
    )
    log.info(
        "Bloch vector = %s (expect +x axis)",
        tuple(round(v, 4) for v in bloch_vector(plus)),
    )

    log.info("=== Check 3: interference. H then H returns |0> ===")
    # If H merely 'randomized' the qubit, a second H could not undo it.
    # It does, because amplitudes (not probabilities) add, and the two
    # paths to |1> carry opposite signs and cancel.
    back = apply(KET0, "H", "H")
    log.info("HH|0> amplitudes = %s", np.round(back, 4))
    results.append(
        check(
            np.allclose(probabilities(back), [1.0, 0.0]),
            "HH|0> = |0> with certainty (destructive interference)",
        )
    )

    log.info("=== Check 4: phase is invisible until you interfere ===")
    # Z changes only the relative sign, so a direct measurement cannot see it,
    # but H·Z·H = X turns that hidden phase into a visible bit flip.
    z_plus = apply(KET0, "H", "Z")
    results.append(
        check(
            np.allclose(probabilities(z_plus), [0.5, 0.5]),
            "HZ|0> still measures 50/50 (phase hidden)",
        )
    )
    results.append(
        check(
            np.allclose(probabilities(apply(KET0, "H", "Z", "H")), [0.0, 1.0]),
            "HZH|0> = |1> (phase revealed by interference)",
        )
    )

    log.info("=== Check 5: sampling converges to Born probabilities ===")
    for n in sorted({100, 1000, args.shots}):
        c = measure(plus, n, rng)
        p1 = c["1"] / n
        # Standard error of a binomial proportion at p=0.5 is 0.5/sqrt(n);
        # 5 sigma keeps the check from flaking on an unlucky seed.
        results.append(
            check(
                abs(p1 - 0.5) < 5 * 0.5 / np.sqrt(n),
                f"shots={n:>6}: counts={c}  P(1)={p1:.4f}",
            )
        )

    failures = results.count(False)
    if failures:
        fail("%d of %d checks failed", failures, len(results))
        return 1
    ok("All %d checks passed", len(results))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        warn_("Interrupted by user")
        sys.exit(130)
