#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab06_shor.py

Purpose
-------
Lab 6 factors a small integer N with Shor's algorithm, end to end.

Factoring reduces to ORDER FINDING. Pick a coprime to N; the smallest r with
a^r = 1 (mod N) is the order. If r is even and a^(r/2) != -1 (mod N), then
gcd(a^(r/2) - 1, N) is a non-trivial factor. Finding r is the hard part
classically. Quantumly it is phase estimation (Lab 5) on the unitary
U|y> = |a*y mod N>, whose eigenphases are s/r. The measured phase m/2^t is
then turned into r by continued fractions (a classical step).

The lab verifies each stage against a classical reference:
  1. Classical order finding (the ground truth).
  2. U is unitary and U^k equals multiplication by a^k mod N.
  3. The circuit's output peaks sit at multiples of 2^t / r.
  4. Continued fractions + gcd turn peaks into the factors of N.
  5. A seeded sampled run, like a real experiment.
  6. A bad choice of a (a^(r/2) = -1) yields no factors, as theory predicts.
  7. Optional (--big): N = 21, where peaks are not exact and fractions matter.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab06_shor.py
    python3 labs/lab06_shor.py --shots 50 --seed 7
    python3 labs/lab06_shor.py --big        # adds N=21 (about 20 s)

Log files
---------
logs/lab06/lab06_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import math
import sys
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from applog import get_logger
from lab05_qft_qpe import qft_circuit

log = get_logger("lab06")


def classical_order(a, n):
    """Smallest r > 0 with a^r = 1 (mod n): the ground truth we test against."""
    # Brute force is fine at this size; its exponential cost at 2048 bits is
    # exactly why Shor's algorithm matters.
    if math.gcd(a, n) != 1:
        raise ValueError(f"a={a} shares a factor with N={n}; no order exists")
    r, x = 1, a % n
    while x != 1:
        x = (x * a) % n
        r += 1
    return r


def modmult_matrix(a, n, width):
    """Permutation matrix of |y> -> |a*y mod n> on a width-qubit register.

    States y >= n are not valid residues, so they map to themselves. That
    keeps the matrix a permutation (hence unitary) for any width.
    """
    # A width-bit register holds 2**width basis states; the permutation
    # matrix below acts on all of them.
    d = 2**width
    u = np.zeros((d, d), dtype=complex)
    for y in range(d):
        u[(a * y) % n if y < n else y, y] = 1
    return u


def shor_circuit(a, n):
    """Order-finding circuit: counting register t = 2*width, target = width.

    The target starts at |1>, which is an equal superposition of the
    eigenstates of U, so each run samples one eigenphase s/r at random.
    Counting qubit j controls U^(2^j) = multiplication by a^(2^j) mod n,
    which we compute classically with pow() instead of repeating U.
    """
    # width bits hold any residue mod n; the counting register gets twice
    # as many qubits (set just below).
    width = n.bit_length()
    t = 2 * width  # 2*width bits give enough resolution for n^2 > r^2
    qc = QuantumCircuit(t + width)
    target = list(range(t, t + width))
    qc.x(target[0])  # target = |1>
    qc.h(range(t))  # counting register in superposition
    for j in range(t):
        gate = UnitaryGate(modmult_matrix(pow(a, 2**j, n), n, width))
        qc.append(gate.control(1), [j] + target)
    # Inverse QFT converts phase information into a readable integer m.
    qc.compose(qft_circuit(t, inverse=True), qubits=list(range(t)), inplace=True)
    return qc, t, width


def distribution(a, n):
    """Exact probability of each counting-register integer m, plus t."""
    qc, t, _ = shor_circuit(a, n)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities(qargs=list(range(t))), t


def factors_from_measurement(m, t, a, n):
    """Classical post-processing: measured m -> factors of n, or None.

    m / 2^t ~ s / r, so the best fraction with denominator <= n gives r.
    The candidate must really be an order, be even, and avoid a^(r/2) = -1.
    """
    # Continued-fraction step: the best rational approximation to m / 2^t
    # with denominator at most n is the candidate order r.
    r = Fraction(m, 2**t).limit_denominator(n).denominator
    if pow(a, r, n) != 1 or r % 2:
        return None
    half = pow(a, r // 2, n)
    if half == n - 1:
        return None
    found = {math.gcd(half - 1, n), math.gcd(half + 1, n)} - {1, n}
    return tuple(sorted(found)) or None


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def success_probability(dist, t, a, n):
    """Chance that one run yields factors, summed over every outcome m."""
    return float(
        sum(
            p
            for m, p in enumerate(dist)
            if p > 1e-12 and factors_from_measurement(m, t, a, n)
        )
    )


def main(argv=None):
    """Run every Lab 6 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 6: Shor's algorithm")
    ap.add_argument(
        "--shots",
        type=int,
        default=20,
        help="independent runs in the sampled experiment",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--big", action="store_true", help="also factor N=21 (15 qubits, about 20 s)"
    )
    args = ap.parse_args(argv)
    if args.shots < 1:
        ap.error(f"--shots must be >= 1. Got {args.shots}.")
    n, a = 15, 7
    r = []

    # Check 1: ground truth. Everything quantum below must agree with this.
    log.info("=== Check 1: classical order finding, N=%d ===", n)
    order = classical_order(a, n)
    log.info(
        "order of %d mod %d is %d (%s)",
        a,
        n,
        order,
        ", ".join(str(pow(a, k, n)) for k in range(order + 1)),
    )
    r.append(check(order == 4, "order of 7 mod 15 is 4"))
    r.append(check(classical_order(2, 21) == 6, "order of 2 mod 21 is 6"))

    # Check 2: the oracle. If U is not an exact permutation the whole
    # algorithm falls apart, so test it directly, including its powers.
    log.info("=== Check 2: modular multiplication is unitary ===")
    u = modmult_matrix(a, n, 4)
    r.append(check(np.allclose(u.conj().T @ u, np.eye(16)), "U is unitary"))
    # U^k must equal multiplication by a^k: this is why pow(a, 2^j, n) is legal.
    powers_ok = all(
        np.allclose(np.linalg.matrix_power(u, k), modmult_matrix(pow(a, k, n), n, 4))
        for k in range(1, 6)
    )
    r.append(check(powers_ok, "U^k = multiply by a^k mod N for k=1..5"))
    r.append(
        check(
            np.allclose(np.linalg.matrix_power(u, order), np.eye(16)),
            f"U^{order} = identity on the 4-qubit register",
        )
    )

    # Check 3: the quantum step. Peaks should sit at multiples of 2^t / order.
    log.info("=== Check 3: counting-register peaks, a=%d N=%d ===", a, n)
    dist, t = distribution(a, n)
    top = sorted(range(len(dist)), key=lambda m: -dist[m])[:order]
    log.info(
        "t=%d counting qubits; top outcomes %s",
        t,
        [(m, round(float(dist[m]), 4)) for m in sorted(top)],
    )
    spacing = 2**t // order
    r.append(
        check(
            sorted(top) == [k * spacing for k in range(order)],
            f"peaks at multiples of 2^t/r = {spacing}",
        )
    )
    r.append(
        check(
            abs(sum(dist[m] for m in top) - 1) < 1e-9,
            "all probability sits on those peaks (exact: r | 2^t)",
        )
    )

    # Check 4: classical post-processing. Only some peaks give factors; m=0
    # carries no information and m=2^t/2 gives the wrong denominator.
    log.info("=== Check 4: continued fractions and gcd ===")
    for m in sorted(top):
        got = factors_from_measurement(m, t, a, n)
        log.info("m=%3d -> m/2^t = %-5s factors=%s", m, Fraction(m, 2**t), got)
    p_win = success_probability(dist, t, a, n)
    r.append(
        check(
            abs(p_win - 0.5) < 1e-9,
            f"P(one run gives factors) = {p_win:.3f} (expect 0.5)",
        )
    )
    r.append(
        check(
            factors_from_measurement(64, t, a, n) == (3, 5),
            "m=64 recovers r=4 and factors 15 = 3 x 5",
        )
    )

    # Check 5: a seeded experiment. Real hardware only gives samples, so
    # repeat until a run succeeds; expected tries = 1 / 0.5 = 2.
    log.info("=== Check 5: sampled runs ===")
    rng = np.random.default_rng(args.seed)
    draws = rng.choice(len(dist), size=args.shots, p=dist / dist.sum())
    wins = [factors_from_measurement(int(m), t, a, n) for m in draws]
    hits = [w for w in wins if w]
    log.info("%d of %d runs produced factors", len(hits), args.shots)
    r.append(check(len(hits) > 0, "at least one run factored 15"))
    r.append(check(all(w == (3, 5) for w in hits), "every success gave exactly {3, 5}"))
    r.append(
        check(
            all(n % f == 0 for w in hits for f in w),
            "every reported factor divides N (verified classically)",
        )
    )

    # Check 6: a = 14 = -1 mod 15 has order 2 and a^(r/2) = -1, the known
    # failure mode. Theory says no run can ever succeed; the circuit agrees.
    log.info("=== Check 6: an unlucky choice of a ===")
    bad_dist, bad_t = distribution(14, n)
    r.append(check(classical_order(14, n) == 2, "order of 14 mod 15 is 2"))
    r.append(
        check(
            success_probability(bad_dist, bad_t, 14, n) == 0.0,
            "a=14: P(success) = 0 because a^(r/2) = -1 (mod N)",
        )
    )

    # Check 7 (optional): N=21 has r=6, which does not divide 2^t, so peaks
    # blur and only continued fractions can recover r.
    if args.big:
        log.info("=== Check 7: N=21, a=2 (non-exact peaks) ===")
        big_dist, big_t = distribution(2, 21)
        p_big = success_probability(big_dist, big_t, 2, 21)
        log.info("P(one run gives factors) = %.4f", p_big)
        r.append(check(p_big > 0.3, "N=21 succeeds with probability > 0.3"))
        # m=171 is about 1024/6: its fraction 1/6 hands us r=6 directly.
        r.append(
            check(
                factors_from_measurement(171, big_t, 2, 21) == (3, 7),
                "m=171 recovers r=6 and factors 21 = 3 x 7",
            )
        )

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
