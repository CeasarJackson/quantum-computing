#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab04_algorithms.py

Purpose
-------
Lab 4 implements the first three algorithms where a quantum computer needs
fewer queries to a black-box function f than any classical method:

  Deutsch-Jozsa      constant vs. balanced f(x) : 1 query  (classical: 2^(n-1)+1)
  Bernstein-Vazirani hidden bit-string s in f(x)=s.x xor b : 1 query (classical: n)
  Grover             find the one marked item in N = 2^n   : ~sqrt(N) (classical: ~N/2)

All three share one mechanism: superposition lets one oracle call touch every
input at once, and interference (Lab 1's H.H = I trick) cancels the wrong
answers so the right one is read out. Every result is checked against exact
statevector math, then Grover is also sampled like a real experiment.

Bit convention: Qiskit is little-endian, so integer bit i is qubit i, and the
integers returned here match the hidden masks directly (see Lab 3).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab04_algorithms.py
    python3 labs/lab04_algorithms.py --shots 20000 --seed 7 --marked 5

Log files
---------
logs/lab04/lab04_<YYYYmmdd_HHMMSS>.log

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

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from applog import get_logger

log = get_logger("lab04")
N_BITS = 3  # input register size; N = 2^3 = 8 items for every algorithm


def linear_oracle(n, s, b=0):
    """Oracle for f(x) = (s . x) xor b, using one ancilla (qubit n).

    A CX from input qubit i into the ancilla XORs x_i into the answer, so
    one CX per set bit of s computes the parity s.x. An X flips the constant b.
    s = 0 gives a constant function; any s != 0 gives a balanced one.
    """
    qc = QuantumCircuit(n + 1, name="Uf")
    for i in range(n):
        if (s >> i) & 1:
            qc.cx(i, n)
    if b:
        qc.x(n)
    return qc


def interference_circuit(n, oracle):
    """H on inputs, one oracle call, H on inputs: the DJ / BV skeleton.

    The ancilla starts in |-> (X then H). Then the oracle's XOR on the ancilla
    becomes a PHASE (-1)^f(x) on the input (phase kickback), and the final
    Hadamards turn those phases into a readable bit string.
    """
    qc = QuantumCircuit(n + 1)
    qc.x(n)
    qc.h(n)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    qc.h(range(n))
    return qc


def input_distribution(qc, n):
    """Exact probabilities over the n input qubits (ancilla traced out)."""
    return Statevector.from_instruction(qc).probabilities(qargs=list(range(n)))


def mark(qc, n, w):
    """Phase-flip the single basis state |w> on qubits 0..n-1 (the oracle)."""
    zeros = [i for i in range(n) if not (w >> i) & 1]
    if zeros:
        qc.x(zeros)  # turn |w> into |11...1> temporarily
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)  # H . MCX . H = multi-controlled Z
    qc.h(n - 1)
    if zeros:
        qc.x(zeros)  # undo


def diffuse(qc, n):
    """Grover diffusion: reflect every amplitude about the mean."""
    # H X (multi-controlled Z) X H is a reflection about |+...+>, the uniform
    # state, which is exactly "reflect about the mean amplitude".
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def grover_circuit(n, w, k, measure=False):
    """Uniform superposition, then k rounds of (oracle, diffusion)."""
    qc = QuantumCircuit(n)
    qc.h(range(n))  # equal amplitude on all N items
    # The oracle flips the sign of |w>; diffusion reflects every amplitude
    # about the mean, which turns that small dip into a larger peak on |w>.
    for _ in range(k):
        mark(qc, n, w)
        diffuse(qc, n)
    if measure:
        qc.measure_all()
    return qc


def grover_theory(n, k):
    """Closed form: P(marked) = sin^2((2k+1) * asin(1/sqrt(N)))."""
    return math.sin((2 * k + 1) * math.asin(1 / math.sqrt(2**n))) ** 2


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def main(argv=None):
    """Run every Lab 4 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 4: DJ, BV and Grover")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--marked", type=int, default=5, help="Grover's marked item, 0..7")
    args = ap.parse_args(argv)
    if args.shots < 1:
        ap.error(f"--shots must be >= 1. Got {args.shots}.")
    if not 0 <= args.marked < 2**N_BITS:
        ap.error(f"--marked must be 0..{2 ** N_BITS - 1}. Got {args.marked}.")
    # r collects every check's boolean so the exit code reflects all of them.
    n, r = N_BITS, []

    # Check 1: DJ. Probability of reading all-zeros is 1 iff f is constant.
    # For constant f every phase is equal, so the final H's return |000>. For
    # balanced f half the phases are -1 and they cancel |000> exactly (the same
    # interference as H.H = I in Lab 1, now across all 2^n inputs at once).
    log.info("=== Check 1: Deutsch-Jozsa, 1 query decides constant/balanced ===")
    for b in (0, 1):
        p0 = input_distribution(interference_circuit(n, linear_oracle(n, 0, b)), n)[0]
        r.append(check(abs(p0 - 1) < 1e-9, f"constant f={b}: P(000) = {p0:.3f}"))
    # Worst case over every balanced oracle: if even the largest P(000) is ~0,
    # then each one is rejected with certainty, not merely with high odds.
    worst = max(
        input_distribution(interference_circuit(n, linear_oracle(n, s)), n)[0]
        for s in range(1, 2**n)
    )
    r.append(
        check(worst < 1e-9, f"all {2 ** n - 1} balanced oracles: P(000) <= {worst:.1e}")
    )
    log.info("classical worst case needs %d queries; quantum used 1", 2 ** (n - 1) + 1)

    # Check 2: BV. The measured integer equals the hidden mask, exactly.
    # s = 0 is included on purpose: BV must also report "all zeros" correctly.
    # The constant b only adds a global phase, so it must never change the answer.
    log.info("=== Check 2: Bernstein-Vazirani recovers the hidden string ===")
    ok_all = True
    for s in range(2**n):
        for b in (0, 1):
            dist = input_distribution(
                interference_circuit(n, linear_oracle(n, s, b)), n
            )
            got = int(dist.argmax())
            # Demand the right answer AND probability 1: deterministic, not lucky.
            ok_all &= (got == s) and abs(dist[got] - 1) < 1e-9
    r.append(check(ok_all, f"all {2 ** (n + 1)} (s, b) pairs recovered with P=1"))
    log.info("classical needs %d queries (one per bit); quantum used 1", n)

    # Check 3: Grover amplitude amplification versus the closed form.
    # k = 0 is just the uniform superposition (P = 1/N). Each round rotates the
    # state by 2*theta toward |w>, so P climbs, peaks near k = pi/4*sqrt(N), and
    # then FALLS as the rotation overshoots. Stopping at the right k matters.
    log.info("=== Check 3: Grover, marked item %d of %d ===", args.marked, 2**n)
    for k in (0, 1, 2, 3):
        sv = Statevector.from_instruction(grover_circuit(n, args.marked, k))
        p = float(sv.probabilities()[args.marked])
        log.info(
            "k=%d iterations: P(marked) = %.4f (theory %.4f)", k, p, grover_theory(n, k)
        )
        r.append(
            check(
                abs(p - grover_theory(n, k)) < 1e-9,
                f"k={k} matches sin^2((2k+1)*theta)",
            )
        )
    # Optimal rounds grow like sqrt(N): this is Grover's quadratic speedup.
    best = math.floor(math.pi / 4 * math.sqrt(2**n))
    r.append(check(grover_theory(n, best) > 0.9, f"optimal k={best} gives P > 0.9"))
    r.append(
        check(
            grover_theory(n, 3) < grover_theory(n, best),
            "k=3 OVERSHOOTS: more iterations can lower the odds",
        )
    )

    # Check 4: sample it like a real experiment. Labels are q2 q1 q0.
    # Sampling adds shot noise (the ~6% of shots that miss), which is what the
    # same circuit looks like on real hardware before any device noise is added.
    log.info("=== Check 4: sampled Grover run ===")
    qc = grover_circuit(n, args.marked, best, measure=True)
    counts = (
        StatevectorSampler(seed=args.seed)
        .run([qc], shots=args.shots)
        .result()[0]
        .data.meas.get_counts()
    )
    top, hits = max(counts.items(), key=lambda kv: kv[1])
    log.info(
        "top outcomes = %s", dict(sorted(counts.items(), key=lambda kv: -kv[1])[:3])
    )
    r.append(
        check(
            int(top, 2) == args.marked,
            f"most frequent outcome is '{top}' = {args.marked}",
        )
    )
    r.append(
        check(hits / args.shots > 0.9, f"found with frequency {hits / args.shots:.3f}")
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
