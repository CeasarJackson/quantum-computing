#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab05_qft_qpe.py

Purpose
-------
Lab 5 builds the two subroutines that power Shor's algorithm:

  Quantum Fourier Transform (QFT)  turns a hidden periodicity into peaks you
      can read out. It is the discrete Fourier transform, F[j,k] = w^(jk)/sqrt(N)
      with w = exp(2*pi*i/N), done with O(n^2) gates instead of O(N log N).
  Quantum Phase Estimation (QPE)   reads the eigenvalue exp(2*pi*i*phi) of a
      unitary U into a binary register, using controlled-U^(2^j) followed by
      the INVERSE QFT.

Everything is verified against exact linear algebra: the circuit's unitary is
compared entry by entry with the DFT matrix you know from signal processing,
and QPE's accuracy is compared with its theoretical guarantee.

Bit convention: Qiskit is little-endian (Lab 3); integers returned here are
read directly from the register, so 'm' means phi ~ m / 2^t.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab05_qft_qpe.py
    python3 labs/lab05_qft_qpe.py --phi 0.3

Log files
---------
logs/lab05/lab05_<YYYYmmdd_HHMMSS>.log

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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from applog import get_logger

log = get_logger("lab05")


def qft_circuit(n, inverse=False):
    """n-qubit QFT: a Hadamard then controlled phases of halving angle per qubit.

    Qubit j gets H, then a phase of pi/2^(j-k) controlled by each lower qubit k.
    The final swaps reverse the qubit order so the output index matches the
    DFT's, which is how this circuit equals the DFT matrix exactly.
    """
    qc = QuantumCircuit(n, name="QFT")
    # Work from the top qubit down: each H makes qubit j an equal superposition,
    # then the controlled phases write the lower qubits' values into its phase.
    for j in reversed(range(n)):
        qc.h(j)
        for k in range(j):
            qc.cp(math.pi / 2 ** (j - k), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc.inverse() if inverse else qc


def dft_matrix(n):
    """The reference answer: F[j,k] = w^(jk) / sqrt(N)."""
    # Direct O(N^2) definition: slow but unambiguous, which is what a reference
    # answer should be. The circuit above reaches the same matrix with O(n^2) gates.
    big_n = 2**n
    w = np.exp(2j * np.pi / big_n)
    return np.array(
        [[w ** (j * k) for k in range(big_n)] for j in range(big_n)]
    ) / np.sqrt(big_n)


def qpe_circuit(t, phi):
    """Estimate phi for U = phase gate with U|1> = exp(2*pi*i*phi)|1>.

    Counting qubits 0..t-1, eigenstate qubit t (prepared in |1>). Counting
    qubit j applies U^(2^j) by a controlled phase of 2*pi*phi*2^j; phase
    kickback writes phi into the counting register as a Fourier state, and
    the inverse QFT converts it to the binary digits of phi.
    """
    qc = QuantumCircuit(t + 1)
    qc.x(t)  # eigenstate |1> of the phase gate
    qc.h(range(t))  # superposition of all U powers
    for j in range(t):
        qc.cp(2 * math.pi * phi * 2**j, j, t)
    qc.compose(qft_circuit(t, inverse=True), qubits=list(range(t)), inplace=True)
    return qc


def qpe_distribution(t, phi):
    """Exact probability of each counting-register integer m."""
    sv = Statevector.from_instruction(qpe_circuit(t, phi))
    return sv.probabilities(qargs=list(range(t)))


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def main(argv=None):
    """Run every Lab 5 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 5: QFT and phase estimation")
    ap.add_argument(
        "--phi",
        type=float,
        default=1 / 3,
        help="phase to estimate in [0,1) for the inexact QPE test",
    )
    args = ap.parse_args(argv)
    if not 0 <= args.phi < 1:
        ap.error(f"--phi must be in [0, 1). Got {args.phi}.")
    # r collects every check's boolean so the exit code reflects all of them.
    r = []

    # Check 1: the circuit IS the DFT. Compared entry by entry, not just by eye.
    # allclose, not ==: gate angles are floats, so entries agree only to ~1e-16.
    log.info("=== Check 1: QFT circuit == DFT matrix ===")
    for n in (2, 3, 4):
        u = Operator(qft_circuit(n)).data
        r.append(
            check(
                np.allclose(u, dft_matrix(n)),
                f"n={n}: {2 ** n}x{2 ** n} unitary matches DFT",
            )
        )
    inv = Operator(qft_circuit(3, inverse=True)).data
    r.append(
        check(
            np.allclose(inv @ Operator(qft_circuit(3)).data, np.eye(8)),
            "inverse QFT undoes QFT",
        )
    )

    # Check 2: a basis state has no structure, so its QFT is flat in magnitude;
    # all the information moves into the PHASES. Position <-> frequency.
    log.info("=== Check 2: QFT of a basis state is flat in magnitude ===")
    out = Statevector.from_label("001").evolve(qft_circuit(3)).data
    r.append(
        check(
            np.allclose(np.abs(out), 1 / np.sqrt(8)),
            "all 8 amplitudes have magnitude 1/sqrt(8)",
        )
    )

    # Check 3: the heart of Shor. A state periodic with period r has QFT
    # peaks at multiples of N/r. Here N=16 and r=4, so peaks at 0, 4, 8, 12.
    log.info("=== Check 3: periodicity becomes peaks (Shor's core idea) ===")
    # This is the kind of superposition Shor's algorithm builds; the spacing of
    # the peaks then reveals the period r, which is what factors the number.
    big_n, period = 16, 4
    amp = np.zeros(big_n, dtype=complex)
    amp[::period] = 1  # |0> + |4> + |8> + |12>
    amp /= np.linalg.norm(amp)
    p = Statevector(amp).evolve(qft_circuit(4)).probabilities()
    peaks = sorted(int(i) for i in np.nonzero(p > 1e-9)[0])
    log.info("peaks at %s, each with probability %.3f", peaks, p[peaks[0]])
    r.append(check(peaks == [0, 4, 8, 12], "peaks at exact multiples of N/r"))
    r.append(check(np.allclose(p[peaks], 0.25), "each peak has probability 1/4"))

    # Check 4: QPE is exact when phi is a multiple of 1/2^t.
    # Exact case: when phi * 2^t is an integer the Fourier state is a basis
    # state, so the inverse QFT lands on it with certainty (P = 1).
    log.info("=== Check 4: phase estimation, exactly representable phases ===")
    for t, phi in ((3, 0.25), (3, 0.625), (4, 0.8125)):
        dist = qpe_distribution(t, phi)
        m = int(dist.argmax())
        r.append(
            check(
                m == round(phi * 2**t) and dist[m] > 1 - 1e-9,
                f"t={t}, phi={phi}: read m={m} -> {m / 2 ** t} with P=1",
            )
        )

    # Check 5: for other phases QPE returns the nearest t-bit fraction with
    # probability at least 4/pi^2 ~ 0.405, and the error shrinks as t grows.
    log.info("=== Check 5: inexact phase phi = %.4f ===", args.phi)
    # 4/pi^2 is the WORST case (phi exactly halfway between two fractions).
    # Typical phases do better, as the logged probabilities show.
    floor_p = 4 / math.pi**2
    for t in (3, 4, 5, 6):
        dist = qpe_distribution(t, args.phi)
        m = int(dist.argmax())
        err = abs(args.phi - m / 2**t)
        log.info(
            "t=%d: best m=%d -> %.5f, error %.5f, P=%.3f", t, m, m / 2**t, err, dist[m]
        )
        r.append(
            check(
                err <= 1 / 2 ** (t + 1) + 1e-12 and dist[m] >= floor_p - 1e-9,
                f"t={t}: error <= 1/2^(t+1) and P >= 4/pi^2",
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
