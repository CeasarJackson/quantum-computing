#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab02_entanglement.py

Purpose
-------
Lab 2 scales the simulator to n qubits and shows what a single qubit cannot:
entanglement. An n-qubit state is a vector of 2^n amplitudes. Gates act on
one axis of that tensor (or two, for CNOT). A state is entangled exactly when
it cannot be factored into a product of single-qubit states; we detect that
with the Schmidt rank (SVD of the amplitude matrix) and quantify it with the
entanglement entropy of one qubit.

Qubit ordering: qubit 0 is the MOST significant bit, so '10' means qubit 0=1.
Qiskit uses the opposite (little-endian) convention; remember this in Lab 3.

Security
--------
Handles no secrets. Writes only a plain-text log under logs/lab02/.

Usage
-----
    python3 labs/lab02_entanglement.py
    python3 labs/lab02_entanglement.py --shots 50000 --seed 7

Log files
---------
logs/lab02/lab02_<YYYYmmdd_HHMMSS>.log

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

log = get_logger("lab02")

# Hadamard: the only gate we need to create superposition before entangling.
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def zero_state(n):
    """|00...0>: amplitude 1 on index 0, so a 2^n vector."""
    # Index 0 is the bitstring 00...0 because qubit 0 is the most significant bit.
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1
    return psi


def apply_1q(psi, U, q, n):
    """Apply 2x2 gate U to qubit q by contracting one tensor axis.

    Reshaping to [2]*n lets us touch one qubit without building the
    2^n x 2^n matrix kron(I,..,U,..,I), which would not scale.
    """
    t = psi.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))  # new axis 0 = output index
    return np.moveaxis(t, 0, q).reshape(-1)


def apply_cnot(psi, c, t, n):
    """Flip target t on the branch where control c is 1."""
    # Copy: reshape returns a view, and we must not mutate the caller's state.
    tensor = psi.reshape([2] * n).copy()
    sel = [slice(None)] * n
    sel[c] = 1  # the control=1 slice
    sub = tensor[tuple(sel)]
    # Removing axis c shifts later axes down by one.
    axis = t if t < c else t - 1
    tensor[tuple(sel)] = np.flip(sub, axis=axis)  # X on target = swap 0/1
    return tensor.reshape(-1)


def probs(psi):
    """Born rule for all 2^n outcomes."""
    return np.abs(psi) ** 2


def sample(psi, shots, rng):
    """Return {bitstring: count} for the nonzero outcomes."""
    # Recover n from the vector length so callers need not pass it separately.
    n = int(np.log2(psi.size))
    draws = rng.choice(psi.size, size=shots, p=probs(psi))
    vals, cnt = np.unique(draws, return_counts=True)
    return {format(v, f"0{n}b"): int(c) for v, c in zip(vals, cnt)}


def schmidt_rank(psi, n_a, n):
    """Rank of the amplitude matrix split after n_a qubits. 1 = product."""
    # Rows index qubits [0, n_a), columns the rest. The matrix has rank 1
    # exactly when psi factors as (state of A) x (state of B).
    m = psi.reshape(2**n_a, 2 ** (n - n_a))
    s = np.linalg.svd(m, compute_uv=False)
    return int(np.sum(s > 1e-10)), s


def entropy_bits(s):
    """Entanglement entropy in bits from Schmidt coefficients."""
    # Squared singular values are probabilities; drop zeros so 0*log(0) = 0
    # does not produce NaN.
    p = s**2
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log2(p)))


def reduced_qubit0(psi, n):
    """Density matrix of qubit 0 after tracing out the rest."""
    # Partial trace over everything but qubit 0: rho = M M^dagger where M is
    # the (2 x 2^(n-1)) amplitude matrix.
    m = psi.reshape(2, -1)
    return m @ m.conj().T


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def main(argv=None):
    """Run every Lab 2 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 2: entanglement")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.shots < 1:
        ap.error(f"--shots must be >= 1. Got {args.shots}.")
    # Seeded RNG keeps sampling reproducible so a failure can be replayed.
    rng = np.random.default_rng(args.seed)
    r = []

    # Check 1 baseline: two independent H gates cannot create correlation.
    log.info("=== Check 1: H on both qubits gives a PRODUCT state ===")
    prod = apply_1q(apply_1q(zero_state(2), H, 0, 2), H, 1, 2)
    rank, s = schmidt_rank(prod, 1, 2)
    r.append(check(np.allclose(probs(prod), [0.25] * 4), "uniform over 00..11"))
    r.append(check(rank == 1, f"Schmidt rank = {rank} (product, not entangled)"))

    # Check 2: H puts qubit 0 in superposition; CNOT copies that choice onto
    # qubit 1 as a correlation, which is what entanglement is.
    log.info("=== Check 2: H then CNOT builds the Bell state ===")
    bell = apply_cnot(apply_1q(zero_state(2), H, 0, 2), 0, 1, 2)
    log.info("Bell amplitudes (00,01,10,11) = %s", np.round(bell, 4))
    r.append(
        check(
            np.allclose(bell, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "state = (|00> + |11>)/sqrt(2)",
        )
    )
    rank, s = schmidt_rank(bell, 1, 2)
    r.append(check(rank == 2, f"Schmidt rank = {rank} (entangled)"))
    r.append(
        check(
            abs(entropy_bits(s) - 1.0) < 1e-9, "entanglement entropy = 1 bit (maximal)"
        )
    )

    # Check 3 tolerance: 0.02 is ~6 sigma at 20k shots, so it will not flake.
    log.info("=== Check 3: outcomes are perfectly correlated ===")
    counts = sample(bell, args.shots, rng)
    log.info("counts = %s", counts)
    r.append(check(set(counts) <= {"00", "11"}, "only 00 and 11 ever occur"))
    r.append(
        check(
            abs(counts.get("00", 0) / args.shots - 0.5) < 0.02,
            "each correlated outcome has probability ~1/2",
        )
    )

    # Check 4: tracing out qubit 1 leaves qubit 0 with no definite state of
    # its own, even though the joint state is perfectly known.
    log.info("=== Check 4: whole is pure, part is maximally mixed ===")
    rho = reduced_qubit0(bell, 2)
    purity = float(np.real(np.trace(rho @ rho)))
    log.info("qubit-0 density matrix = %s", np.round(rho.real, 4).tolist())
    r.append(check(np.allclose(rho, np.eye(2) / 2), "rho_0 = I/2"))
    r.append(
        check(
            abs(purity - 0.5) < 1e-9,
            f"purity = {purity:.3f} (1.0 would be a pure qubit)",
        )
    )

    # Check 5: CNOT chain 0->1->2 spreads the correlation across 3 qubits.
    log.info("=== Check 5: 3-qubit GHZ state ===")
    g = apply_1q(zero_state(3), H, 0, 3)
    g = apply_cnot(apply_cnot(g, 0, 1, 3), 1, 2, 3)
    r.append(
        check(
            set(sample(g, args.shots, rng)) <= {"000", "111"},
            "GHZ yields only 000 and 111",
        )
    )
    r.append(
        check(
            schmidt_rank(g, 1, 3)[0] == 2,
            "GHZ is entangled across qubit 0 | qubits 1,2",
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
