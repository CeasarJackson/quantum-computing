#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab07_noise_qec.py

Purpose
-------
Lab 7 shows why real quantum computers need error correction, and builds the
smallest code that works: the 3-qubit bit-flip repetition code.

Noise is modelled with density matrices and Kraus operators, rho -> sum K rho K†.
That is the honest description of a qubit that talks to its environment: the
state becomes MIXED, so the Bloch vector shrinks inside the sphere.

The lab verifies, against closed forms and against Qiskit:
  1. Bit-flip, phase-flip, depolarizing and amplitude-damping channels are
     trace preserving (sum K†K = I), so they are physical.
  2. Each channel shrinks the Bloch vector by the predicted factors.
  3. Qiskit's Kraus / DensityMatrix agrees with our NumPy channels.
  4. Fidelity decays exponentially with the number of noisy steps (T1 too).
  5. The bit-flip code turns error rate p into 3p^2 - 2p^3, and keeps
     superpositions intact (not just classical bits).
  6. The same code does NOT protect against phase flips (it makes them worse).
  7. A Qiskit circuit with syndrome ancillas corrects every single X error.
  8. A seeded Monte Carlo run reproduces the logical error rate.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab07_noise_qec.py
    python3 labs/lab07_noise_qec.py --p 0.2 --shots 500000 --seed 7

Log files
---------
logs/lab07/lab07_<YYYYmmdd_HHMMSS>.log

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
from qiskit.quantum_info import (
    DensityMatrix,
    Kraus,
    Statevector,
    partial_trace,
    state_fidelity,
)

from applog import get_logger

log = get_logger("lab07")

# Big-endian convention (qubit 0 = leftmost bit), as in Lab 2. The Qiskit
# circuit in check 7 is compared only through qubit-order-independent
# quantities (fidelity of the traced-out data register), so no conversion.
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
THETA, PHI = 1.0, 0.7  # a generic, non-special test state


# ----------------------------------------------------------------------------
# Channels (each is a list of Kraus operators)
# ----------------------------------------------------------------------------
def bit_flip(p):
    """X error with probability p."""
    return [math.sqrt(1 - p) * I2, math.sqrt(p) * PX]


def phase_flip(p):
    """Z error with probability p: destroys phase, leaves populations alone."""
    return [math.sqrt(1 - p) * I2, math.sqrt(p) * PZ]


def depolarizing(p):
    """Random X, Y or Z error (p/3 each): shrinks the whole Bloch sphere."""
    return [math.sqrt(1 - p) * I2] + [math.sqrt(p / 3) * g for g in (PX, PY, PZ)]


def amplitude_damping(g):
    """Energy loss |1> -> |0> with probability g: the T1 process."""
    return [
        np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=complex),
        np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex),
    ]


CHANNELS = {
    "bit-flip": bit_flip,
    "phase-flip": phase_flip,
    "depolarizing": depolarizing,
    "amplitude-damping": amplitude_damping,
}


def apply_channel(rho, kraus):
    """rho -> sum_k K rho K†."""
    return sum(k @ rho @ k.conj().T for k in kraus)


def op_on(op, qubit, n):
    """Embed a 1-qubit operator on one qubit of an n-qubit register."""
    # Kronecker product of identities with op in slot `qubit`: the standard
    # way to lift a 1-qubit gate to the full 2^n-dimensional space.
    out = np.array([[1]], dtype=complex)
    for k in range(n):
        out = np.kron(out, op if k == qubit else I2)
    return out


def channel_on(rho, kraus, qubit, n):
    """Apply a 1-qubit channel to one qubit of an n-qubit density matrix."""
    return apply_channel(rho, [op_on(k, qubit, n) for k in kraus])


def pure(theta, phi):
    """cos(t/2)|0> + e^(i phi) sin(t/2)|1>."""
    return np.array([math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)])


def bloch(rho):
    """(x, y, z) = (tr rho X, tr rho Y, tr rho Z); length < 1 means mixed."""
    return tuple(float(np.real(np.trace(rho @ g))) for g in (PX, PY, PZ))


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# The 3-qubit bit-flip code (density-matrix version)
# ----------------------------------------------------------------------------
def encode(psi):
    """a|0> + b|1>  ->  a|000> + b|111>: the logical qubit, as an 8-vector."""
    v = np.zeros(8, dtype=complex)
    v[0], v[7] = psi[0], psi[1]
    return v


def correct(rho):
    """Measure the parities (q0^q1, q1^q2), then undo the implied X flip.

    The syndrome says WHERE an error is without revealing the encoded
    amplitudes, so superpositions survive. Averaged over syndrome outcomes
    this is the channel  rho -> sum_s C_s P_s rho P_s C_s†.
    """
    fix = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}
    out = np.zeros_like(rho)
    for syn, qubit in fix.items():
        # Bit k of basis index i (big-endian): q0 = i>>2, q1 = i>>1, q2 = i.
        diag = [
            1.0 if ((((i >> 2) ^ (i >> 1)) & 1, ((i >> 1) ^ i) & 1) == syn) else 0.0
            for i in range(8)
        ]
        proj = np.diag(diag).astype(complex)
        c = np.eye(8, dtype=complex) if qubit is None else op_on(PX, qubit, 3)
        out += c @ proj @ rho @ proj @ c.conj().T
    return out


def noisy_code_fidelity(psi, kraus, with_correction=True):
    """Encode psi, hit every qubit with the channel, (correct,) compare."""
    logical = encode(psi)
    rho = np.outer(logical, logical.conj())
    # Independent noise on each physical qubit, as in a real device.
    for q in range(3):
        rho = channel_on(rho, kraus, q, 3)
    if with_correction:
        rho = correct(rho)
    return float(np.real(logical.conj() @ rho @ logical))


def bitflip_circuit(error_qubit=None, correct_it=True, extra_error=None):
    """Qiskit circuit: encode, inject X error(s), extract syndrome, correct.

    Qubits 0-2 are data, 3-4 are syndrome ancillas. Correction is coherent
    (Toffolis controlled on the ancillas) so we can use exact statevectors.
    """
    qc = QuantumCircuit(5)
    # ry sets the amplitudes and p adds the relative phase: a generic state.
    qc.ry(THETA, 0)
    qc.p(PHI, 0)  # the state to protect
    qc.cx(0, 1)
    qc.cx(0, 2)  # encode: a|000> + b|111>
    for q in (error_qubit, extra_error):
        if q is not None:
            qc.x(q)  # the "noise"
    qc.cx(0, 3)
    qc.cx(1, 3)  # ancilla 3 = q0 xor q1
    qc.cx(1, 4)
    qc.cx(2, 4)  # ancilla 4 = q1 xor q2
    # Each Toffoli fires on exactly one syndrome pattern; the X gates on the
    # ancillas turn "control on 0" into "control on 1" and are undone after.
    if correct_it:
        qc.x(4)
        qc.ccx(3, 4, 0)  # syndrome (1,0): flip q0
        qc.x(4)
        qc.ccx(3, 4, 1)  # syndrome (1,1): flip q1
        qc.x(3)
        qc.ccx(3, 4, 2)  # syndrome (0,1): flip q2
        qc.x(3)
    return qc


def data_fidelity(qc):
    """Fidelity of qubits 0-2 with the ideal encoded state (ancillas traced)."""
    # Reference: the same encoding on 3 qubits with no noise and no ancillas.
    clean = QuantumCircuit(3)
    clean.ry(THETA, 0)
    clean.p(PHI, 0)
    clean.cx(0, 1)
    clean.cx(0, 2)
    ref = DensityMatrix(Statevector.from_instruction(clean))
    got = partial_trace(Statevector.from_instruction(qc), [3, 4])
    return float(state_fidelity(ref, got))


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
P_GRID = (0.01, 0.05, 0.1, 0.2, 0.4)  # physical error rates to sweep


def check_channels(p, r):
    """Checks 1-3: physical channels, Bloch shrinkage, Qiskit agreement."""
    # A channel that fails sum K†K = I would create or destroy probability,
    # so this is the minimum bar for calling an operation "physical".
    log.info("=== Check 1: channels are trace preserving (physical) ===")
    for name, make in CHANNELS.items():
        kraus = make(p)
        total = sum(k.conj().T @ k for k in kraus)
        r.append(check(np.allclose(total, I2), f"{name}: sum K†K = I"))

    log.info("=== Check 2: Bloch vector shrinks as predicted (p=%.2f) ===", p)
    psi = pure(THETA, PHI)
    rho0 = np.outer(psi, psi.conj())
    x, y, z = bloch(rho0)
    g = p  # damping strength reuses p
    expect = {
        "bit-flip": (x, (1 - 2 * p) * y, (1 - 2 * p) * z),
        "phase-flip": ((1 - 2 * p) * x, (1 - 2 * p) * y, z),
        "depolarizing": tuple((1 - 4 * p / 3) * v for v in (x, y, z)),
        "amplitude-damping": (
            math.sqrt(1 - g) * x,
            math.sqrt(1 - g) * y,
            (1 - g) * z + g,
        ),
    }
    for name, make in CHANNELS.items():
        got = bloch(apply_channel(rho0, make(p)))
        log.info("%-18s Bloch = %s", name, tuple(round(v, 4) for v in got))
        r.append(check(np.allclose(got, expect[name]), f"{name}: matches closed form"))

    log.info("=== Check 3: NumPy channels equal Qiskit's Kraus ===")
    same = True
    for make in CHANNELS.values():
        mine = apply_channel(rho0, make(p))
        theirs = DensityMatrix(psi).evolve(Kraus(make(p))).data
        same &= bool(np.allclose(mine, theirs))
    r.append(check(same, "all four channels agree with qiskit.quantum_info"))


def check_decay(p, r):
    """Check 4: errors accumulate; fidelity decays exponentially."""
    log.info("=== Check 4: repeated noise decays the state ===")
    rho = np.diag([1.0, 0.0]).astype(complex)  # start in |0>
    ok_form = True
    # Depolarizing noise multiplies the Bloch vector by a fixed factor each
    # step, so fidelity relaxes geometrically toward the mixed-state value 1/2.
    for n in range(0, 41):
        # Depolarizing scales the Bloch z by s each step, so F = (1 + s^n)/2.
        want = (1 + (1 - 4 * p / 3) ** n) / 2
        ok_form &= abs(float(np.real(rho[0, 0])) - want) < 1e-12
        if n in (0, 1, 5, 20, 40):
            log.info("n=%2d noisy steps: fidelity with |0> = %.4f", n, want)
        rho = apply_channel(rho, depolarizing(p))
    r.append(check(ok_form, "fidelity = (1 + (1-4p/3)^n)/2 for n=0..40"))
    r.append(
        check(
            abs(float(np.real(rho[0, 0])) - 0.5) < 0.01,
            "fully mixed (F -> 0.5): the quantum info is gone",
        )
    )

    # T1: excited-state population after n damping steps is (1-g)^n.
    rho = np.diag([0.0, 1.0]).astype(complex)
    for _ in range(10):
        rho = apply_channel(rho, amplitude_damping(p))
    t1 = -1 / math.log(1 - p)
    log.info(
        "T1 = %.2f steps; P(1) after 10 steps = %.4f", t1, float(np.real(rho[1, 1]))
    )
    r.append(
        check(
            abs(float(np.real(rho[1, 1])) - math.exp(-10 / t1)) < 1e-12,
            "P(|1>) = exp(-t/T1)",
        )
    )


def check_code(r):
    """Checks 5-6: bit-flip code works for X noise and fails for Z noise."""
    log.info("=== Check 5: bit-flip code, logical error = 3p^2 - 2p^3 ===")
    zero = np.array([1, 0], dtype=complex)
    gen = pure(THETA, PHI)
    xexp = math.sin(THETA) * math.cos(PHI)  # <psi|X|psi>
    ok_formula = ok_better = ok_general = True
    for p in P_GRID:
        fail_p = 3 * p**2 - 2 * p**3
        f0 = noisy_code_fidelity(zero, bit_flip(p))
        ok_formula &= abs(f0 - (1 - fail_p)) < 1e-12
        ok_better &= fail_p < p
        # A failure applies logical X, so F = 1 - P(1 - <X>^2) for general psi.
        ok_general &= (
            abs(noisy_code_fidelity(gen, bit_flip(p)) - (1 - fail_p * (1 - xexp**2)))
            < 1e-12
        )
        log.info("p=%.2f: raw error %.4f -> logical error %.4f", p, p, fail_p)
    r.append(check(ok_formula, "fidelity of |0_L> = 1 - (3p^2 - 2p^3)"))
    r.append(check(ok_better, "encoding beats no encoding for every p < 0.5"))
    r.append(
        check(ok_general, "a generic superposition is protected too (coherence kept)")
    )

    # Z flips leave every basis state unchanged, so the parity checks see
    # nothing. Worse, three qubits give three chances to flip the logical
    # phase. Fixing both error types needs Shor's 9-qubit code.
    log.info("=== Check 6: the same code cannot fix phase flips ===")
    plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
    worse = True
    for p in P_GRID[:3]:
        odd = 3 * p * (1 - p) ** 2 + p**3  # odd number of Z errors
        f = noisy_code_fidelity(plus, phase_flip(p))
        worse &= abs(f - (1 - odd)) < 1e-12 and odd > p
        log.info("p=%.2f: raw phase error %.4f -> encoded %.4f", p, p, odd)
    r.append(check(worse, "Z errors: encoded error 3p(1-p)^2 + p^3 > p (worse!)"))
    r.append(
        check(
            abs(
                noisy_code_fidelity(plus, phase_flip(0.1), False)
                - noisy_code_fidelity(plus, phase_flip(0.1))
            )
            < 1e-12,
            "syndrome never fires on Z errors, so correction does nothing",
        )
    )


def check_circuit(r):
    """Check 7: syndrome-extraction circuit in Qiskit."""
    log.info("=== Check 7: Qiskit circuit with syndrome ancillas ===")
    for q in (None, 0, 1, 2):
        f = data_fidelity(bitflip_circuit(q))
        r.append(
            check(
                abs(f - 1) < 1e-9,
                f"X error on {'nothing' if q is None else f'q{q}'}: "
                f"corrected, F = {f:.6f}",
            )
        )
    f = data_fidelity(bitflip_circuit(0, correct_it=False))
    r.append(check(f < 1e-9, f"X on q0 WITHOUT correction: F = {f:.6f}"))
    # Two errors fool the code: it 'fixes' the wrong qubit and applies a
    # logical X, leaving F = <X>^2 = sin^2(theta) cos^2(phi).
    f2 = data_fidelity(bitflip_circuit(0, extra_error=1))
    want = (math.sin(THETA) * math.cos(PHI)) ** 2
    log.info("two errors: F = %.4f (predicted <X>^2 = %.4f)", f2, want)
    r.append(
        check(
            abs(f2 - want) < 1e-9,
            "distance 3 corrects ONE error; two become a logical X",
        )
    )


def check_montecarlo(shots, seed, r):
    """Check 8: sampled logical error rate versus 3p^2 - 2p^3."""
    log.info("=== Check 8: Monte Carlo, %d shots per p ===", shots)
    # Classical sampling is enough here: X errors on basis states act like
    # independent coin flips, and the syndrome decoder is a majority vote.
    rng = np.random.default_rng(seed)
    for p in (0.02, 0.05, 0.1, 0.2):
        flips = rng.random((shots, 3)) < p
        # Majority vote fails exactly when two or more bits flipped.
        rate = float(np.mean(flips.sum(axis=1) >= 2))
        theory = 3 * p**2 - 2 * p**3
        sigma = math.sqrt(theory * (1 - theory) / shots)
        log.info(
            "p=%.2f: measured %.5f, theory %.5f (+/- %.5f)", p, rate, theory, sigma
        )
        r.append(
            check(abs(rate - theory) < 5 * sigma, f"p={p}: within 5 sigma of theory")
        )


def main(argv=None):
    """Run every Lab 7 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 7: noise and bit-flip code")
    ap.add_argument(
        "--p",
        type=float,
        default=0.1,
        help="channel strength for checks 1-4, in (0, 0.5)",
    )
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not 0 < args.p < 0.5:
        ap.error(f"--p must be in (0, 0.5). Got {args.p}.")
    if args.shots < 1000:
        ap.error(f"--shots must be >= 1000. Got {args.shots}.")
    r = []
    check_channels(args.p, r)
    check_decay(args.p, r)
    check_code(r)
    check_circuit(r)
    check_montecarlo(args.shots, args.seed, r)

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
