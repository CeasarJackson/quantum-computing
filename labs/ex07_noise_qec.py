#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex07_noise_qec.py

Purpose
-------
Lab 7 optional exercises, each as predict-then-verify:
  1. Run the lab at p = 0.3 and p = 0.45: how much does the 3-qubit code
     help when noise is strong? (ratio = 3p - 2p^2)
  2. A Z error injected in the Qiskit circuit: syndrome and fidelity.
  3. The phase-flip code (Hadamards around the bit-flip code): it fixes Z
     noise and fails against X noise.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex07_noise_qec.py

Log files
---------
logs/ex07/ex07_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import lab07_noise_qec as lab07
from applog import get_logger
from exlib import Tally, base_parser

log = get_logger("ex07")

H1 = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
H3 = np.kron(np.kron(H1, H1), H1)  # Hadamard on all three code qubits


def generic_state():
    """The amplitude pair used by Lab 7's circuit (ry theta, then phase phi)."""
    return lab07.pure(lab07.THETA, lab07.PHI)


def raw_infidelity(kraus):
    """Infidelity of ONE unencoded qubit after the channel, generic state."""
    psi = generic_state()
    # pure() returns a ket; the channel needs a density matrix.
    out = lab07.apply_channel(np.outer(psi, psi.conj()), kraus)
    return 1.0 - float(np.real(psi.conj() @ out @ psi))


def ex1(t):
    """Exercise 1: strong noise at p = 0.3 and p = 0.45."""
    t.ask("Exercise 1: how much does the code help at p = 0.3 and p = 0.45?")
    t.predict(
        "logical/raw error = 3p - 2p^2: 0.72 at p=0.3 and 0.945 at"
        " p=0.45, so the code barely helps; it is 'much smaller'"
        " (ratio < 0.5) only for p < (3 - sqrt5)/4 = 0.191"
    )
    psi = generic_state()
    for p in (0.3, 0.45):
        # The full lab must still pass: its checks hold for every p < 0.5.
        # Reusing main() as a test: its exit code is the verdict.
        rc = lab07.main(["--p", str(p), "--shots", "50000"])
        t.check(rc == 0, f"Lab 7 passes end to end at p = {p}")
        # Infidelity of the encoded qubit after noise and correction.
        logical = 1.0 - lab07.noisy_code_fidelity(psi, lab07.bit_flip(p))
        # Infidelity of one bare qubit under the same channel.
        raw = raw_infidelity(lab07.bit_flip(p))
        ratio = logical / raw
        t.note(
            "p=%.2f: raw error %.4f, logical error %.4f, ratio %.4f",
            p,
            raw,
            logical,
            ratio,
        )
        t.check(
            abs(ratio - (3 * p - 2 * p**2)) < 1e-9,
            f"p={p}: ratio equals 3p - 2p^2 = {3 * p - 2 * p ** 2:.4f}",
        )
    t.check(
        abs((3 - math.sqrt(5)) / 4 - 0.1910) < 1e-4,
        "ratio drops below 0.5 only for p < 0.191",
    )
    # Ratio < 1 for every p < 0.5, but it creeps toward 1: encoding costs
    # 3x the qubits and 8 gates, and near p = 0.5 buys almost nothing.
    t.check(
        all(3 * p - 2 * p**2 < 1 for p in np.linspace(0.01, 0.49, 49)),
        "the code never hurts below p = 0.5, but the benefit vanishes",
    )


def z_error_circuit(z_qubit, correct_it=True):
    """Lab 7's syndrome circuit, but with a Z (phase) error instead of X."""
    qc = QuantumCircuit(5)
    qc.ry(lab07.THETA, 0)
    qc.p(lab07.PHI, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)  # encode
    qc.z(z_qubit)  # the error under test
    qc.cx(0, 3)
    qc.cx(1, 3)  # ancilla 3 = q0 xor q1
    qc.cx(1, 4)
    qc.cx(2, 4)  # ancilla 4 = q1 xor q2
    if correct_it:
        # The same Toffoli corrections as Lab 7; they will not fire.
        qc.x(4)
        qc.ccx(3, 4, 0)
        qc.x(4)
        qc.ccx(3, 4, 1)
        qc.x(3)
        qc.ccx(3, 4, 2)
        qc.x(3)
    return qc


def ex2(t):
    """Exercise 2: Z error through the bit-flip circuit."""
    t.ask("Exercise 2: a Z error in the syndrome circuit")
    t.predict(
        "Z leaves every basis state alone, so both parities stay 0:"
        " syndrome (0,0) with certainty, no correction fires, and the"
        " data suffers a logical Z with fidelity cos^2(theta)"
    )
    # A logical Z on this state leaves overlap cos^2(theta).
    want = math.cos(lab07.THETA) ** 2
    for q in (0, 1, 2):
        qc = z_error_circuit(q)
        # Ancillas are qubits 3 and 4; index 0 of this vector is syndrome 00.
        # Marginal over the two ancilla qubits gives the syndrome probabilities.
        syn = Statevector.from_instruction(qc).probabilities([3, 4])
        f = lab07.data_fidelity(qc)
        f_off = lab07.data_fidelity(z_error_circuit(q, correct_it=False))
        t.note(
            "Z on q%d: P(syndrome 00) = %.4f, fidelity %.4f (no-correct %.4f),"
            " cos^2(theta) = %.4f",
            q,
            syn[0],
            f,
            f_off,
            want,
        )
        t.check(abs(syn[0] - 1) < 1e-9, f"Z on q{q}: syndrome is (0,0)")
        t.check(
            abs(f - want) < 1e-9 and abs(f - f_off) < 1e-9,
            f"Z on q{q}: logical Z, F = cos^2(theta), correction inert",
        )
    # Contrast: the same position with an X error IS fixed by the circuit.
    t.check(
        abs(lab07.data_fidelity(lab07.bitflip_circuit(0)) - 1) < 1e-9,
        "an X error on q0 is fully corrected (the contrast case)",
    )


def phase_code_fidelity(psi, kraus, with_correction=True):
    """Phase-flip code: encode, H on all, noise, H on all, bit-flip correct."""
    # Conjugating by H swaps X and Z, so the repetition code that protects
    # against X protects against Z when wrapped in Hadamards.
    base = lab07.encode(psi)
    logical = H3 @ base
    rho = np.outer(logical, logical.conj())
    for q in range(3):
        rho = lab07.channel_on(rho, kraus, q, 3)
    rho = H3 @ rho @ H3.conj().T  # back to the bit-flip frame
    if with_correction:
        rho = lab07.correct(rho)
    return float(np.real(base.conj() @ rho @ base))


def ex3(t):
    """Exercise 3: the phase-flip code."""
    t.ask("Exercise 3: the phase-flip code, against Z noise and X noise")
    t.predict(
        "vs Z noise: logical error 3p^2 - 2p^3 < p (it works, the"
        " mirror image of Lab 7). vs X noise: errors look like Z in the"
        " rotated frame, go undetected, and odd counts give a logical"
        " flip: error 3p(1-p)^2 + p^3 > p, and correction is inert"
    )
    psi = generic_state()
    # Expectation values of the state set the size of each failure.
    zexp = math.cos(lab07.THETA)  # <psi|Z|psi>
    xexp = math.sin(lab07.THETA) * math.cos(lab07.PHI)
    # Small and moderate p, where the code should clearly work.
    for p in (0.01, 0.05, 0.1, 0.2):
        fail = 3 * p**2 - 2 * p**3
        f_z = phase_code_fidelity(psi, lab07.phase_flip(p))
        t.note(
            "p=%.2f vs Z noise: F=%.6f, predicted %.6f",
            p,
            f_z,
            1 - fail * (1 - xexp**2),
        )
        t.check(
            abs(f_z - (1 - fail * (1 - xexp**2))) < 1e-12 and fail < p,
            f"p={p}: Z noise suppressed to 3p^2 - 2p^3 = {fail:.5f} < p",
        )
    for p in (0.01, 0.05, 0.1):
        odd = 3 * p * (1 - p) ** 2 + p**3
        f_x = phase_code_fidelity(psi, lab07.bit_flip(p))
        f_x_off = phase_code_fidelity(psi, lab07.bit_flip(p), False)
        t.note("p=%.2f vs X noise: F=%.6f (uncorrected %.6f)", p, f_x, f_x_off)
        t.check(
            abs(f_x - (1 - odd * (1 - zexp**2))) < 1e-12 and odd > p,
            f"p={p}: X noise NOT suppressed: error {odd:.5f} > p",
        )
        t.check(
            abs(f_x - f_x_off) < 1e-12,
            f"p={p}: correction does nothing against X noise",
        )


def main(argv=None):
    """Run all three Lab 7 exercises; return the exit code."""
    base_parser("Lab 7 exercises").parse_args(argv)
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
