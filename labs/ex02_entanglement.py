#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex02_entanglement.py

Purpose
-------
Lab 2 optional exercises, each as predict-then-verify:
  1. All four Bell states from H, X, Z and CNOT only.
  2. A reversed CNOT (control = qubit 1) on H(x)I|00>.
  3. GHZ entropy across the cuts 0|12 and 01|2.
  4. A classical 50/50 mix of |00> and |11> versus the Bell state, and a
     measurement (X(x)X, then CHSH) that tells them apart.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex02_entanglement.py

Log files
---------
logs/ex02/ex02_<YYYYmmdd_HHMMSS>.log

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
from lab02_entanglement import (
    H,
    apply_1q,
    apply_cnot,
    entropy_bits,
    schmidt_rank,
    zero_state,
)

log = get_logger("ex02")

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bell(a, b):
    """Bell state from input bits (a, b): X flips, then H on 0, then CNOT."""
    psi = zero_state(2)
    # X on qubit 0 makes the relative sign negative after H; X on qubit 1
    # swaps which correlation (equal or opposite) the pair carries.
    if a:
        psi = apply_1q(psi, X, 0, 2)
    if b:
        psi = apply_1q(psi, X, 1, 2)
    psi = apply_1q(psi, H, 0, 2)
    return apply_cnot(psi, 0, 1, 2)


def ex1(t):
    """Exercise 1: build and test all four Bell states."""
    t.ask("Exercise 1: the four Bell states")
    # Rule: X on qubit 0 gives the minus sign, X on qubit 1 gives opposite bits.
    t.predict(
        "(0,0)=Phi+ (|00>+|11>), (1,0)=Phi-, (0,1)=Psi+, (1,1)=Psi-"
        "; each has entropy 1 bit; the four are orthogonal"
    )
    s = 1 / np.sqrt(2)
    known = {
        (0, 0): [s, 0, 0, s],
        (1, 0): [s, 0, 0, -s],
        (0, 1): [0, s, s, 0],
        (1, 1): [0, s, -s, 0],
    }
    states = {}
    for key, want in known.items():
        psi = bell(*key)
        states[key] = psi
        _, sv = schmidt_rank(psi, 1, 2)
        ent = entropy_bits(sv)
        t.note("%s -> %s entropy %.3f", key, np.round(psi.real, 3), ent)
        t.check(
            np.allclose(psi, want) and abs(ent - 1) < 1e-9,
            f"inputs {key} give the expected Bell state, entropy 1 bit",
        )
    # Gram matrix of inner products: identity means orthonormal,
    # which is what lets the four Bell states serve as a measurement basis.
    gram = np.array([[np.vdot(u, v) for v in states.values()] for u in states.values()])
    t.check(np.allclose(gram, np.eye(4)), "the four Bell states are orthonormal")
    # Z also makes the minus sign: H, Z, CNOT builds Phi- without any X.
    psi = apply_cnot(apply_1q(apply_1q(zero_state(2), H, 0, 2), Z, 0, 2), 0, 1, 2)
    t.check(np.allclose(psi, states[(1, 0)]), "H, Z, CNOT also gives Phi-")


def ex2(t):
    """Exercise 2: reverse the CNOT."""
    t.ask("Exercise 2: CNOT with control = qubit 1")
    t.predict(
        "qubit 1 is |0>, so the control never fires: the result stays"
        " |+>|0>, a product state with entropy 0"
    )
    psi = apply_1q(zero_state(2), H, 0, 2)
    out = apply_cnot(psi, 1, 0, 2)
    # Schmidt rank 1 means the state still factors into two qubits.
    rank, sv = schmidt_rank(out, 1, 2)
    t.note("amplitudes %s, Schmidt rank %d", np.round(out.real, 3), rank)
    t.check(
        np.allclose(out, psi) and rank == 1 and entropy_bits(sv) < 1e-9,
        "reversed CNOT does nothing here: still unentangled",
    )
    # The fix is to put the superposition on the control qubit instead.
    both = apply_cnot(apply_1q(zero_state(2), H, 1, 2), 1, 0, 2)
    t.check(
        schmidt_rank(both, 1, 2)[0] == 2,
        "H on qubit 1 first, then CNOT(1->0), entangles again",
    )


def ex3(t):
    """Exercise 3: GHZ entropy across two cuts."""
    t.ask("Exercise 3: GHZ entropy for the cuts 0|12 and 01|2")
    t.predict(
        "GHZ = (|000>+|111>)/sqrt2: either cut separates the two"
        " branches, so Schmidt rank 2 and entropy 1 bit for both"
    )
    psi = apply_cnot(apply_cnot(apply_1q(zero_state(3), H, 0, 3), 0, 1, 3), 1, 2, 3)
    # Both cuts are tested because entanglement depends on where you cut,
    # even though GHZ happens to look the same from every angle.
    for n_a, name in ((1, "0|12"), (2, "01|2")):
        rank, sv = schmidt_rank(psi, n_a, 3)
        ent = entropy_bits(sv)
        t.note("cut %s: rank %d, entropy %.3f bits", name, rank, ent)
        t.check(rank == 2 and abs(ent - 1) < 1e-9, f"cut {name}: rank 2, entropy 1 bit")


def expect(rho, a, b):
    """<A (x) B> for a two-qubit density matrix."""
    return float(np.real(np.trace(rho @ np.kron(a, b))))


def chsh(rho):
    """CHSH value with the angles that are optimal for Phi+."""
    # Alice: Z or X. Bob: (Z+X)/sqrt2 or (Z-X)/sqrt2, rotated 45 degrees.
    b0, b1 = (Z + X) / np.sqrt(2), (Z - X) / np.sqrt(2)
    return (
        expect(rho, Z, b0)
        + expect(rho, Z, b1)
        + expect(rho, X, b0)
        - expect(rho, X, b1)
    )


def ex4(t):
    """Exercise 4: a classical mixture versus the Bell state."""
    t.ask("Exercise 4: classical 50/50 mix versus the Bell state")
    t.predict(
        "both give perfectly correlated Z outcomes; X(x)X separates"
        " them (+1 for Bell, 0 for the mix); CHSH is 2*sqrt2 for"
        " Bell and at most 2 for any classical mix"
    )
    phi = bell(0, 0)
    rho_bell = np.outer(phi, phi.conj())
    # A mixture is a probability-weighted SUM of projectors, not a sum of
    # amplitudes: there are no off-diagonal (coherence) terms.
    rho_mix = np.zeros((4, 4), dtype=complex)
    rho_mix[0, 0] = rho_mix[3, 3] = 0.5
    t.check(
        abs(expect(rho_bell, Z, Z) - 1) < 1e-9
        and abs(expect(rho_mix, Z, Z) - 1) < 1e-9,
        "<ZZ> = 1 for both: Z-basis data cannot tell them apart",
    )
    xx_b, xx_m = expect(rho_bell, X, X), expect(rho_mix, X, X)
    t.note("<XX>: Bell %.3f, mix %.3f", xx_b, xx_m)
    t.check(
        abs(xx_b - 1) < 1e-9 and abs(xx_m) < 1e-9,
        "<XX> = +1 for Bell and 0 for the mix: the distinguishing test",
    )
    # Same test as sampled data: H on both qubits, then read the Z bits.
    # H on both qubits maps X-basis outcomes onto Z-basis bits,
    # so ordinary readout can see the X(x)X correlation.
    hh = np.kron(H, H)
    for name, rho in (("Bell", rho_bell), ("mix", rho_mix)):
        p = np.real(np.diag(hh @ rho @ hh.conj().T))
        t.note("%s after H(x)H: P(00,01,10,11) = %s", name, np.round(p, 3))
    s_b, s_m = chsh(rho_bell), chsh(rho_mix)
    t.note("CHSH: Bell %.4f (Tsirelson 2.8284), mix %.4f (classical <= 2)", s_b, s_m)
    t.check(abs(s_b - 2 * np.sqrt(2)) < 1e-9, "Bell reaches CHSH = 2*sqrt2")
    t.check(abs(s_m) <= 2 + 1e-9, "the classical mix stays within |CHSH| <= 2")


def main(argv=None):
    """Run all four Lab 2 exercises; return the exit code."""
    base_parser("Lab 2 exercises").parse_args(argv)
    t = Tally(log)
    ex1(t)
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
