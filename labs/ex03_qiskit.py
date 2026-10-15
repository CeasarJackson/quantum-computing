#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex03_qiskit.py

Purpose
-------
Lab 3 optional exercises, each as predict-then-verify:
  1. The four Bell states in Qiskit, checked against Lab 2 (reverse_qargs).
  2. Predict the bitstring for x(0) and x(2) on three qubits.
  3. Draw the GHZ circuit and name the state it makes.
  4. Transpile the Bell circuit to {rz, sx, cx}: what does H become?

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex03_qiskit.py

Log files
---------
logs/ex03/ex03_<YYYYmmdd_HHMMSS>.log

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
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

from applog import get_logger
from ex02_entanglement import bell
from exlib import Tally, base_parser
from lab03_qiskit import bell_circuit, ghz_circuit

log = get_logger("ex03")


def ex1(t):
    """Exercise 1: all Bell states in Qiskit, compared with Lab 2."""
    t.ask("Exercise 1: four Bell states in Qiskit vs Lab 2")
    t.predict(
        "same gates, but Qiskit is little-endian: reversing the qubit"
        " order of Qiskit's vector must equal Lab 2's vector exactly"
    )
    # Looping over all four (a, b) pairs covers every Bell state with one recipe.
    for a in (0, 1):
        for b in (0, 1):
            qc = QuantumCircuit(2)
            # Same recipe as Lab 2: X prepares the input bits, H then CNOT.
            if a:
                qc.x(0)
            if b:
                qc.x(1)
            qc.h(0)
            qc.cx(0, 1)
            # Statevector gives exact amplitudes, so no sampling noise can hide a bug.
            qk = Statevector.from_instruction(qc)
            t.note("inputs (%d,%d): Qiskit %s", a, b, np.round(qk.data.real, 3))
            t.check(
                np.allclose(qk.reverse_qargs().data, bell(a, b)),
                f"inputs ({a},{b}): reverse_qargs() matches Lab 2",
            )
    # Without the reversal, only the endian-symmetric states agree. Psi-
    # (inputs 1,1) has an antisymmetric sign pattern, so it exposes the trap.
    qc = QuantumCircuit(2)
    qc.x([0, 1])
    qc.h(0)
    qc.cx(0, 1)
    raw = Statevector.from_instruction(qc).data
    t.check(
        not np.allclose(raw, bell(1, 1)),
        "WITHOUT reverse_qargs, Psi- disagrees with Lab 2 (the trap)",
    )


def ex2(t):
    """Exercise 2: predict the bitstring from X placement."""
    t.ask("Exercise 2: x(0) and x(2) on three qubits")
    t.predict(
        "Qiskit prints qubit 0 as the RIGHTMOST bit: x(0) -> '001'," " x(2) -> '100'"
    )
    # Qubit 0 is the rightmost character of the printed bitstring (little-endian).
    for q, want in ((0, "001"), (2, "100")):
        qc = QuantumCircuit(3)
        qc.x(q)
        got = [str(k) for k in Statevector.from_instruction(qc).probabilities_dict()]
        t.note("x(%d) -> %s", q, got)
        t.check(got == [want], f"x({q}) gives '{want}'")


def ex3(t):
    """Exercise 3: draw GHZ and identify it."""
    t.ask("Exercise 3: the 3-qubit GHZ circuit")
    t.predict("H on qubit 0 then a CNOT chain gives (|000>+|111>)/sqrt2")
    # Reading the drawn circuit is a skill: one H, then a CNOT chain.
    qc = ghz_circuit(3)
    for line in str(qc.draw("text")).splitlines():
        log.info("%s", line)
    # Only two outcomes carry probability if the state is GHZ.
    probs = Statevector.from_instruction(qc).probabilities_dict()
    t.note("probabilities = %s", {str(k): round(float(v), 3) for k, v in probs.items()})
    t.check(
        set(probs) == {"000", "111"}
        and all(abs(v - 0.5) < 1e-9 for v in probs.values()),
        "GHZ state: 50% '000', 50% '111', nothing else",
    )


def ex4(t):
    """Exercise 4: transpile to the {rz, sx, cx} basis."""
    t.ask("Exercise 4: transpile to rz, sx, cx at optimization_level=0")
    t.predict(
        "H is not native, so it becomes rz(pi/2) sx rz(pi/2) (equal to"
        " H up to a global phase); the CNOT is already native"
    )
    # Keep the untouched circuit to compare the compiled one against.
    src = bell_circuit()
    out = transpile(src, basis_gates=["rz", "sx", "cx"], optimization_level=0)
    for line in str(out.draw("text")).splitlines():
        log.info("%s", line)
    # count_ops is the quickest way to see what the compiler did.
    ops = out.count_ops()
    t.note("gate counts: %s", dict(ops))
    t.check(set(ops) <= {"rz", "sx", "cx"}, "only rz, sx, cx remain")
    t.check(
        ops.get("cx") == 1 and ops.get("sx") == 1 and ops.get("rz") == 2,
        "H became rz, sx, rz (2 rz + 1 sx), CNOT kept",
    )
    # Qiskit stores the global phase on the circuit instead of dropping it,
    # so the two operators match EXACTLY once that metadata is included.
    t.note(
        "global phase carried by the circuit = %.4f (pi/4 = %.4f)",
        float(out.global_phase),
        np.pi / 4,
    )
    t.check(
        abs(float(out.global_phase) - np.pi / 4) < 1e-9,
        "the decomposition of H leaves a global phase of pi/4",
    )
    t.check(
        np.allclose(Operator(src).data, Operator(out).data),
        "transpiled circuit equals the original exactly, phase included",
    )


def main(argv=None):
    """Run all four Lab 3 exercises; return the exit code."""
    base_parser("Lab 3 exercises").parse_args(argv)
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
