#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab03_qiskit.py

Purpose
-------
Lab 3 rebuilds the Bell and GHZ circuits in Qiskit and proves, number for
number, that the library agrees with the simulator you wrote in Labs 1-2.
Qiskit is only worth trusting once you can show it reproduces known physics.
The lab also pins down the one trap that bites everyone: Qiskit orders qubits
LITTLE-endian (qubit 0 is the RIGHTMOST bit), while Lab 2 used big-endian.

Everything runs locally with Qiskit's reference primitives (no hardware, no
account, no network). A circuit written this way later runs on a real device
by swapping the sampler for a hardware backend.

Security
--------
Handles no secrets and makes no network calls. A real-hardware run would need
an API token; when we get there it will be read from an environment variable
and never logged, never placed in argv, never written to a file.

Usage
-----
    python3 labs/lab03_qiskit.py
    python3 labs/lab03_qiskit.py --shots 50000 --seed 7

Log files
---------
logs/lab03/lab03_<YYYYmmdd_HHMMSS>.log

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
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, entropy, partial_trace

from applog import get_logger

# Reuse the simulator from Lab 2 as the reference ("ground truth") answer.
# Importing it also opens a lab02 log file; that side effect is harmless.
from lab02_entanglement import H, apply_1q, apply_cnot, zero_state

log = get_logger("lab03")


def bell_circuit(measure=False):
    """Bell pair: H on qubit 0, then CNOT(0 -> 1)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    if measure:
        qc.measure_all()  # adds a classical register named 'meas'
    return qc


def ghz_circuit(n, measure=False):
    """n-qubit GHZ: H on qubit 0, then a CNOT chain 0->1->...->n-1."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    if measure:
        qc.measure_all()
    return qc


def run_counts(qc, shots, seed):
    """Sample a measured circuit with the local reference sampler."""
    # A fixed seed makes the sampling reproducible so failures can be replayed.
    sampler = StatevectorSampler(seed=seed)
    result = sampler.run([qc], shots=shots).result()
    return result[0].data.meas.get_counts()


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def main(argv=None):
    """Run every Lab 3 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 3: Qiskit vs. your simulator")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.shots < 1:
        ap.error(f"--shots must be >= 1. Got {args.shots}.")
    r = []

    # Check 1: the circuit diagram is the program. Reading it is a core skill.
    log.info("=== Check 1: the Bell circuit as Qiskit draws it ===")
    for line in str(bell_circuit().draw("text")).splitlines():
        log.info("%s", line)

    # Check 2: Qiskit's statevector must equal Lab 2's. The Bell state is
    # symmetric under swapping the two qubits, so endianness cannot matter yet.
    log.info("=== Check 2: Qiskit statevector == your Lab 2 simulator ===")
    qk = Statevector.from_instruction(bell_circuit())
    mine = apply_cnot(apply_1q(zero_state(2), H, 0, 2), 0, 1, 2)
    log.info("Qiskit amplitudes = %s", np.round(qk.data, 4))
    r.append(check(np.allclose(qk.data, mine), "Bell amplitudes match Lab 2"))

    # Check 3: an asymmetric state exposes the endianness difference.
    log.info("=== Check 3: endianness (the classic trap) ===")
    x0 = QuantumCircuit(2)
    x0.x(0)  # flip qubit 0 only
    qk_x = Statevector.from_instruction(x0)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    mine_x = apply_1q(zero_state(2), X, 0, 2)  # Lab 2: big-endian
    # Convert NumPy scalars to plain Python so the log reads cleanly.
    qk_probs = {str(k): float(v) for k, v in qk_x.probabilities_dict().items()}
    log.info("Qiskit says X on qubit 0 gives %s", qk_probs)
    log.info(
        "Lab 2 says the same gate gives index %d (bitstring '10')",
        int(np.argmax(np.abs(mine_x))),
    )
    r.append(
        check(
            list(qk_probs) == ["01"],
            "Qiskit writes qubit 0 as the RIGHTMOST bit ('01')",
        )
    )
    # reverse_qargs() flips the qubit order, converting between conventions.
    r.append(
        check(
            np.allclose(qk_x.reverse_qargs().data, mine_x),
            "reversing qubit order converts Qiskit to Lab 2 exactly",
        )
    )

    # Check 4: sampled counts show the correlation, as in Lab 2.
    log.info("=== Check 4: sampled Bell counts ===")
    counts = run_counts(bell_circuit(measure=True), args.shots, args.seed)
    log.info("counts = %s", dict(counts))
    r.append(check(set(counts) <= {"00", "11"}, "only 00 and 11 occur"))
    p00 = counts.get("00", 0) / args.shots
    r.append(check(abs(p00 - 0.5) < 0.02, f"P(00) = {p00:.4f} ~ 0.5"))

    # Check 5: Qiskit's own entanglement tools agree with the Lab 2 math.
    log.info("=== Check 5: pure whole, maximally mixed part ===")
    rho0 = partial_trace(qk, [1])  # trace out qubit 1
    purity = float(np.real(rho0.purity()))  # purity is real by definition
    r.append(
        check(
            abs(purity - 0.5) < 1e-9,
            f"purity of one qubit = {purity:.3f} (1.0 would be pure)",
        )
    )
    r.append(
        check(abs(entropy(rho0, base=2) - 1.0) < 1e-9, "entanglement entropy = 1 bit")
    )

    # Check 6: scale up. A 4-qubit GHZ state is just a longer CNOT chain.
    log.info("=== Check 6: 4-qubit GHZ ===")
    g = run_counts(ghz_circuit(4, measure=True), args.shots, args.seed)
    log.info("counts = %s", dict(g))
    r.append(check(set(g) <= {"0000", "1111"}, "only 0000 and 1111 occur"))

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
