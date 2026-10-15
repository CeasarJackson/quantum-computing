#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab10_capstone_shor.py

Purpose
-------
Capstone. Factor 15 with Shor's algorithm as a GATE-LEVEL circuit on a noisy,
simulated 7-qubit device, using everything from Labs 1-9:

  Lab 5/6  QFT, phase estimation, order finding, continued fractions
  Lab 7    noise: errors accumulate with every CNOT
  Lab 9    transpilation to native gates and coupled qubits, readout-error
           mitigation, zero-noise extrapolation

Lab 6 used dense 16x16 matrices for "multiply by a mod N". A real chip has none
of those, so here the multiplier is built from controlled-SWAP and CNOT gates.
For N=15 this is exact: multiplying by 2 mod 15 rotates the 4 bits of y left by
one place, so x2^k are bit rotations and x7 = NOT(rotate by 3).

HONESTY NOTE: this circuit is "compiled": it hard-codes a=7 and drops the
counting qubit whose control power is the identity because a^4 = 1 (mod 15).
That uses prior knowledge of the answer (the order is 4), so it demonstrates
the machinery and the noise cost, NOT a quantum speedup. Smolin, Smith and
Vargo ("Pretending to factor large numbers on a quantum computer") explain why
such demonstrations prove little about scaling.

The lab verifies:
  1. The gate-level circuit equals the dense Lab 6 unitary (exact).
  2. Ideal output: peaks at m = 0, 2, 4, 6 (each 0.25); success probability 0.5.
  3. Compilation to the device's native gates and coupling map.
  4. The noisy run keeps signal above the 0.25 'pure garbage' floor but loses
     to the ideal 0.5.
  5. Readout mitigation does no harm; folding + extrapolation recovers part
     of the gate-error loss.
  6. Repeat-until-success: the average number of runs matches 1 / p, and every
     reported answer is verified classically (3 x 5 = 15).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab10_capstone_shor.py
    python3 labs/lab10_capstone_shor.py --shots 400000 --trials 5000 --seed 7

Log files
---------
logs/lab10/lab10_<YYYYmmdd_HHMMSS>.log

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
from qiskit.circuit.library import UnitaryGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from applog import get_logger
from lab05_qft_qpe import qft_circuit
from lab06_shor import factors_from_measurement, modmult_matrix
from lab09_hardware_workflow import (
    BACKEND_SEED,
    BASIS,
    NATIVE_OPS,
    fold_cx,
    probs_from_counts,
    to_hardware,
)

log = get_logger("lab10")

# Base a=7 is coprime to 15 and has order 4, so r is even and a^(r/2) = 4 is
# not -1 mod 15: it is a 'good' base that can succeed (compare a=14 in Lab 6).
N, A = 15, 7  # factor 15 using base a = 7 (order 4)
# Three counting bits resolve phases in steps of 1/8, enough for k/4 exactly.
T = 3  # counting qubits: outcomes m = 0..7
# 15 needs 4 bits, so the target register holds every residue y in 0..15.
W = 4  # target qubits: hold y in 0..15
# IBM's 7-qubit 'H' layout: qubit 1 and qubit 5 each touch three neighbours.
H_TOPOLOGY = [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)]


def make_device():
    """Simulated 7-qubit chip: native gates and calibration like Lab 9."""
    # Same seed as Lab 9 so the calibration 'day' is comparable, but a larger
    # chip: the algorithm needs 3 counting + 4 target qubits.
    return GenericBackendV2(
        num_qubits=7, coupling_map=H_TOPOLOGY, basis_gates=BASIS, seed=BACKEND_SEED
    )


def controlled_rotations(qc, ctl, tq, left_by):
    """Controlled bit rotation of the target, built from controlled SWAPs.

    Multiplying y by 2 (mod 15) moves bit k to bit k+1 and bit 3 to bit 0.
    A rotation by one place is 3 adjacent swaps; by two places is 2 swaps of
    opposite bits; rotating left by three equals rotating right by one.
    """
    # cswap(c, a, b) swaps qubits a and b only when the control c is 1, which
    # makes the whole rotation conditional on that counting qubit.
    if left_by == 2:
        qc.cswap(ctl, tq[0], tq[2])
        qc.cswap(ctl, tq[1], tq[3])
    elif left_by == 3:  # = rotate right by one
        for a, b in ((1, 0), (2, 1), (3, 2)):
            qc.cswap(ctl, tq[a], tq[b])
    else:
        raise ValueError(f"unsupported rotation {left_by}")


def shor_15_compiled(measure=True):
    """Gate-level order finding for a=7, N=15 with 3 counting qubits.

    Counting qubit j controls multiplication by 7^(2^j) mod 15:
      j=0: x7 = NOT after rotate-left-3   (3 cswaps + 4 cnots)
      j=1: x4 = rotate-left-2             (2 cswaps)
      j=2: x1 = identity, so it needs no gates at all (the 'compiled' step).
    """
    qc = QuantumCircuit(T + W, T if measure else 0)
    tq = list(range(T, T + W))
    qc.x(tq[0])  # target register starts at y = 1
    qc.h(range(T))
    # Order matters: x7 = NOT(rotate-left-3), so rotate first, then flip.
    controlled_rotations(qc, 0, tq, 3)
    for t in tq:  # NOT on all 4 bits, controlled
        qc.cx(0, t)
    controlled_rotations(qc, 1, tq, 2)
    qc.compose(qft_circuit(T, inverse=True), qubits=list(range(T)), inplace=True)
    if measure:
        qc.measure(range(T), range(T))
    return qc


def shor_15_dense():
    """The Lab 6 version: each controlled multiplier is a dense unitary."""
    # Same layout as the compiled circuit, so the two are comparable gate for
    # gate; only the multiplier implementation differs.
    qc = QuantumCircuit(T + W)
    tq = list(range(T, T + W))
    qc.x(tq[0])
    qc.h(range(T))
    for j in range(T):
        gate = UnitaryGate(modmult_matrix(pow(A, 2**j, N), N, W))
        qc.append(gate.control(1), [j] + tq)
    qc.compose(qft_circuit(T, inverse=True), qubits=list(range(T)), inplace=True)
    return qc


def success_set():
    """Counting outcomes m whose post-processing yields factors of 15."""
    return [m for m in range(2**T) if factors_from_measurement(m, T, A, N)]


def success_prob(probs):
    """Probability that one run produces factors."""
    return float(sum(probs[m] for m in success_set()))


def run_probs(sim, tqc, shots):
    """Run a transpiled circuit; return the 2^T outcome probabilities."""
    counts = sim.run(tqc, shots=shots).result().get_counts()
    return probs_from_counts(counts, T)


def measured_physical_qubits(tqc):
    """Physical qubit feeding each classical bit, found from the measures."""
    # The router may place counting qubits anywhere, so read where each ended
    # up from the measure instructions rather than assuming 0, 1, 2.
    pairs = [
        (tqc.find_bit(i.qubits[0]).index, tqc.find_bit(i.clbits[0]).index)
        for i in tqc.data
        if i.operation.name == "measure"
    ]
    return [q for q, _ in sorted(pairs, key=lambda x: x[1])]


def readout_matrix(backend, sim, phys, shots):
    """A[i, j] = P(read i | prepared j) on exactly the measured qubits."""
    # Readout error differs per qubit, so calibrate exactly the qubits used.
    mat = np.zeros((2**T, 2**T))
    for j in range(2**T):
        qc = QuantumCircuit(backend.num_qubits, T)
        for k in range(T):
            if (j >> k) & 1:  # bit k of j sets the k-th qubit
                qc.x(phys[k])
        for k in range(T):
            qc.measure(phys[k], k)
        # Identity layout and level 0: keep every gate on the qubit we meant.
        tqc = to_hardware(backend, qc, layout=list(range(backend.num_qubits)), level=0)
        mat[:, j] = run_probs(sim, tqc, shots)
    return mat


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def check_circuit(r):
    """Checks 1-2: the gate-level circuit is right before any noise."""
    log.info("=== Check 1: gate-level circuit equals the dense Lab 6 unitary ===")
    # Exact statevectors (no sampling) so any mismatch is a real bug, not noise.
    comp = Statevector.from_instruction(shor_15_compiled(False)).probabilities(
        qargs=list(range(T))
    )
    dense = Statevector.from_instruction(shor_15_dense()).probabilities(
        qargs=list(range(T))
    )
    gap = float(np.abs(comp - dense).max())
    log.info("largest probability difference = %.2e", gap)
    r.append(
        check(
            gap < 1e-9,
            "compiled and dense circuits give the same "
            "counting-register distribution",
        )
    )

    log.info("=== Check 2: ideal output ===")
    log.info("ideal P(m) = %s", np.round(comp, 4).tolist())
    r.append(
        check(
            np.allclose(comp, [0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0]),
            "peaks at m = 0, 2, 4, 6 with 0.25 each (phases k/4)",
        )
    )
    # m=0 and m=4 reveal nothing (r=1 or 2); m=2 and m=6 give r=4 -> factors.
    r.append(
        check(
            abs(success_prob(comp) - 0.5) < 1e-9,
            "ideal success probability is 0.5 (m=2 and m=6 succeed)",
        )
    )
    return comp


def check_compile(backend, r):
    """Check 3: native gates, coupled qubits, and optimization level."""
    log.info("=== Check 3: compile to the 7-qubit device ===")
    pairs = {tuple(e) for e in backend.coupling_map.get_edges()}
    stats = {}
    # Compare two optimization levels: the router's choices change CNOT count.
    for level in (1, 3):
        tqc = to_hardware(backend, shor_15_compiled(), level=level)
        stats[level] = (tqc.count_ops().get("cx", 0), tqc.depth(), tqc)
        log.info("level %d: %3d cx, depth %3d", level, *stats[level][:2])
    tqc = stats[3][2]
    # Same coupling test as Lab 9: direction is fixed by the transpiler, so
    # accept either order of a pair.
    ok_pairs = True
    for inst in tqc.data:
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
            a, b = (tqc.find_bit(q).index for q in inst.qubits)
            ok_pairs &= (a, b) in pairs or (b, a) in pairs
    r.append(
        check(
            set(tqc.count_ops()) <= NATIVE_OPS, "only native gates after transpilation"
        )
    )
    r.append(check(ok_pairs, "every 2-qubit gate acts on a coupled pair"))
    # Each controlled-SWAP costs about 8 CNOTs, so even this toy needs dozens.
    r.append(
        check(
            stats[3][0] <= stats[1][0] and stats[3][0] > 40,
            f"level 3 uses {stats[3][0]} CNOTs (<= level 1's "
            f"{stats[1][0]}); even 15 needs dozens",
        )
    )
    return tqc


def check_noisy(sim, tqc, shots, r):
    """Check 4: the signal survives noise, but is damaged."""
    log.info("=== Check 4: noisy run, %d shots ===", shots)
    # Density-matrix simulation with the device's noise model: gate errors,
    # thermal relaxation and readout error all included.
    raw = run_probs(sim, tqc, shots)
    p = success_prob(raw)
    sigma = math.sqrt(p * (1 - p) / shots)
    log.info("noisy P(m) = %s", np.round(raw, 4).tolist())
    log.info(
        "success rate = %.4f +/- %.4f (ideal 0.5000, garbage floor " "0.2500)", p, sigma
    )
    # Uniform random output would hit the two good outcomes 2 times in 8.
    r.append(
        check(
            p > 0.25 + 5 * sigma,
            "noisy success is >5 sigma above the 0.25 garbage floor",
        )
    )
    r.append(check(p + 5 * sigma < 0.5, "noise measurably reduces success"))
    return raw, p, sigma


def check_mitigation(backend, sim, tqc, raw, p_raw, sigma, shots, r):
    """Checks 5-6: readout mitigation and zero-noise extrapolation."""
    log.info("=== Check 5: readout-error mitigation ===")
    phys = measured_physical_qubits(tqc)
    log.info("counting qubits were measured on physical qubits %s", phys)
    mat = readout_matrix(backend, sim, phys, shots)
    r.append(
        check(
            np.allclose(mat.sum(axis=0), 1),
            "each column of the calibration matrix sums to 1",
        )
    )
    sol, *_ = np.linalg.lstsq(mat, raw, rcond=None)
    sol = np.clip(sol, 0, None)
    sol /= sol.sum()
    p_mit = success_prob(sol)
    log.info(
        "mitigated success = %.4f (raw %.4f, difference %+.4f, " "one sigma %.4f)",
        p_mit,
        p_raw,
        p_mit - p_raw,
        sigma,
    )
    # Readout error here is ~0.3% per qubit, far below the gate error, so
    # the honest expectation is "no harm", not a big gain.
    r.append(
        check(
            p_mit > p_raw - 3 * sigma,
            "mitigation does not make the result worse (within 3 sigma)",
        )
    )

    log.info("=== Check 6: zero-noise extrapolation (CNOT folding) ===")
    # Fold the transpiled circuit (not the logical one) so the extra CNOT pairs
    # survive and the physical layout stays the same.
    scales, vals = [1, 3, 5], []
    for k in range(3):
        vals.append(success_prob(run_probs(sim, fold_cx(tqc, k), shots)))
        log.info("noise x%d: success = %.4f", scales[k], vals[-1])
    # Success decays roughly exponentially with the number of noisy gates.
    _, icept = np.polyfit(scales, np.log(vals), 1)
    p_zne = float(np.exp(icept))
    log.info("exponential fit back to zero noise: %.4f (ideal 0.5000)", p_zne)
    r.append(
        check(
            vals[0] > vals[1] > vals[2], "success falls as more noisy CNOTs are added"
        )
    )
    r.append(
        check(
            abs(0.5 - p_zne) < abs(0.5 - p_raw),
            f"extrapolated {p_zne:.4f} is closer to 0.5 than raw " f"{p_raw:.4f}",
        )
    )
    return p_mit, p_zne


def run_until_success(probs, rng, limit=10000):
    """One user session: repeat the experiment until the post-processing
    produces factors. Returns (attempts, factors, trace)."""
    trace = []
    # The limit only guards against an infinite loop if probabilities are bad.
    for attempt in range(1, limit + 1):
        m = int(rng.choice(len(probs), p=probs))
        factors = factors_from_measurement(m, T, A, N)
        trace.append((attempt, m, factors))
        if factors:
            return attempt, factors, trace
    raise RuntimeError("no success within limit; probabilities are wrong")


def check_repeat(raw, p_raw, trials, seed, r):
    """Check 7: the full user-facing loop, with classical verification."""
    log.info("=== Check 7: repeat until the factors appear ===")
    # Sample from the NOISY distribution: this is what a user of the device
    # would experience, including the runs that fail.
    rng = np.random.default_rng(seed)
    probs = raw / raw.sum()
    attempts, answers, first = [], set(), None
    for _ in range(trials):
        n_try, fac, trace = run_until_success(probs, rng)
        attempts.append(n_try)
        answers.add(fac)
        # Keep the first session that needed a retry: it is the instructive
        # one, because it shows a failed run being discarded.
        if first is None and n_try > 1:
            first = trace
    first = first or trace
    for attempt, m, fac in first:
        log.info(
            "attempt %d: m=%d (phase %d/%d) -> %s",
            attempt,
            m,
            m,
            2**T,
            fac if fac else "no factors, run again",
        )
    # Attempts until first success are geometric: mean 1/p, variance (1-p)/p^2.
    mean = float(np.mean(attempts))
    theory = 1 / p_raw
    sd = math.sqrt((1 - p_raw) / p_raw**2 / trials)
    log.info(
        "mean attempts over %d sessions = %.3f (theory 1/p = %.3f)",
        trials,
        mean,
        theory,
    )
    r.append(
        check(
            abs(mean - theory) < 5 * sd,
            "average attempts matches the geometric law 1/p",
        )
    )
    r.append(
        check(
            answers == {(3, 5)} and 3 * 5 == N,
            "every session ended with {3, 5}, verified: 3 x 5 = 15",
        )
    )


def main(argv=None):
    """Run the whole capstone; return the process exit code."""
    ap = argparse.ArgumentParser(
        description="Lab 10: capstone, Shor on a " "noisy simulated device"
    )
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.shots < 20000:
        ap.error(f"--shots must be >= 20000. Got {args.shots}.")
    if args.trials < 100:
        ap.error(f"--trials must be >= 100. Got {args.trials}.")
    backend = make_device()
    # Density-matrix simulation keeps exact noise for 7 qubits and is fast.
    sim = AerSimulator.from_backend(
        backend, method="density_matrix", seed_simulator=args.seed
    )
    r = []
    ideal = check_circuit(r)
    tqc = check_compile(backend, r)
    raw, p_raw, sigma = check_noisy(sim, tqc, args.shots, r)
    p_mit, p_zne = check_mitigation(backend, sim, tqc, raw, p_raw, sigma, args.shots, r)
    check_repeat(raw, p_raw, args.trials, args.seed, r)

    log.info("--- Summary: probability that one run yields the factors ---")
    log.info("ideal circuit        %.4f", success_prob(ideal))
    log.info("noisy device (raw)   %.4f", p_raw)
    log.info("+ readout mitigation %.4f", p_mit)
    log.info("+ zero-noise extrap. %.4f", p_zne)
    log.info("pure garbage floor   0.2500")
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
