#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab19_circuit_level_surface_code.py

Purpose
-------
Lab 12 corrected errors on data qubits only (threshold ~9%). Lab 18 added
faulty syndrome readout (threshold ~3%). Lab 19 completes the ladder with
CIRCUIT-LEVEL noise: every gate, reset, measurement and idle step of the
syndrome-extraction circuit can fail, and a single CNOT error can spread to
several qubits ("hook" errors). The threshold drops again, to about 0.7%.

The lab uses Stim, the standard fast stabilizer simulator, to build the full
rotated-surface-code memory experiment (d rounds of CNOT-based syndrome
extraction plus a final readout), samples noisy runs, turns the circuit into a
detector error model, and decodes with minimum-weight matching (pymatching).

Noise model (all four knobs equal to p): depolarizing noise after every
Clifford gate, a flip after every reset, a flip before every measurement, and
depolarizing noise on the data qubits at the start of each round.

The lab verifies:
  1. Circuit structure: 2d^2 - 1 qubits, (d^2 - 1) * d detectors, one logical
     observable; with p = 0 no detector ever fires and the logical never flips.
  2. Fault distance: the shortest chain of single faults that flips the logical
     without any detector firing has length d, so hook errors did not cost
     distance with this gate ordering.
  3. Failure scales as p^2 for d=3 (it takes two faults to defeat the code).
  4. Threshold: the d=3, 5, 7 curves cross near 0.7% (read-off range 0.4-1.0%).
  5. Below threshold a bigger code wins; above it, it loses.
  6. The ladder: 9.4% (Lab 12) -> 2.5% (Lab 18) -> ~0.7% (this lab).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab19_circuit_level_surface_code.py
    python3 labs/lab19_circuit_level_surface_code.py --shots 40000 --seed 3
    python3 labs/lab19_circuit_level_surface_code.py --quick

Log files
---------
logs/lab19/lab19_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (stim, pymatching).
130 User interrupted execution.
===============================================================================
"""

import argparse
import math
import sys

import numpy as np

from applog import get_logger

log = get_logger("lab19")

# Both packages are required; fail early with the fix instead of a traceback.
try:
    import pymatching
    import stim
except ImportError:  # pragma: no cover
    log.fail("missing package: pip install stim pymatching")
    sys.exit(2)

DISTANCES = (3, 5, 7)
P_GRID = (0.003, 0.005, 0.006, 0.007, 0.008, 0.010)
# Reference thresholds from Labs 12 and 18 (their crossings of d=3 and d=7).
LADDER = {"data errors only (Lab 12)": 0.094, "plus noisy readout (Lab 18)": 0.025}


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def stderr(x, shots):
    """Binomial standard error of a measured rate."""
    return math.sqrt(max(x, 1e-12) * (1 - x) / shots)


def memory_circuit(d, p, rounds=None):
    """Rotated surface code memory experiment with circuit-level noise p."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds if rounds else d,
        after_clifford_depolarization=p,  # gate errors
        after_reset_flip_probability=p,  # bad state preparation
        before_measure_flip_probability=p,  # readout errors
        before_round_data_depolarization=p,
    )  # idle errors on data qubits


def logical_rate(d, p, shots, seed):
    """Fraction of memory experiments whose decoded logical qubit is wrong."""
    # Build the noisy circuit once; everything else is derived from it.
    circuit = memory_circuit(d, p)
    # The detector error model lists every independent fault the circuit can
    # suffer and which detectors/logical it flips; decomposition splits
    # correlated faults into graph edges that matching can handle.
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    # Detection events and the true logical flip, for every shot at once.
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    # Matching predicts whether the logical flipped; compare with the truth.
    pred = matching.decode_batch(det)
    return float((pred != obs).any(axis=1).mean())


def check_structure(r):
    """Check 1: sizes, and silence when there is no noise."""
    log.info("=== Check 1: circuit structure ===")
    # Sizes follow exact formulas, so any mismatch signals a wrong circuit.
    for d in DISTANCES:
        c = memory_circuit(d, 0.0)
        nq = len(c.get_final_qubit_coordinates())
        log.info(
            "d=%d: %d qubits, %d detectors, %d observable(s), %d " "instructions",
            d,
            nq,
            c.num_detectors,
            c.num_observables,
            len(c),
        )
        r.append(
            check(
                nq == 2 * d * d - 1
                and c.num_observables == 1
                and c.num_detectors == (d * d - 1) * d,
                f"d={d}: 2d^2-1 = {2 * d * d - 1} qubits and "
                f"(d^2-1)*d = {(d * d - 1) * d} detectors",
            )
        )
    # With zero noise the code is just a stabilizer state, so nothing fires.
    c = memory_circuit(5, 0.0)
    det, obs = c.compile_detector_sampler(seed=1).sample(
        2000, separate_observables=True
    )
    r.append(
        check(
            det.sum() == 0 and obs.sum() == 0,
            "p=0: no detector fires and the logical never flips " "(2000 shots)",
        )
    )


def check_distance(r):
    """Check 2: shortest undetectable logical error equals d."""
    log.info("=== Check 2: fault distance of the noisy circuit ===")
    for d in (3, 5):
        c = memory_circuit(d, 0.001)
        # search_for_undetectable... would ignore decomposition; graphlike
        # error search is what matching can actually decode.
        chain = c.shortest_graphlike_error()
        log.info(
            "d=%d: shortest graphlike logical error uses %d fault(s)", d, len(chain)
        )
        r.append(
            check(
                len(chain) == d,
                f"d={d}: it takes {d} faults to flip "
                "the logical undetected (distance preserved)",
            )
        )


def check_scaling(r, seed):
    """Check 3: d=3 failure rate goes as p squared."""
    log.info("=== Check 3: failure ~ p^2 at d=3 ===")
    # Small p and many shots: failures need TWO faults, so they are rare.
    ps = (0.002, 0.004, 0.008)
    shots = 400000
    rates = [logical_rate(3, p, shots, seed + i) for i, p in enumerate(ps)]
    for p, x in zip(ps, rates):
        log.info("p=%.3f: logical error %.5f (+/- %.5f)", p, x, stderr(x, shots))
    # Straight-line fit on log-log axes; the slope is the power of p.
    slope = float(np.polyfit(np.log(ps), np.log(rates), 1)[0])
    log.info(
        "fitted exponent: %.2f (two faults needed => 2; higher-order "
        "terms push it a little above)",
        slope,
    )
    r.append(check(1.7 < slope < 2.5, f"log-log slope {slope:.2f} is about 2"))


def check_threshold(r, shots, seed):
    """Checks 4 to 6: threshold crossing and scaling with distance."""
    log.info("=== Check 4: threshold, %d shots per point ===", shots)
    # rates[(d, p)] holds the measured logical failure rate of each point.
    rates = {}
    for i, d in enumerate(DISTANCES):
        for j, p in enumerate(P_GRID):
            # Failures are rare at the lowest p, so that column gets 4x the
            # shots to keep its counts large enough for the ratio checks.
            n_shots = shots * (4 if j == 0 else 1)
            rates[d, p] = logical_rate(d, p, n_shots, seed + 100 * i + j)
    for p in P_GRID:
        log.info(
            "p=%.3f: d=3 %.4f   d=5 %.4f   d=7 %.4f",
            p,
            *(rates[d, p] for d in DISTANCES),
        )
    # The crossing of the smallest and largest code approximates the threshold.
    diff = [rates[3, p] - rates[7, p] for p in P_GRID]
    cross = None
    for k in range(len(P_GRID) - 1):
        if diff[k] > 0 >= diff[k + 1]:
            frac = diff[k] / (diff[k] - diff[k + 1])
            cross = P_GRID[k] + frac * (P_GRID[k + 1] - P_GRID[k])
    log.info(
        "d=3 and d=7 curves cross near p = %s (literature ~0.7%% for " "this model)",
        f"{cross:.4f}" if cross else "n/a",
    )
    r.append(
        check(
            cross is not None and 0.004 < cross < 0.010,
            "threshold estimate lies in 0.4%-1.0%",
        )
    )
    log.info("=== Check 5: below and above threshold ===")
    # One point well below the threshold and one above it.
    lo, hi = P_GRID[0], P_GRID[-1]
    r.append(
        check(
            rates[3, lo] > rates[5, lo] > rates[7, lo],
            f"p={lo}: bigger code, fewer failures (d=3>5>7)",
        )
    )
    r.append(
        check(
            rates[3, hi] < rates[5, hi] < rates[7, hi],
            f"p={hi}: above threshold, bigger code is WORSE",
        )
    )
    ratio = (
        rates[3, lo] / max(rates[5, lo], 1e-9),
        rates[5, lo] / max(rates[7, lo], 1e-9),
    )
    log.info("suppression per +2 in distance at p=%.3f: x%.1f, x%.1f", lo, *ratio)
    r.append(
        check(
            all(x > 1.3 for x in ratio),
            "each +2 in distance cuts the failure rate by more than "
            "1.3x (modestly, because p is only 2x below threshold)",
        )
    )
    log.info("=== Check 6: the ladder of thresholds ===")
    # Copy so the module constant is not modified by adding this lab's value.
    ladder = dict(LADDER)
    ladder["circuit-level noise (this lab)"] = cross if cross else float("nan")
    for name, val in ladder.items():
        log.info("%-32s threshold ~ %.1f%%", name, 100 * val)
    # Thresholds must fall as realism rises: data < readout < circuit level.
    vals = list(ladder.values())
    r.append(
        check(
            all(a > b for a, b in zip(vals, vals[1:])),
            "each added realism lowers the threshold: 9.4% > 2.5% > "
            f"{100 * vals[-1]:.1f}%",
        )
    )
    return rates


def hardware_note(rates):
    """Relate the threshold to the measured ibm_fez error (information only)."""
    # Value copied from the Lab 17 live run (not measured by this script).
    measured = 0.0028  # effective error per CNOT from Lab 17 (ibm_fez)
    log.info("=== Note: where does ibm_fez sit? ===")
    log.info(
        "Lab 17 measured ~%.2f%% per CNOT. Simulated here at p=%.3f: "
        "d=3 %.4f, d=5 %.4f, d=7 %.4f logical error per memory run.",
        100 * measured,
        P_GRID[0],
        rates[3, P_GRID[0]],
        rates[5, P_GRID[0]],
        rates[7, P_GRID[0]],
    )
    # A fair caveat: this simulation gives EVERY operation the CNOT error
    # rate, while real chips have worse readout and extra effects (leakage,
    # crosstalk, drift) that the model does not include.
    log.info(
        "If every operation were as good as ibm_fez's CNOT, the code "
        "would sit below threshold; real chips add worse readout, "
        "leakage and crosstalk that this model leaves out."
    )


def main(argv=None):
    """Run every Lab 19 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 19: circuit-level surface code")
    ap.add_argument(
        "--shots", type=int, default=20000, help="memory experiments per data point"
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use 6000 shots")
    args = ap.parse_args(argv)
    if args.shots < 2000:
        ap.error(f"--shots must be >= 2000. Got {args.shots}.")
    # --quick trades statistical power for speed; checks use wide margins.
    shots = 6000 if args.quick else args.shots
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_structure(r)
    check_distance(r)
    check_scaling(r, args.seed)
    rates = check_threshold(r, shots, args.seed)
    hardware_note(rates)
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
