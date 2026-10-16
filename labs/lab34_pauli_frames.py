#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab34_pauli_frames.py

Purpose
-------
Lab 33 showed that frozen coherent errors break zero-noise extrapolation:
their bias grows like fold^2 where random errors grow like fold^1, so a
straight-line extrapolation overshoots. RANDOMISED COMPILING (Pauli twirling)
is the standard cure: wrap each noisy gate in random Pauli "frames" that cancel
in the ideal circuit, so that a coherent error looks, on average over frames,
like a random Pauli error.

For this circuit the frames come free. The Heisenberg bond gate
exp(-i a (XX+YY+ZZ)) commutes with P (x) P for every Pauli P (the same one on
both qubits), so applying P (x) P before the gate and again after its error
leaves the ideal circuit unchanged. The error E seen by the state is then
(P x P) E (P x P). The four frames {II, XX, YY, ZZ} give the four single-qubit
Paulis four different sign patterns (orthogonal characters), so every
first-order cross term of a one-qubit coherent error averages to zero.

The lab verifies:
  1. The algebra: frames commute with the gate, their sign patterns are
     orthogonal, and zero error leaves the ideal state untouched.
  2. The exact frame average (a density matrix with 4 Kraus operators per
     bond) against sampled frame-randomised MPS runs.
  3. The bias falls several-fold (more at higher fold factors) and now grows
     like fold^1, so the straight-line extrapolation of Lab 30 works.
  4. End to end with sampling: extrapolated twirled energies recover the clean
     energy within statistics, where the un-twirled coherent run does not.
  5. The limits: a two-qubit coherent error (ZZ crosstalk) and a mis-scaled
     gate angle both commute with every frame, so frames leave them alone.
  6. Long chain (n=16): the same recovery with MPS sampling.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab34_pauli_frames.py
    python3 labs/lab34_pauli_frames.py --samples 300
    python3 labs/lab34_pauli_frames.py --quick

Log files
---------
logs/lab34/lab34_<YYYYmmdd_HHMMSS>.log

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
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import (
    PAULI,
    bond_unitary,
    overlap,
    run_hva,
    sparse_hamiltonian,
    train_angles,
)
from noiselib import (
    coherent_mps,
    dm_expectation,
    dm_twirled_state,
    mean_se,
    richardson_weights,
    twirled_mps,
)

log = get_logger("lab34")

# The simulated "device": its frozen detunings are fixed, so --seed changes only
# the trained angles and the frame sampling, never the errors being mitigated.
DEVICE_SEED = 8


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def slope(small, large, ratio):
    """Local power-law exponent: error(large) / error(small) = ratio^slope."""
    return float(np.log(large / small) / np.log(ratio))


def exact_bias(ctx, errs, fold, clean, **kwargs):
    """Frame-averaged energy bias from the exact density matrix (n=6)."""
    rho = dm_twirled_state(6, ctx["params6"], errs, fold=fold, **kwargs)
    return dm_expectation(rho, 6) - clean


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_algebra(r, ctx):
    """Check 1: why the frames are free, and why they twirl."""
    log.info("=== Check 1: the frame algebra ===")
    # An arbitrary angle, so nothing special about 0 or pi/4 is being used.
    u = bond_unitary(0.37)
    worst = 0.0
    for name in "XYZ":
        frame = np.kron(PAULI[name], PAULI[name])
        worst = max(worst, float(np.abs(u @ frame - frame @ u).max()))
    r.append(check(worst < 1e-12, f"the bond gate commutes with P x P ({worst:.1e})"))
    # Sign each frame gives to a one-qubit Pauli: (P x P)(s x I)(P x P) = +-(s x I).
    # Row i of `signs` is the pattern of signs that one-qubit Pauli i receives.
    frames = [np.kron(PAULI[p], PAULI[p]) for p in "IXYZ"]
    signs = np.zeros((4, 4))
    for i, name in enumerate("IXYZ"):
        target = np.kron(PAULI[name], PAULI["I"])
        for j, frame in enumerate(frames):
            moved = frame @ target @ frame
            signs[i, j] = np.real(np.trace(moved.conj().T @ target)) / 4
    r.append(
        check(
            np.allclose(signs @ signs.T, 4 * np.eye(4)),
            "the four one-qubit Paulis get orthogonal sign patterns",
        )
    )
    # With no error, any frames must reproduce the ideal state exactly.
    # The ideal state is the reference for every frame draw below.
    rng = np.random.default_rng(ctx["seed"])
    ideal = run_hva(6, ctx["params6"], 64)
    worst = 1.0
    for fold in (1, 3):
        for _ in range(5):
            mps = twirled_mps(6, ctx["params6"], 64, np.zeros(6), rng, fold=fold)
            worst = min(worst, abs(overlap(mps, ideal)))
    r.append(
        check(
            worst > 1 - 1e-9, f"zero error: frames leave the state alone ({worst:.12f})"
        )
    )
    # frames=False is the plain coherent run of Lab 33.
    errs = np.random.default_rng(ctx["seed"] + 1).normal(0, 0.02, 6)
    plain = twirled_mps(6, ctx["params6"], 64, errs, rng, frames=False).energy()
    same = coherent_mps(6, ctx["params6"], 64, errs).energy()
    r.append(
        check(abs(plain - same) < 1e-9, "without frames it is the Lab 33 coherent run")
    )
    # Four Kraus operators of weight 1/2 each must still sum to the identity.
    rho = dm_twirled_state(6, ctx["params6"], errs)
    r.append(
        check(
            abs(np.trace(rho.reshape(64, 64)) - 1) < 1e-10,
            "the frame-averaged channel keeps unit trace",
        )
    )


def check_exact_vs_sampled(r, ctx):
    """Check 2: the exact frame average against sampled MPS runs."""
    log.info("=== Check 2: exact average versus sampled frames, n=6 ===")
    clean, errs = ctx["clean6"], ctx["errs6"]
    # The density matrix averages over all frames; sampling draws a few of them.
    exact = exact_bias(ctx, errs, 1, clean)
    rng = np.random.default_rng(ctx["seed"] + 2)
    energies = [
        twirled_mps(6, ctx["params6"], 64, errs, rng).energy()
        for _ in range(ctx["small_samples"])
    ]
    mean, se = mean_se(np.array(energies))
    log.info("exact bias %.5f | sampled %.5f +- %.5f", exact, mean - clean, se)
    r.append(
        check(
            abs(mean - clean - exact) < 4 * se,
            "sampled frames reproduce the exact average",
        )
    )
    # The averaged channel is not a single coherent run: it has a spread.
    spread = float(np.std(energies, ddof=1))
    r.append(
        check(
            spread > 1e-4, f"individual runs differ from run to run (sd {spread:.4f})"
        )
    )


def check_bias_reduction(r, ctx):
    """Check 3: how much the frames help, exactly."""
    log.info("=== Check 3: bias with and without frames, n=6 ===")
    clean, errs = ctx["clean6"], ctx["errs6"]
    # Fold factors 1, 3, 5 are the same ones Lab 30 extrapolates over.
    ctx["coh"], ctx["twirl"] = [], []
    for fold in (1, 3, 5):
        coh = coherent_mps(6, ctx["params6"], 64, errs, fold=fold).energy() - clean
        twirl = exact_bias(ctx, errs, fold, clean)
        ctx["coh"].append(coh)
        ctx["twirl"].append(twirl)
        log.info(
            "fold %d: coherent %.5f, framed %.5f (x%.1f smaller)",
            fold,
            coh,
            twirl,
            coh / twirl,
        )
    ratios = [c / t for c, t in zip(ctx["coh"], ctx["twirl"])]
    # At fold 1 the coherent bias depends on how the first-order terms happen
    # to cancel for these angles, so it can even be the smaller one; the framed
    # bias is steady. The gain is therefore judged at higher folds.
    log.info("coherent / framed at folds 1, 3, 5: %s", np.round(ratios, 2))
    r.append(check(ratios[1] > 2, f"fold 3: frames cut the bias {ratios[1]:.1f}x"))
    r.append(check(ratios[2] > 3, f"fold 5: {ratios[2]:.1f}x"))
    r.append(check(ratios[2] > ratios[0], "the gain grows with the fold factor"))
    # SU(2) symmetry: the error axis does not matter, with frames either.
    # The singlet circuit is rotation invariant, so the error axis cannot matter.
    other = exact_bias(ctx, errs, 1, clean, axis="X")
    r.append(check(abs(other - ctx["twirl"][0]) < 1e-9, "X and Z error axes agree"))


def check_extrapolation(r, ctx):
    """Check 4: folding now behaves, so extrapolation works (exact)."""
    log.info("=== Check 4: scaling and extrapolation, exact ===")
    # Both lists came from check 3, so no extra simulation is needed here.
    coh, twirl = ctx["coh"], ctx["twirl"]
    s_twirl = slope(twirl[0], twirl[2], 5)
    log.info(
        "growth exponent: framed %.2f, coherent %.2f", s_twirl, slope(coh[0], coh[2], 5)
    )
    r.append(
        check(0.9 < s_twirl < 1.1, f"the framed bias grows like fold^{s_twirl:.2f}")
    )
    w2 = richardson_weights((1, 3))
    lin = float(w2 @ twirl[:2])
    lin_coh = float(w2 @ coh[:2])
    log.info("after a straight-line fit: framed %.2e, coherent %.4f", lin, lin_coh)
    r.append(
        check(
            abs(lin) < 0.02 * abs(twirl[0]),
            "a straight line removes 98% of the framed bias",
        )
    )
    r.append(
        check(abs(lin_coh) > abs(coh[0]), "while without frames it still overshoots")
    )


def check_sampled_mitigation(r, ctx):
    """Check 5: frames plus extrapolation, with statistical error, n=6."""
    log.info("=== Check 5: sampled mitigation, n=6 ===")
    clean, errs = ctx["clean6"], ctx["errs6"]
    rng = np.random.default_rng(ctx["seed"] + 5)
    # Two fold factors suffice for a straight-line fit, as in Lab 30.
    stats = {}
    for fold in (1, 3):
        energies = np.array(
            [
                twirled_mps(6, ctx["params6"], 64, errs, rng, fold=fold).energy()
                for _ in range(ctx["samples6"])
            ]
        )
        stats[fold] = mean_se(energies)
    w = richardson_weights((1, 3))
    est = w[0] * stats[1][0] + w[1] * stats[3][0] - clean
    se = float(np.hypot(w[0] * stats[1][1], w[1] * stats[3][1]))
    log.info(
        "raw coherent bias %.4f | framed raw %.4f | extrapolated %.4f +- %.4f",
        ctx["coh"][0],
        stats[1][0] - clean,
        est,
        se,
    )
    r.append(
        check(
            abs(est) < 4 * se + 1e-3, "the extrapolated energy equals the clean energy"
        )
    )
    r.append(
        check(
            abs(est) < 0.25 * ctx["coh"][0], "and is far closer than the coherent run"
        )
    )


def check_limits(r, ctx):
    """Check 6: what frames cannot do."""
    log.info("=== Check 6: errors that commute with every frame, n=6 ===")
    clean = ctx["clean6"]
    # No single-qubit error here: only the two-qubit or gate-angle error acts.
    zero = np.zeros(6)
    # ZZ crosstalk: a two-qubit error on each bond commutes with P x P.
    with_f = [exact_bias(ctx, zero, f, clean, crosstalk=0.03) for f in (1, 3, 5)]
    without = [
        exact_bias(ctx, zero, f, clean, crosstalk=0.03, frames=False) for f in (1, 3, 5)
    ]
    log.info(
        "crosstalk bias at folds 1,3,5: framed %s, plain %s",
        np.round(with_f, 5),
        np.round(without, 5),
    )
    r.append(
        check(
            np.allclose(with_f, without, atol=1e-9),
            "ZZ crosstalk: frames change nothing",
        )
    )
    s = slope(with_f[0], with_f[2], 5)
    r.append(check(s > 1.5, f"and the bias still grows like fold^{s:.2f}"))
    lin = float(richardson_weights((1, 3)) @ with_f[:2])
    r.append(
        check(abs(lin) > abs(with_f[0]), "so a straight-line fit still makes it worse")
    )
    # Mis-scaled gate angle: U(a(1+d)) also commutes with every frame.
    # delta = 0.05 means every bond angle is 5% too large.
    gate = [exact_bias(ctx, zero, f, clean, delta=0.05) for f in (1, 3)]
    r.append(
        check(
            gate[0] > 1e-3 and abs(gate[0] - gate[1]) < 1e-9,
            "gate-angle error: unchanged and still invisible to folding",
        )
    )


def check_long_chain(r, ctx):
    """Check 7: the same recovery on the long chain, by MPS sampling."""
    n, chi, count = ctx["n"], ctx["chi"], ctx["samples"]
    params = ctx["params8"]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 7: n=%d, 4 layers, %d frames per fold ===", n, count)
    # A different draw from the n=6 device, because there are more qubits.
    errs = np.random.default_rng(DEVICE_SEED + 1).normal(0, 0.02, n)
    coh = [coherent_mps(n, params, chi, errs, fold=f).energy() - clean for f in (1, 3)]
    coherent = coh[0]
    rng = np.random.default_rng(ctx["seed"] + 9)
    stats = {}
    for fold in (1, 3):
        energies = np.array(
            [
                twirled_mps(n, params, chi, errs, rng, fold=fold).energy()
                for _ in range(count)
            ]
        )
        stats[fold] = mean_se(energies)
    w = richardson_weights((1, 3))
    est = w[0] * stats[1][0] + w[1] * stats[3][0] - clean
    se = float(np.hypot(w[0] * stats[1][1], w[1] * stats[3][1]))
    log.info(
        "coherent bias %.4f | framed raw %.4f +- %.4f | extrapolated %.4f +- %.4f",
        coherent,
        stats[1][0] - clean,
        stats[1][1],
        est,
        se,
    )
    r.append(
        check(
            abs(est) < 4 * se + 1e-3, "the extrapolated energy equals the clean energy"
        )
    )
    r.append(
        check(
            abs(est) < 0.25 * coherent,
            f"the error falls from {coherent:.3f} to {abs(est):.3f}",
        )
    )


def main(argv=None):
    """Run every Lab 34 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 34: random Pauli frames")
    ap.add_argument("--samples", type=int, default=150, help="frames per fold")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.samples < 30:
        ap.error(f"--samples must be at least 30. Got {args.samples}.")
    quick = args.quick
    # Quick mode shortens the chain and uses fewer frame samples.
    ctx = {
        "seed": args.seed,
        "n": 12 if quick else 16,
        "chi": 32,
        "samples": 50 if quick else args.samples,
        "samples6": 150 if quick else 400,
        "small_samples": 200 if quick else 600,
    }
    # Angles are trained on small chains, exactly as in Labs 28-33.
    for size, key in ((6, "params6"), (8, "params8")):
        e0 = float(
            eigsh(sparse_hamiltonian(size), k=1, which="SA", return_eigenvectors=False)[
                0
            ].real
        )
        trained = train_angles(
            size, e0, 4, np.random.default_rng(args.seed), extra_starts=0
        )
        ctx[key] = trained[3][1]
    # Every bias in the lab is measured against this error-free energy.
    ctx["clean6"] = run_hva(6, ctx["params6"], 64).energy()
    # Frozen per-qubit detuning of about one degree, as in Lab 33 (fixed device).
    ctx["errs6"] = np.random.default_rng(DEVICE_SEED).normal(0, 0.02, 6)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_algebra(r, ctx)
    check_exact_vs_sampled(r, ctx)
    check_bias_reduction(r, ctx)
    check_extrapolation(r, ctx)
    check_sampled_mitigation(r, ctx)
    check_limits(r, ctx)
    check_long_chain(r, ctx)
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
