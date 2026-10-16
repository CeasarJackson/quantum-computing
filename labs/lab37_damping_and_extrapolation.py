#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab37_damping_and_extrapolation.py

Purpose
-------
Lab 33 added amplitude damping (T1 decay), a non-unitary, non-unital channel
that always pushes spins the same way. Lab 30 extrapolated Pauli noise to zero
by unitary folding, Lab 31 verified S_z, Lab 34 twirled with Pauli frames. This
lab asks how those tools behave on damping, with exact density matrices (n=6)
and sampled trajectories (long chain).

What it finds:
  1. Damping bias is large (about 0.1 per 1e-3 of rate on 6 sites) and grows
     LINEARLY with the fold factor, so folding plus a line or parabola removes
     it (zero-noise extrapolation works as for Pauli noise).
  2. Every damping jump shifts S_z by the same unit, so S_z verification keeps
     only no-jump runs. What is left is the back-action of "no jump happened"
     (a tiny non-unitary Z-type squeeze), second order in the rate, so the
     verified bias grows like fold^2: the straight line of Lab 30 is the wrong
     fit for it (as in Lab 36) and one parabola removes it.
  3. Pauli frames (Lab 34) do NOT help here. They leave the raw bias alone and
     destroy the one-directional structure verification relies on: the
     verified bias gets about 100 times larger.
  4. Sampled, long chain: the no-jump runs are all the SAME state, so the
     verified energy has zero variance. Extrapolating raw data multiplies the
     sampling error; extrapolating verified data buys nothing, because
     verification already sits at the noise floor.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab37_damping_and_extrapolation.py
    python3 labs/lab37_damping_and_extrapolation.py --trajectories 400
    python3 labs/lab37_damping_and_extrapolation.py --quick

Log files
---------
logs/lab37/lab37_<YYYYmmdd_HHMMSS>.log

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
from mpslib import run_hva, sparse_hamiltonian, train_angles
from noiselib import (
    amplitude_damping_kraus,
    damped_mps,
    dm_damping_state,
    dm_expectation,
    dm_kraus_state,
    dm_sz_from_rho,
    mean_se,
    richardson_weights,
    sz_ratio,
    sz_verified,
)

log = get_logger("lab37")

FOLDS = (1, 3, 5)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def exponent(values):
    """Fitted power b in |bias| ~ fold^b over FOLDS."""
    return float(np.polyfit(np.log(FOLDS), np.log(np.abs(values)), 1)[0])


def exact_biases(n, params, clean, gamma, frames=False):
    """Raw and S_z-verified biases at every fold, plus the verified acceptance."""
    # Same circuit at each fold; the density matrix gives raw and verified numbers
    # from one state, so the comparison has no sampling noise.
    raw, ver, acc = [], [], []
    for fold in FOLDS:
        rho = dm_damping_state(n, params, gamma, fold, frames)
        raw.append(dm_expectation(rho, n) - clean)
        energy, accept = dm_sz_from_rho(rho, n)
        ver.append(energy - clean)
        acc.append(accept)
    return np.array(raw), np.array(ver), np.array(acc)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_machinery(r, ctx):
    """Check 1: the new density matrix agrees with the old one."""
    log.info("=== Check 1: machinery ===")
    n, params, g = 6, ctx["params6"], 3e-3
    rho = dm_damping_state(n, params, g)
    old = dm_kraus_state(n, params, lambda q: amplitude_damping_kraus(g))
    r.append(check(np.allclose(rho, old, atol=1e-12), "plain channel equals Lab 33's"))
    # The frame average must stay a proper channel (trace one) and must change
    # the state, otherwise the frames would be doing nothing.
    twirled = dm_damping_state(n, params, g, frames=True)
    tr = float(np.trace(twirled.reshape(2**n, 2**n)).real)
    r.append(check(abs(tr - 1) < 1e-12, "the frame-twirled channel keeps the trace"))
    r.append(
        check(not np.allclose(twirled, rho, atol=1e-6), "twirling changes the state")
    )
    ideal = dm_damping_state(n, params, 0.0, frames=True)
    e = dm_expectation(ideal, n)
    r.append(check(abs(e - ctx["clean6"]) < 1e-10, "no damping: frames change nothing"))


def check_fold_scaling(r, ctx):
    """Check 2: raw bias is linear in the fold factor, verified bias quadratic."""
    log.info("=== Check 2: fold scaling ===")
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    raw, ver, acc = exact_biases(n, params, clean, 1e-3)
    log.info("raw bias at folds 1,3,5: %s", np.round(raw, 5))
    log.info("S_z verified bias:       %s", np.round(ver, 7))
    log.info("S_z acceptance:          %s", np.round(acc, 4))
    # 0.05 is far above rounding; the damping bias is first order in the rate.
    r.append(check(raw[0] > 0.05, f"damping bias is large ({raw[0]:.3f})"))
    s_raw, s_ver = exponent(raw), exponent(ver)
    r.append(check(0.9 < s_raw < 1.1, f"raw bias grows like fold^{s_raw:.2f}"))
    r.append(check(1.7 < s_ver < 2.4, f"verified bias grows like fold^{s_ver:.2f}"))
    ctx["exact"] = (raw, ver, acc)


def check_extrapolation(r, ctx):
    """Check 3 and 4: zero-noise extrapolation of raw and verified data."""
    raw, ver, _ = ctx["exact"]
    # Lagrange weights for a line (folds 1,3) and a parabola (folds 1,3,5).
    w2, w3 = richardson_weights(FOLDS[:2]), richardson_weights(FOLDS)
    log.info("=== Check 3: extrapolating the raw data ===")
    lin, quad = float(w2 @ raw[:2]), float(w3 @ raw)
    log.info("raw: single %+.2e  line %+.2e  parabola %+.2e", raw[0], lin, quad)
    r.append(check(abs(lin) < 0.03 * raw[0], "a line removes more than 97%"))
    r.append(check(abs(quad) < 0.005 * raw[0], "a parabola removes more than 99.5%"))
    r.append(check(abs(quad) < abs(lin), "the parabola beats the line"))
    log.info("=== Check 4: verification, then extrapolation ===")
    v_lin, v_quad = float(w2 @ ver[:2]), float(w3 @ ver)
    log.info(
        "verified: single %+.2e  line %+.2e  parabola %+.2e", ver[0], v_lin, v_quad
    )
    r.append(check(abs(ver[0]) < 1e-3 * raw[0], "verification alone removes >99.9%"))
    r.append(
        check(abs(v_lin) > abs(ver[0]), "on verified data the straight line is worse")
    )
    r.append(
        check(abs(v_quad) < 0.2 * abs(ver[0]), "the parabola still improves it 5x")
    )


def check_frames(r, ctx):
    """Check 5: Pauli frames do not help damping."""
    log.info("=== Check 5: Pauli frames ===")
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    raw0, ver0, _ = ctx["exact"]
    # Same rate, same circuit, with the Lab 34 frames switched on.
    raw1, ver1, _ = exact_biases(n, params, clean, 1e-3, frames=True)
    log.info("framed raw bias:      %s", np.round(raw1, 5))
    log.info("framed verified bias: %s", np.round(ver1, 7))
    r.append(
        check(
            abs(raw1[0] - raw0[0]) < 0.05 * raw0[0], "frames leave the raw bias alone"
        )
    )
    # Compare what extrapolation can achieve with and without frames.
    w3 = richardson_weights(FOLDS)
    q0, q1 = abs(float(w3 @ raw0)), abs(float(w3 @ raw1))
    r.append(check(0.2 < q1 / q0 < 5, "and do not change what extrapolation achieves"))
    ratio = abs(ver1[0]) / abs(ver0[0])
    log.info("verified bias with frames / without: %.0fx", ratio)
    r.append(check(ratio > 20, "frames make S_z verification >20x worse"))
    r.append(check(abs(ver1[0]) < 0.05 * raw1[0], "(it still beats the raw run)"))


def check_long_chain(r, ctx):
    """Check 6: sampled trajectories on the long chain."""
    n, chi, count, g = ctx["n"], 32, ctx["count"], ctx["gamma"]
    params = ctx["params8"]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 6: n=%d, g=%g, N=%d, folds 1 and 3 ===", n, g, count)
    rng = np.random.default_rng(ctx["seed"] + 37)
    # One entry per fold: bias, error bar, verified bias, acceptance, kept runs.
    res = {}
    for fold in (1, 3):
        # Raw energies, plus numerator and acceptance of the S_z projection.
        raw, nums, accs = [], [], []
        for _ in range(count):
            mps = damped_mps(n, params, chi, g, rng, fold)
            raw.append(mps.energy())
            num, acc = sz_verified(mps)
            nums.append(num)
            accs.append(acc)
        m_raw, se_raw = mean_se(np.array(raw))
        m_ver, _ = sz_ratio(nums, accs)
        # Accepted runs only: their energies are num / acc each.
        kept = [a / b for a, b in zip(nums, accs) if b > 0.5]
        res[fold] = (m_raw - clean, se_raw, m_ver - clean, float(np.mean(accs)), kept)
        log.info(
            "fold %d: raw %+.3f +- %.3f | verified %+.2e | kept %.0f%% | spread %.1e",
            fold,
            res[fold][0],
            se_raw,
            res[fold][2],
            100 * res[fold][3],
            np.std(kept) if kept else float("nan"),
        )
    b1, s1, v1, a1, k1 = res[1]
    b3, s3, v3, a3, k3 = res[3]
    r.append(check(b1 > 2 * s1, "the raw damped energy is biased"))
    # A line through folds 1 and 3 multiplies the two errors by 1.5 and 0.5.
    # Errors of independent estimates add in quadrature after weighting.
    zne_se = float(np.hypot(1.5 * s1, 0.5 * s3))
    log.info(
        "raw extrapolated: %+.3f +- %.3f (single run: +- %.3f)",
        1.5 * b1 - 0.5 * b3,
        zne_se,
        s1,
    )
    r.append(
        check(zne_se > 1.5 * s1, "extrapolating raw data inflates the error bar >1.5x")
    )
    r.append(check(abs(v1) < 0.01 * b1, "verification removes >99% of the bias"))
    r.append(
        check(
            len(k1) > 0 and np.std(k1) < 1e-8,
            "the accepted runs are all the SAME state",
        )
    )
    r.append(check(a3 < a1, "folding costs shots: acceptance falls with the fold"))
    r.append(
        check(
            abs(v1) < 5e-3 and abs(v3) < 5e-3,
            "verified bias is below 5e-3 at both folds",
        )
    )


def main(argv=None):
    """Run every Lab 37 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 37: damping and extrapolation")
    ap.add_argument("--trajectories", type=int, default=200, help="long chain")
    ap.add_argument("--gamma", type=float, default=0.002, help="damping rate")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.trajectories < 50:
        ap.error(f"--trajectories must be at least 50. Got {args.trajectories}.")
    if not 1e-4 <= args.gamma <= 1e-2:
        ap.error(f"--gamma must be between 1e-4 and 1e-2. Got {args.gamma}.")
    quick = args.quick
    # Quick mode shortens the chain and uses a stronger rate to compensate.
    ctx = {
        "seed": args.seed,
        "n": 12 if quick else 16,
        "count": 80 if quick else args.trajectories,
        "gamma": 2 * args.gamma if quick else args.gamma,
    }
    # Angles are trained on small chains, exactly as in Labs 28-36.
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
    # Every exact bias is measured against this error-free energy.
    ctx["clean6"] = run_hva(6, ctx["params6"], 64).energy()
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_machinery(r, ctx)
    check_fold_scaling(r, ctx)
    check_extrapolation(r, ctx)
    check_frames(r, ctx)
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
