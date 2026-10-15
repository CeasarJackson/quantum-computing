#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab29_noisy_mps.py

Purpose
-------
Lab 28 pushed the Hamiltonian Variational Ansatz to 12 layers on noiseless
simulators. Real hardware is noisy, and the density matrix of a noisy n-qubit
state needs 4^n numbers, so a noisy circuit cannot be simulated exactly beyond
about a dozen qubits. This lab simulates it on a matrix product state with
QUANTUM TRAJECTORIES (Monte Carlo wave functions):

  * The depolarizing channel  rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)
    is unravelled into random pure-state histories: after each gate layer,
    every qubit the layer touched suffers, with probability p, a random Pauli
    X, Y or Z (each p/3).
  * A Pauli is a unitary on one physical index, so each trajectory is still an
    ordinary MPS with no extra bond dimension, and the average of an
    observable over trajectories equals its value in the noisy density
    matrix. The price is statistical error 1/sqrt(N).
  * A trajectory that draws no error equals the noiseless circuit, so its
    energy is reused instead of being simulated (an exact saving at small p).

The lab verifies:
  1. A Pauli error on the MPS equals the same Pauli on the state vector, and
     keeps the state normalised.
  2. With p = 0 every trajectory is the noiseless circuit.
  3. The trajectory average matches an exact density-matrix simulation
     (n=6) within its statistical error, and noise raises the energy.
  4. The statistical error falls like 1/sqrt(N).
  5. On a chain of 16 sites the energy error grows with p at every depth,
     deeper circuits are best without noise and worst with enough noise; the
     break-even error rate is estimated.
  6. Bond-dimension truncation does not change noisy trajectories.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab29_noisy_mps.py
    python3 labs/lab29_noisy_mps.py --trajectories 400 --seed 5
    python3 labs/lab29_noisy_mps.py --quick

Log files
---------
logs/lab29/lab29_<YYYYmmdd_HHMMSS>.log

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
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import (
    DMRG,
    PAULI,
    heisenberg_mpo,
    run_hva,
    sparse_hamiltonian,
    train_angles,
)
from noiselib import (
    apply_pauli,
    dm_energy,
    mean_se,
    noisy_energy,
    sample,
)

log = get_logger("lab29")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_pauli(r, ctx):
    """Check 1: a Pauli error is a unitary on one site."""
    log.info("=== Check 1: Pauli errors on the MPS (n=6) ===")
    n = 6
    mps = run_hva(n, ctx["angles"][2], 64)
    vec = mps.to_vector()
    worst, norm_gap = 0.0, 0.0
    # Every site and every Pauli: a complete test of the injection routine.
    for q in range(n):
        for name in "XYZ":
            hit = mps.copy()
            apply_pauli(hit, q, name)
            # Little endian: qubit q is bit q of the amplitude index.
            t = vec.reshape(2 ** (n - q - 1), 2, 2**q)
            want = np.einsum("ab,ibj->iaj", PAULI[name], t).reshape(-1)
            worst = max(worst, float(np.abs(hit.to_vector() - want).max()))
            norm_gap = max(norm_gap, abs(hit.norm() - mps.norm()))
    r.append(check(worst < 1e-12, f"MPS and state-vector Paulis agree ({worst:.1e})"))
    r.append(check(norm_gap < 1e-12, f"the norm is unchanged ({norm_gap:.1e})"))


def check_noiseless(r, ctx):
    """Check 2: p = 0 gives the noiseless circuit for every trajectory."""
    n, chi = ctx["n"], ctx["chi"]
    params = ctx["angles"][3]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 2: p = 0 (n=%d, 4 layers) ===", n)
    rng = np.random.default_rng(1)
    # No e_clean here, so the circuit really is simulated.
    es = sample(n, params, chi, 0.0, 3, rng)
    gap = float(np.abs(es - clean).max())
    # Zero tolerance on structure: p = 0 must reproduce the clean run exactly.
    r.append(
        check(
            gap < 1e-9, f"3 simulated trajectories equal the clean energy ({gap:.1e})"
        )
    )


def check_exact(r, ctx):
    """Check 3: the trajectory average equals the density-matrix result."""
    n, p, count = 6, ctx["p_exact"], ctx["exact_count"]
    params = ctx["angles"][2]
    log.info("=== Check 3: trajectories versus the density matrix (n=%d) ===", n)
    clean = run_hva(n, params, 64).energy()
    exact = dm_energy(n, params, p)
    exact0 = dm_energy(n, params, 0.0)
    # A different seed per check keeps the random streams independent.
    rng = np.random.default_rng(ctx["seed"] + 1)
    # Reusing the clean energy for error-free trajectories is exact and fast.
    es = sample(n, params, 64, p, count, rng, clean)
    mean, se = mean_se(es)
    log.info(
        "p=%.3f: clean %.5f | exact noisy %.5f | trajectories %.5f +- %.5f (N=%d)",
        p,
        clean,
        exact,
        mean,
        se,
        count,
    )
    # Four standard errors keeps the false-failure rate near 1 in 15000.
    r.append(
        check(abs(exact0 - clean) < 1e-9, "density matrix at p=0 equals the MPS energy")
    )
    r.append(check(exact > clean + 0.1, "noise raises the energy"))
    r.append(
        check(
            abs(mean - exact) < 4 * se,
            f"trajectory average within 4 standard errors "
            f"({abs(mean - exact) / se:.1f})",
        )
    )
    ctx["traj_sample"] = es


def check_statistics(r, ctx):
    """Check 4: standard error falls like 1/sqrt(N)."""
    es = ctx["traj_sample"]
    log.info("=== Check 4: statistical error versus N ===")
    quarter = es[: len(es) // 4]
    se_small, se_big = mean_se(quarter)[1], mean_se(es)[1]
    ratio = se_small / se_big
    log.info(
        "SE with N/4 = %.4f, with N = %.4f, ratio %.2f (ideal 2)",
        se_small,
        se_big,
        ratio,
    )
    # Quartering N should double the standard error if samples are independent.
    r.append(
        check(1.4 < ratio < 2.9, f"4x the trajectories halves the error (x{ratio:.2f})")
    )


def check_sweep(r, ctx):
    """Check 5: energy error versus noise rate and depth on a longer chain."""
    n, chi, count = ctx["sweep_n"], ctx["chi"], ctx["count"]
    e_ref = ctx["ref"]
    depths, rates = ctx["depths"], ctx["rates"]
    log.info("=== Check 5: n=%d, error above DMRG (N=%d trajectories) ===", n, count)
    log.info("%-8s %s", "p", "  ".join(f"depth {d:2d}        " for d in depths))
    table = {}
    rng = np.random.default_rng(ctx["seed"] + 2)
    t0 = time.perf_counter()
    for p in rates:
        row = []
        for d in depths:
            params = ctx["angles"][d - 1]
            clean = ctx["clean"][d]
            es = sample(n, params, chi, p, count, rng, clean)
            mean, se = mean_se(es) if p > 0 else (clean, 0.0)
            table[(p, d)] = (mean - e_ref, se)
            row.append(f"{mean - e_ref:7.3f} +- {se:5.3f}")
        log.info("%-8g %s", p, "  ".join(row))
    log.info("sweep time %.0f s", time.perf_counter() - t0)
    shallow, deep = depths[0], depths[-1]
    # Noise only ever hurts: compare the highest rate with the noiseless run.
    top = rates[-1]
    # Differences are compared in units of the combined standard error.
    r.append(
        check(
            all(
                table[(top, d)][0] > table[(0.0, d)][0] + 3 * table[(top, d)][1]
                for d in depths
            ),
            f"at p={top:g} every depth is worse than noiseless",
        )
    )
    # Deeper circuits have more gates and so more chances for an error.
    r.append(
        check(
            table[(0.0, deep)][0] < table[(0.0, shallow)][0],
            f"noiseless: {deep} layers beat {shallow} layers",
        )
    )
    gap = table[(top, deep)][0] - table[(top, shallow)][0]
    se = (table[(top, deep)][1] ** 2 + table[(top, shallow)][1] ** 2) ** 0.5
    r.append(
        check(
            gap > 3 * se,
            f"at p={top:g}: {deep} layers are worse than {shallow} by {gap:.2f} "
            f"({gap / se:.1f} standard errors)",
        )
    )
    # Break-even: the first pair of neighbouring rates between which the deep
    # circuit goes from better to worse than the shallow one; interpolate the
    # difference linearly in p to locate the crossing.
    diff = [table[(p, deep)][0] - table[(p, shallow)][0] for p in rates]
    p_star = None
    for k in range(len(rates) - 1):
        if diff[k] < 0 <= diff[k + 1]:
            frac = -diff[k] / (diff[k + 1] - diff[k])
            p_star = rates[k] + frac * (rates[k + 1] - rates[k])
            break
    if p_star is not None:
        log.info(
            "break-even error rate between %d and %d layers: p* ~ %.1e per qubit "
            "per layer",
            shallow,
            deep,
            p_star,
        )
    r.append(
        check(
            p_star is not None and 1e-5 < p_star < 1e-2,
            "the deep circuit stops paying off at a rate between 1e-5 and 1e-2",
        )
    )


def check_truncation(r, ctx):
    """Check 6: the same error histories at two bond dimensions."""
    n, d = ctx["sweep_n"], ctx["depths"][-1]
    params = ctx["angles"][d - 1]
    log.info(
        "=== Check 6: truncation of noisy trajectories (n=%d, %d layers) ===", n, d
    )
    worst = 0.0
    # Identical seeds reproduce identical error histories at both chi.
    for chi in (ctx["chi"], 2 * ctx["chi"]):
        # Identical seeds reproduce identical error histories at both chi.
        rng = np.random.default_rng(99)
        es = np.array(
            [noisy_energy(n, params, chi, 0.02, rng) for _ in range(ctx["trunc_count"])]
        )
        if chi == ctx["chi"]:
            low = es
        else:
            worst = float(np.abs(es - low).max())
    log.info(
        "largest energy difference between chi=%d and %d: %.1e",
        ctx["chi"],
        2 * ctx["chi"],
        worst,
    )
    r.append(
        check(worst < 5e-3, f"noisy trajectories converged in chi ({worst:.1e} < 5e-3)")
    )


def main(argv=None):
    """Run every Lab 29 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 29: noisy circuits on an MPS")
    ap.add_argument("--trajectories", type=int, default=300, help="per sweep point")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.trajectories < 40:
        ap.error(f"--trajectories must be at least 40. Got {args.trajectories}.")
    quick = args.quick
    # Quick mode shortens the chain and uses fewer trajectories; the logic of
    # every check is unchanged.
    ctx = {
        "seed": args.seed,
        "n": 10,
        "sweep_n": 12 if quick else 16,
        "chi": 32,
        "count": 60 if quick else args.trajectories,
        "p_exact": 0.02,
        "exact_count": 400 if quick else 2000,
        "depths": [2, 6] if quick else [2, 4, 8],
        "rates": [0.0, 1e-3, 1e-2] if quick else [0.0, 1e-4, 1e-3, 3e-3],
        "trunc_count": 6 if quick else 12,
    }
    # Angles trained on a short chain (state-vector adjoint) and reused,
    # exactly as in Lab 28.
    n_train = 8
    e0 = float(
        eigsh(sparse_hamiltonian(n_train), k=1, which="SA", return_eigenvectors=False)[
            0
        ].real
    )
    trained = train_angles(
        n_train,
        e0,
        max(ctx["depths"]),
        np.random.default_rng(args.seed),
        extra_starts=0,
    )
    ctx["angles"] = [x for _, x in trained]
    n = ctx["sweep_n"]
    ctx["ref"] = DMRG(heisenberg_mpo(n), 32).run(max_sweeps=30, tol=1e-9)[-1]
    ctx["clean"] = {
        d: run_hva(n, ctx["angles"][d - 1], ctx["chi"]).energy() for d in ctx["depths"]
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_pauli(r, ctx)
    check_noiseless(r, ctx)
    check_exact(r, ctx)
    check_statistics(r, ctx)
    check_sweep(r, ctx)
    check_truncation(r, ctx)
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
