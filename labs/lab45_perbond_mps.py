#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab45_perbond_mps.py

Purpose
-------
Lab 40 found that giving every bond gate its own angle (a = b = c on each bond,
so total spin stays exactly conserved) cut the energy error by about 100x, but
only for 6 and 8 spins, where a state-vector simulation can do the training.
Lab 45 asks whether that survives depth and size, by training the per-bond
ansatz on a MATRIX PRODUCT STATE with an exact adjoint gradient (`pbmps.py`).

  1. GRADIENT AND SYMMETRY. The adjoint gradient equals finite differences, the
     MPS energy equals the state-vector energy of Lab 40, tied angles reproduce
     the ordinary HVA of Lab 27, and the state keeps <S^2> = 0 for any angles.
  2. DEPTH. For 10 spins (exact diagonalization available) HVA and per-bond
     circuits are trained at 4, 8 and 14 layers, each stage warm-started from
     the last. The HVA needs many layers (and can stall on a plateau, Lab 28);
     the per-bond circuit is accurate almost at once. The per-bond advantage is
     largest at SHALLOW depth (about 100x, as in Lab 40) and shrinks as the HVA
     catches up with depth, but a 4-layer per-bond circuit still beats a
     14-layer HVA with 3.5x fewer layers.
  3. SIZE. For 14 spins, beyond the comfortable reach of dense training, the
     same comparison is made at 6 layers with bond dimension 32, and the
     truncation is checked by re-evaluating at a larger bond dimension.

Honest limits: the per-bond errors are limited by the optimiser's iteration
budget, not by the ansatz (the parameter count n - 1 per layer exceeds the 42
singlet states of 10 spins by layer 5), so they are upper bounds on what the
ansatz can reach, and the gap to the HVA is a lower bound on its advantage.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab45_perbond_mps.py
    python3 labs/lab45_perbond_mps.py --seed 5
    python3 labs/lab45_perbond_mps.py --quick

Log files
---------
logs/lab45/lab45_<YYYYmmdd_HHMMSS>.log

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
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import mps_energy_and_adjoint_grad, sparse_hamiltonian
from noiselib import general_energy, perbond_from_hva, perbond_steps, s2_matrix
from pbmps import perbond_energy_grad, perbond_state

log = get_logger("lab45")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def ground_energy(n):
    """Exact ground-state energy of the open Heisenberg chain."""
    ham = sparse_hamiltonian(n)
    return float(eigsh(ham, k=1, which="SA", return_eigenvectors=False)[0].real)


def train(fun, x0, iters):
    """L-BFGS on an (energy, gradient) function; return the scipy result."""
    return minimize(
        lambda x: fun(x)[:2],
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": iters},
    )


def staged(n, chi, depths, iters, seed):
    """Train HVA and per-bond circuits at increasing depth, warm-started.

    Returns {layers: (hva_energy, perbond_energy, perbond_angles, hva_angles)}.
    A tiny random offset is added to every start: with all angles zero the
    singlet-product state is a stationary point and the optimiser would not move.
    """
    # prev_h / prev_p carry the last stage's angles into the next, deeper stage.
    out, prev_h, prev_p = {}, np.zeros(0), np.zeros(0)
    for layers in depths:
        rng = np.random.default_rng(seed * 100 + layers)
        # HVA: two angles per layer. New layers start at 0 (the identity).
        x0 = np.concatenate([prev_h, np.zeros(2 * layers - len(prev_h))])
        x0 = x0 + rng.normal(0, 0.05, 2 * layers)
        hva = train(lambda x: mps_energy_and_adjoint_grad(x, n, chi), x0, iters)
        prev_h = hva.x
        # Per-bond: n - 1 angles per layer. Start from the previous per-bond
        # solution when there is one, else from the HVA angles on every bond.
        if len(prev_p):
            x0 = np.concatenate([prev_p, np.zeros(layers * (n - 1) - len(prev_p))])
        else:
            x0 = perbond_from_hva(n, hva.x)
        x0 = x0 + rng.normal(0, 0.02, layers * (n - 1))
        per = train(lambda x: perbond_energy_grad(x, n, chi), x0, iters)
        prev_p = per.x
        out[layers] = (hva.fun, per.fun, per.x, hva.x)
        log.info(
            "n=%d, %2d layers: HVA %d params, per-bond %d params",
            n,
            layers,
            2 * layers,
            layers * (n - 1),
        )
    return out


def check_gradient(r, ctx):
    """Check 1: the adjoint gradient, the energy and the symmetry."""
    log.info("=== Check 1: gradient, energy and spin conservation (8 spins) ===")
    n, layers = 8, 2
    rng = np.random.default_rng(ctx["seed"])
    x = rng.uniform(-0.5, 0.5, layers * 7)
    energy, grad, disc = perbond_energy_grad(x, n, 64)
    # Central finite differences, one angle at a time; h trades truncation
    # error against rounding error, and 1e-5 balances them for this energy.
    h = 1e-5
    fd = np.zeros_like(x)
    for k in range(len(x)):
        step = np.zeros_like(x)
        step[k] = h
        fd[k] = (
            perbond_energy_grad(x + step, n, 64)[0]
            - perbond_energy_grad(x - step, n, 64)[0]
        ) / (2 * h)
    err = float(np.abs(fd - grad).max())
    log.info("max |adjoint - finite difference| = %.1e over %d angles", err, len(x))
    r.append(check(err < 1e-6, "adjoint gradient equals finite differences"))
    # Independent state-vector energy from Lab 40's machinery.
    exact = general_energy(n, perbond_steps(n, x))
    r.append(check(abs(energy - exact) < 1e-9, "MPS energy equals the state vector"))
    r.append(check(disc < 1e-12, "bond dimension 64 is exact for 8 spins"))
    # Tied angles are the ordinary HVA of Lab 27.
    hva_x = rng.uniform(-0.5, 0.5, 2 * layers)
    e_hva = mps_energy_and_adjoint_grad(hva_x, n, 64)[0]
    e_tied = perbond_energy_grad(perbond_from_hva(n, hva_x), n, 64)[0]
    r.append(check(abs(e_hva - e_tied) < 1e-9, "tied angles reproduce the HVA"))
    # Total spin is conserved for ANY angles: the state stays a singlet.
    vec = perbond_state(n, x, 64).to_vector()
    vec = vec / np.linalg.norm(vec)
    spin2 = float((vec.conj() @ s2_matrix(n) @ vec).real)
    log.info("<S^2> after the random per-bond circuit: %.1e", spin2)
    r.append(check(abs(spin2) < 1e-9, "<S^2> = 0 for random per-bond angles"))


def check_depth(r, ctx):
    """Check 2: 10 spins, 4 / 8 / 14 layers, HVA versus per-bond."""
    log.info("=== Check 2: depth (10 spins, exact reference) ===")
    n, chi = 10, 32
    # The exact E0 comes from sparse diagonalization (1024 states).
    e0 = ground_energy(n)
    depths = ctx["depths"]
    # res[layers] = (HVA energy, per-bond energy, per-bond angles, HVA angles).
    res = staged(n, chi, depths, ctx["iters"], ctx["seed"])
    herr = {k: v[0] - e0 for k, v in res.items()}
    perr = {k: v[1] - e0 for k, v in res.items()}
    for k in depths:
        log.info(
            "%2d layers: HVA error %.2e | per-bond error %.2e | ratio %.1fx",
            k,
            herr[k],
            perr[k],
            herr[k] / perr[k],
        )
    r.append(check(min(*herr.values(), *perr.values()) > -1e-9, "no energy below E0"))
    r.append(
        check(
            all(perr[k] < 0.5 * herr[k] for k in depths),
            "per-bond error is under half the HVA error at every depth",
        )
    )
    ratios = [herr[k] / perr[k] for k in depths]
    # Lab 40 found about 100x at shallow depth; the gap closes as the HVA
    # gains layers (my first guess, that it widens with depth, was wrong).
    r.append(
        check(
            ratios[0] > 50 and ratios[-1] < ratios[0],
            f"the advantage is largest when shallow ({ratios[0]:.0f}x) and shrinks",
        )
    )
    r.append(
        check(
            all(perr[b] < perr[a] for a, b in zip(depths, depths[1:])),
            "per-bond error falls at every stage",
        )
    )
    # The same accuracy for a fraction of the CNOTs: 4 per-bond layers against
    # the deepest HVA (15 CNOTs per layer at 10 spins is not needed; the layer
    # count is the ratio).
    r.append(
        check(
            perr[depths[0]] < herr[depths[-1]],
            f"{depths[0]} per-bond layers beat {depths[-1]} HVA layers "
            f"({depths[-1] / depths[0]:.1f}x fewer gates)",
        )
    )
    r.append(
        check(
            all(herr[b] < herr[a] for a, b in zip(depths, depths[1:])),
            "the HVA improves with depth, but needs far more layers",
        )
    )
    # Verify the deepest per-bond result independently of the MPS code.
    x = res[depths[-1]][2]
    sv = general_energy(n, perbond_steps(n, x))
    r.append(
        check(
            abs(sv - res[depths[-1]][1]) < 1e-8,
            "deepest state: MPS energy equals the state vector",
        )
    )
    vec = perbond_state(n, x, chi).to_vector()
    vec = vec / np.linalg.norm(vec)
    spin2 = float((vec.conj() @ s2_matrix(n) @ vec).real)
    r.append(check(abs(spin2) < 1e-8, f"deepest state keeps <S^2> = {spin2:.0e}"))


def check_larger(r, ctx):
    """Check 3: 14 spins at 6 layers, with a truncation check."""
    log.info("=== Check 3: a larger chain (14 spins, chi = 32) ===")
    n, chi, layers = 14, 32, 6
    # Sparse diagonalization of 16384 states is still quick.
    e0 = ground_energy(n)
    res = staged(n, chi, [layers], ctx["iters_large"], ctx["seed"])
    e_hva, e_per, x_per, _ = res[layers]
    log.info(
        "14 spins, %d layers: HVA error %.2e, per-bond error %.2e (%.1fx)",
        layers,
        e_hva - e0,
        e_per - e0,
        (e_hva - e0) / (e_per - e0),
    )
    r.append(check(e_per > e0 - 1e-6, "per-bond energy is not below the exact E0"))
    r.append(
        check(
            e_per - e0 < 0.5 * (e_hva - e0), "per-bond error under half the HVA error"
        )
    )
    # Truncation: the same angles on a larger bond dimension. If the energy
    # does not move, the discarded Schmidt weight did not matter.
    e_big, _, disc_big = perbond_energy_grad(x_per, n, 96)
    _, _, disc = perbond_energy_grad(x_per, n, chi)
    log.info(
        "chi=32 energy %.8f, chi=96 energy %.8f, discarded %.1e / %.1e",
        e_per,
        e_big,
        disc,
        disc_big,
    )
    r.append(
        check(
            abs(e_per - e_big) < 1e-3 * max(e_per - e0, 1e-3) + 1e-4,
            "chi=32 is converged for these angles",
        )
    )
    r.append(check(disc < 1e-3, f"discarded weight {disc:.1e} is small"))


def main(argv=None):
    """Run every Lab 45 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 45: per-bond ansatz on the MPS")
    # The seed fixes the small random offsets of every optimiser start.
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fewer stages and iterations")
    args = ap.parse_args(argv)
    ctx = {
        "seed": args.seed,
        "depths": [4, 8] if args.quick else [4, 8, 14],
        "iters": 100 if args.quick else 130,
        "iters_large": 40 if args.quick else 60,
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_gradient(r, ctx)
    check_depth(r, ctx)
    check_larger(r, ctx)
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
