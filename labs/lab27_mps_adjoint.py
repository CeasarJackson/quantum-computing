#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab27_mps_adjoint.py

Purpose
-------
Lab 26 trained the circuit with the MPS inside the optimiser, but with a
finite-difference gradient: one extra circuit run per parameter, and noisy
when the bond dimension truncates. Lab 23 showed the cure for state vectors,
the ADJOINT method: one forward pass, one backward pass, all derivatives. Lab 27
builds the adjoint method for matrix product states.

The ingredients, all in labs/mpslib.py:
  * the Hamiltonian as an MPO applied to the final MPS, then re-compressed:
    |phi> = H |psi_final>   (bond dimension x5, then truncated back down);
  * an UN-NORMALISED MPS (the norm of |phi> is physical, so truncations must
    not rescale it);
  * overlaps <phi| h_bond |psi> for all bonds of a gate group from one set of
    left and right environments;
  * a backward sweep that undoes the circuit step by step on |phi>.

The gradient of the energy with respect to the angle of step k is
    dE/dtheta_k = 2 Im < phi_k | G_k | psi_k >,
where G_k is the step's group Hamiltonian, psi_k the state after step k and
phi_k = (later steps)^dagger H psi_final.

The lab verifies:
  1. MPO application and compression reproduce H|psi> exactly (small n).
  2. A gate on an un-normalised MPS keeps its norm.
  3. The bond-overlap routine equals the state-vector value.
  4. The adjoint gradient equals the exact state-vector adjoint gradient.
  5. Its error shrinks with the bond dimension, and it is smoother than finite
     differences at a bond dimension where truncation matters.
  6. The cost grows linearly with the number of layers, finite differences
     quadratically.
  7. Training with the MPS adjoint follows the state-vector training path, and
     then runs on a chain far too long for it.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab27_mps_adjoint.py
    python3 labs/lab27_mps_adjoint.py --chi 48 --layers 4
    python3 labs/lab27_mps_adjoint.py --quick

Log files
---------
logs/lab27/lab27_<YYYYmmdd_HHMMSS>.log

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
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import (
    BOND,
    DMRG,
    MPS,
    apply_mpo,
    bond_overlaps,
    heisenberg_mpo,
    layer_plan,
    mps_energy_and_adjoint_grad,
    run_hva,
    singlet_vector,
    sparse_hamiltonian,
    sv_energy_and_grad,
    train_angles,
)

log = get_logger("lab27")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def fd_energy_and_grad(params, n, chi, eps=1e-6):
    """Finite-difference gradient with cached forward states (Lab 26 method)."""
    steps = layer_plan(params)
    start = MPS.singlet_product(n, chi)
    start.cutoff = 1e-10
    states = [start]
    for parity, angle in steps:
        nxt = states[-1].copy()
        nxt.apply_group(parity, angle)
        states.append(nxt)
    energy = states[-1].copy().energy()
    grad = np.zeros(len(steps))
    for k, (parity, angle) in enumerate(steps):
        trial = states[k].copy()
        trial.apply_group(parity, angle + eps)
        for later_parity, later_angle in steps[k + 1 :]:
            trial.apply_group(later_parity, later_angle)
        grad[k] = (trial.energy() - energy) / eps
    return energy, grad


def train_depths(fun, max_layers, rng, maxiter):
    """Warm-started L-BFGS over depths 1..max_layers for any energy+gradient.

    Two starts per depth (previous solution, and a jiggled copy of it); the
    random draws depend only on `rng`, so two calls with identically seeded
    generators follow the same path if their gradients agree.
    """
    prev, out = np.zeros(0), []
    for p in range(1, max_layers + 1):
        warm = np.concatenate([prev, np.zeros(2)])
        starts = [warm, warm + rng.normal(0, 0.1, 2 * p)]
        best = None
        for x0 in starts:
            res = minimize(
                fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}
            )
            if best is None or res.fun < best.fun:
                best = res
        prev = best.x
        out.append((best.fun, best.x))
    return out


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_building_blocks(r, rng):
    """Checks 1-3: MPO application, norm tracking, bond overlaps."""
    log.info("=== Checks 1-3: the building blocks (n=10) ===")
    n = 10
    # Small enough for exact state-vector reference values.
    h = sparse_hamiltonian(n)
    psi = run_hva(n, rng.uniform(-0.6, 0.6, 8), 64)
    vec = psi.to_vector()
    # chi = 2^(n/2) = 32 per cut after H (rank can at most multiply by 5, but
    # never exceeds the exact limit): nothing is truncated.
    hpsi = apply_mpo(psi.copy(), heisenberg_mpo(n), 32)
    err = np.linalg.norm(hpsi.to_vector() - h @ vec)
    r.append(
        check(
            err < 1e-10,
            f"MPO application equals H|psi> (error "
            f"{err:.1e}, norm {hpsi.norm():.6f})",
        )
    )
    # A gate is unitary, so it must not change the norm of an un-normalised
    # state either, as long as nothing is truncated.
    before = hpsi.norm()
    hpsi.apply_group(1, 0.37)
    r.append(
        check(
            abs(hpsi.norm() - before) < 1e-9,
            "a gate on the un-normalised H|psi> keeps its norm",
        )
    )
    # <phi|h_q|psi> from environments versus the state-vector value.
    # A different random state plays the bra, so the overlap is not just an
    # expectation value and any bra/ket mix-up would show.
    phi = run_hva(n, rng.uniform(-0.6, 0.6, 8), 64)
    bonds = list(range(n - 1))
    vals = bond_overlaps(phi, psi, bonds, BOND)
    ref = []
    pv = phi.to_vector()
    # Reference: apply the 4x4 bond term to the ket vector, then take vdot.
    for q in bonds:
        t = vec.reshape(2 ** (n - q - 2), 4, 2**q)
        op = np.einsum("ab,ibj->iaj", BOND, t).reshape(-1)
        ref.append(np.vdot(pv, op))
    gap = max(abs(a - b) for a, b in zip(vals, ref))
    r.append(
        check(
            gap < 1e-10,
            f"bond overlaps equal the state-vector values "
            f"(max difference {gap:.1e})",
        )
    )


def check_exact_gradient(r, rng):
    """Check 4: the adjoint gradient equals the exact state-vector one."""
    log.info("=== Check 4: adjoint gradient versus the exact adjoint (n=12) ===")
    n = 12
    h = sparse_hamiltonian(n)
    x = rng.uniform(-0.6, 0.6, 8)
    e_sv, g_sv = sv_energy_and_grad(x, n, h, singlet_vector(n))
    # chi = 64 = 2^(n/2) and chi_phi = 256 leave nothing to truncate.
    e_mps, g_mps = mps_energy_and_adjoint_grad(x, n, 64, 256)
    gap = float(np.max(np.abs(g_mps - g_sv)))
    r.append(
        check(
            abs(e_mps - e_sv) < 1e-10 and gap < 1e-9,
            f"energy differs by {abs(e_mps - e_sv):.1e}, gradient by " f"{gap:.1e}",
        )
    )


def check_accuracy(r, ctx):
    """Check 5: gradient error against bond dimension, and against FD."""
    n = ctx["acc_n"]
    log.info("=== Check 5: truncation error of the gradient (n=%d) ===", n)
    x = ctx["angles"]
    # The reference is the same method at a bond dimension far above need.
    _, g_ref = mps_energy_and_adjoint_grad(x, n, 128, 256)
    adj, fd = [], []
    # chi_phi = 2 chi: H|psi> is compressed to twice the state's bond limit.
    for chi in (8, 16, 32):
        _, g_a = mps_energy_and_adjoint_grad(x, n, chi, 2 * chi)
        _, g_f = fd_energy_and_grad(x, n, chi)
        adj.append(float(np.max(np.abs(g_a - g_ref))))
        fd.append(float(np.max(np.abs(g_f - g_ref))))
        log.info(
            "chi=%2d: max |gradient error|  adjoint %.2e | finite " "differences %.2e",
            chi,
            adj[-1],
            fd[-1],
        )
    r.append(
        check(adj[-1] < adj[0], "the adjoint gradient error shrinks as chi " "grows")
    )
    r.append(check(adj[-1] < 2e-2, f"chi=32 gradient error {adj[-1]:.1e} < 2e-2"))
    # Finite differences divide a truncation jump by eps = 1e-6; the adjoint
    # method never differentiates a truncated quantity, so once chi is large
    # enough to be useful it is the more accurate of the two.
    r.append(
        check(
            adj[-1] < fd[-1],
            f"at chi=32 the adjoint gradient ({adj[-1]:.1e}) is more "
            f"accurate than finite differences ({fd[-1]:.1e})",
        )
    )


def check_cost(r, ctx):
    """Check 6: time per gradient against the number of steps."""
    n = ctx["cost_n"]
    log.info("=== Check 6: cost versus number of steps (n=%d, chi=32) ===", n)
    rng = np.random.default_rng(3)
    t_adj, t_fd = {}, {}
    for steps in ctx["cost_steps"]:
        # Small random angles keep the entanglement (and the SVD cost) the
        # same at every size, so only the step count changes.
        x = rng.normal(0, 0.15, steps)
        t0 = time.perf_counter()
        mps_energy_and_adjoint_grad(x, n, 32, 64)
        t_adj[steps] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fd_energy_and_grad(x, n, 32)
        t_fd[steps] = time.perf_counter() - t0
        log.info(
            "%2d steps: adjoint %.2f s | finite differences %.2f s",
            steps,
            t_adj[steps],
            t_fd[steps],
        )
    # Compare the growth between the smallest and the largest step count.
    lo, hi = ctx["cost_steps"][0], ctx["cost_steps"][-1]
    g_adj, g_fd = t_adj[hi] / t_adj[lo], t_fd[hi] / t_fd[lo]
    log.info(
        "growth from %d to %d steps (x%.0f): adjoint x%.1f | finite "
        "differences x%.1f",
        lo,
        hi,
        hi / lo,
        g_adj,
        g_fd,
    )
    r.append(
        check(
            g_adj < g_fd,
            "finite differences grow faster with the "
            "number of steps than the adjoint method",
        )
    )
    r.append(
        check(
            t_adj[hi] < t_fd[hi],
            f"at {hi} steps the adjoint gradient is cheaper "
            f"(x{t_fd[hi] / t_adj[hi]:.1f})",
        )
    )


def check_training(r, ctx):
    """Check 7: training with the adjoint, then beyond state-vector reach."""
    n, depths = ctx["train_n"], ctx["depths"]
    log.info("=== Check 7: training (n=%d, %d layers) ===", n, depths)
    h = sparse_hamiltonian(n)
    psi0 = singlet_vector(n)
    e_gs = float(eigsh(h, k=1, which="SA", return_eigenvectors=False)[0].real)
    # Identically seeded generators make both runs draw the same jiggles, so
    # any difference between them comes from the gradients alone.
    sv = train_depths(
        lambda x: sv_energy_and_grad(x, n, h, psi0),
        depths,
        np.random.default_rng(5),
        60,
    )
    ad = train_depths(
        lambda x: mps_energy_and_adjoint_grad(x, n, 48, 96),
        depths,
        np.random.default_rng(5),
        60,
    )
    for p, ((e1, _), (e2, _)) in enumerate(zip(sv, ad), 1):
        log.info(
            "depth %d: error state-vector %.5f | MPS adjoint %.5f",
            p,
            e1 - e_gs,
            e2 - e_gs,
        )
    gaps = [abs(a[0] - b[0]) for a, b in zip(sv, ad)]
    r.append(
        check(
            max(gaps) < 2e-3,
            f"MPS-adjoint training reaches the state-vector energy at "
            f"every depth (largest gap {max(gaps):.1e})",
        )
    )
    r.append(
        check(
            all(e > e_gs - 1e-7 for e, _ in ad),
            "and never goes below the exact ground energy",
        )
    )
    # Now a chain the state vector cannot hold comfortably.
    big = ctx["big_n"]
    # DMRG supplies the exact answer to judge the long-chain training.
    solver = DMRG(heisenberg_mpo(big), 32)
    e_big = solver.run(max_sweeps=30, tol=1e-9)[-1]
    t0 = time.perf_counter()
    res = train_depths(
        lambda x: mps_energy_and_adjoint_grad(x, big, 32, 64),
        ctx["big_depths"],
        np.random.default_rng(5),
        40,
    )
    errs = [e - e_big for e, _ in res]
    log.info(
        "n=%d (DMRG E0 = %.5f): error by depth %s (%.0f s)",
        big,
        e_big,
        " ".join(f"{e:.4f}" for e in errs),
        time.perf_counter() - t0,
    )
    r.append(
        check(
            all(b <= a + 1e-3 for a, b in zip(errs, errs[1:])) and errs[-1] > -1e-6,
            f"n={big}: the error falls with depth and stays above DMRG",
        )
    )
    ctx["big_errors"] = errs


def main(argv=None):
    """Run every Lab 27 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 27: adjoint gradient for MPS")
    ap.add_argument("--layers", type=int, default=3, help="depths in check 7")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if not 1 <= args.layers <= 5:
        ap.error(f"--layers must be 1..5. Got {args.layers}.")
    rng = np.random.default_rng(args.seed)
    quick = args.quick
    # Everything the checks share lives in one dict. Quick mode shrinks the
    # chains and step counts so the whole lab finishes in about 15 s.
    ctx = {
        "acc_n": 16 if quick else 20,
        "cost_n": 16 if quick else 24,
        "cost_steps": [4, 20] if quick else [4, 12, 20],
        "train_n": 10 if quick else 12,
        "depths": 2 if quick else args.layers,
        "big_n": 20 if quick else 32,
        "big_depths": 2 if quick else args.layers,
    }
    # The trained angles of Lab 26 serve as a realistic, low-entanglement point
    # at which to measure gradient accuracy.
    n0 = 10 if quick else 12
    e0 = float(
        eigsh(sparse_hamiltonian(n0), k=1, which="SA", return_eigenvectors=False)[
            0
        ].real
    )
    ctx["angles"] = train_angles(n0, e0, 4 if quick else 6, rng, extra_starts=0)[-1][1]
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_building_blocks(r, rng)
    check_exact_gradient(r, rng)
    check_accuracy(r, ctx)
    check_cost(r, ctx)
    check_training(r, ctx)
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
