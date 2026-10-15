#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab26_hva_vs_dmrg.py

Purpose
-------
Labs 16, 20, 23 and 24 prepared the Heisenberg ground state with a layered
variational circuit (the HVA) but could only judge it against exact answers
up to n = 18. Lab 25 built a DMRG solver that is essentially exact to n = 64
and beyond. Lab 26 uses it as the JUDGE of the circuit on long chains:

  * How far above the true ground energy is the HVA state at n = 16...48, and
    does the error per site settle to a constant (a bulk property)?
  * How much of the true ground STATE does it capture? The overlap (fidelity)
    between two MPS is a cheap contraction, so we can measure it directly.
  * Does a finite-depth circuit have the right ENTANGLEMENT? The true ground
    state of a critical chain has entropy growing like (1/6) ln n, while a
    circuit of fixed depth can only build a bounded amount.
  * Are LOCAL quantities (bond energies in the middle of the chain) right even
    when the global state is not?
  * Can the circuit be trained with the MPS simulator INSIDE the optimiser
    (finite-difference gradient), and do the angles from n = 12 still hold?

The lab verifies:
  1. Angles trained at n = 12 (state vector) give the same energy in the MPS.
  2. The HVA energy is above the DMRG ground energy at every size (variational).
  3. The error per site is a constant: the energy error grows linearly in n.
  4. The fidelity with the true ground state decays exponentially with n.
  5. The circuit's half-chain entropy saturates while the ground state's grows.
  6. Central bond energies are close to the true ones.
  7. Training with the MPS inside the optimiser matches state-vector training.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab26_hva_vs_dmrg.py
    python3 labs/lab26_hva_vs_dmrg.py --layers 6 --chi 32
    python3 labs/lab26_hva_vs_dmrg.py --quick

Log files
---------
logs/lab26/lab26_<YYYYmmdd_HHMMSS>.log

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
    DMRG,
    MPS,
    heisenberg_mpo,
    layer_plan,
    overlap,
    run_hva,
    singlet_vector,
    sparse_hamiltonian,
    sv_apply_group,
    sv_energy_and_grad,
    train_angles,
)

log = get_logger("lab26")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def mps_energy_and_grad(params, n, chi, eps=1e-6):
    """Energy of the HVA state on n sites and its finite-difference gradient.

    The MPS has no cheap adjoint pass, so the gradient is a forward difference.
    To keep the cost down the state after every step is cached: the derivative
    for step k restarts from the cached state before step k, so the work per
    gradient is about half of (number of steps) full circuit runs.
    """
    steps = layer_plan(params)
    start = MPS.singlet_product(n, chi)
    # A tight singular-value cutoff keeps the objective smooth: a coarse one
    # makes the finite difference jump whenever the truncation point moves.
    start.cutoff = 1e-10
    states = [start]
    for parity, angle in steps:
        nxt = states[-1].copy()
        nxt.apply_group(parity, angle)
        states.append(nxt)
    # energy() moves the centre, so work on a copy to keep the cached state.
    energy = states[-1].copy().energy()
    grad = np.zeros(len(steps))
    for k, (parity, angle) in enumerate(steps):
        trial = states[k].copy()
        trial.apply_group(parity, angle + eps)
        for later_parity, later_angle in steps[k + 1 :]:
            trial.apply_group(later_parity, later_angle)
        grad[k] = (trial.energy() - energy) / eps
    return energy, grad


def dmrg_ground(n, chi):
    """DMRG ground state of the n-site chain: (energy, MPS, half-entropy)."""
    # tol = 1e-9 is far below every effect measured in this lab.
    solver = DMRG(heisenberg_mpo(n), chi)
    history = solver.run(max_sweeps=30, tol=1e-9)
    return history[-1], solver.mps.copy(), solver.mid_entropy


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_training(r, ctx):
    """Check 1: train at n = 12 and confirm the MPS reproduces the energy."""
    n = ctx["train_n"]
    # Small enough for the exact state-vector simulator, so the angles are the
    # best this ansatz can do and nothing about them is approximate.
    log.info("=== Check 1: train the angles at n=%d (state vector) ===", n)
    vals = eigsh(sparse_hamiltonian(n), k=1, which="SA", return_eigenvectors=False)
    e0 = float(vals[0].real)
    trained = train_angles(n, e0, ctx["layers"], ctx["rng"], extra_starts=1)
    # Warm starts make the depth sequence monotone: depth p can always copy
    # depth p-1 and add an identity layer.
    errs = [e for e, _ in trained]
    log.info(
        "error above the exact ground energy by depth: %s",
        " ".join(f"{e:.1e}" for e in errs),
    )
    ctx["angles"] = trained[-1][1]
    r.append(
        check(
            all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])),
            "warm-started error never rises with depth",
        )
    )
    # chi = 2^(n/2) means no truncation, so MPS and state vector must agree.
    mps = run_hva(n, ctx["angles"], 2 ** (n // 2))
    h = sparse_hamiltonian(n)
    psi = singlet_vector(n)
    for parity, angle in layer_plan(ctx["angles"]):
        psi = sv_apply_group(psi, n, parity, angle)
    e_sv = float(np.vdot(psi, h @ psi).real)
    r.append(
        check(
            abs(mps.energy() - e_sv) < 1e-9,
            f"the MPS circuit gives the state-vector energy {e_sv:.8f}",
        )
    )


def check_judge(r, ctx):
    """Checks 2 and 3: HVA error against DMRG on long chains."""
    log.info("=== Check 2: HVA versus the DMRG ground state ===")
    ctx["gs"], ctx["hva"], errors = {}, {}, {}
    # Keep both states per size: later checks need them for overlaps,
    # entropies and local energies, and recomputing DMRG would be wasteful.
    for n in ctx["sizes"]:
        t0 = time.perf_counter()
        e_gs, gs, s_gs = dmrg_ground(n, ctx["chi"])
        mps = run_hva(n, ctx["angles"], 128)
        e_hva = mps.energy()
        ctx["gs"][n] = (e_gs, gs, s_gs)
        ctx["hva"][n] = mps
        errors[n] = e_hva - e_gs
        log.info(
            "n=%2d: E_DMRG = %.6f | E_HVA = %.6f | error %.4f "
            "(%.4f per site, %.1f s)",
            n,
            e_gs,
            e_hva,
            errors[n],
            errors[n] / n,
            time.perf_counter() - t0,
        )
    r.append(
        check(
            all(e > -1e-7 for e in errors.values()),
            "variational: the HVA energy is never below the DMRG energy",
        )
    )
    # Cross-check DMRG itself against Lanczos at the smallest size.
    n0 = ctx["sizes"][0]
    if n0 <= 16:
        e_ex = float(
            eigsh(sparse_hamiltonian(n0), k=1, which="SA", return_eigenvectors=False)[
                0
            ].real
        )
        r.append(
            check(
                abs(ctx["gs"][n0][0] - e_ex) < 1e-7,
                f"n={n0}: DMRG equals exact diagonalisation",
            )
        )
    log.info("=== Check 3: the error grows linearly with n ===")
    # E_error(n) = bulk * n + surface. The slope between sizes isolates the
    # bulk term; the (n=16) pair is excluded because the circuit's light cone
    # has not yet separated the two ends there.
    sizes = ctx["sizes"]
    slopes = [(errors[b] - errors[a]) / (b - a) for a, b in zip(sizes, sizes[1:])]
    log.info(
        "bulk error per site between neighbouring sizes: %s",
        " ".join(f"{s:.4f}" for s in slopes),
    )
    spread = max(slopes[1:]) - min(slopes[1:])
    r.append(
        check(
            spread < 2e-3,
            f"the bulk error per site is constant for the longer chains "
            f"(spread {spread:.1e})",
        )
    )
    ctx["bulk_error"] = slopes[-1]


def check_fidelity(r, ctx):
    """Check 4: the overlap with the true ground state decays with n."""
    log.info("=== Check 4: fidelity with the true ground state ===")
    # |<DMRG|HVA>|^2 is a transfer-matrix contraction costing n chi^3, so it
    # works at sizes where the full 2^n vectors could never be compared.
    sizes = ctx["sizes"]
    fid = {}
    for n in sizes:
        fid[n] = abs(overlap(ctx["gs"][n][1], ctx["hva"][n])) ** 2
        log.info(
            "n=%2d: fidelity %.4f, -ln F / n = %.4f", n, fid[n], -np.log(fid[n]) / n
        )
    # For a product of independent local errors F = exp(-eps n), so -ln F is a
    # straight line in n and eps is the infidelity per site.
    logf = [-np.log(fid[n]) for n in sizes]
    r.append(
        check(
            all(b > a for a, b in zip(logf, logf[1:])),
            "the infidelity -ln F grows with n",
        )
    )
    rate = np.polyfit(sizes, logf, 1)
    resid = max(abs(np.polyval(rate, n) - lf) for n, lf in zip(sizes, logf))
    log.info("-ln F = %.4f n %+.3f (largest deviation %.3f)", rate[0], rate[1], resid)
    r.append(
        check(
            resid < 0.1 * max(logf) + 0.02,
            "-ln F is close to linear in n (exponential decay of F)",
        )
    )
    # An exponential decay means the fidelity of a long enough chain is tiny
    # even though every local piece is accurate: extrapolate the fit.
    proj = float(np.exp(-np.polyval(rate, 200)))
    log.info("extrapolated fidelity at n=200: %.3f", proj)
    r.append(
        check(
            rate[0] > 0.003 and proj < 0.4,
            f"infidelity per site {rate[0]:.4f}: F(n=200) is only " f"{proj:.2f}",
        )
    )


def check_entropy(r, ctx):
    """Check 5: bounded circuit entanglement against logarithmic growth."""
    log.info("=== Check 5: half-chain entanglement ===")
    sizes = ctx["sizes"]
    # entropy() moves the orthogonality centre, which is harmless: it changes
    # the gauge of the tensors, not the state.
    s_hva = [ctx["hva"][n].entropy(n // 2 - 1) for n in sizes]
    s_gs = [ctx["gs"][n][2] for n in sizes]
    for n, a, b in zip(sizes, s_hva, s_gs):
        log.info("n=%2d: S(HVA) = %.3f | S(ground state) = %.3f", n, a, b)
    # A circuit of depth p has light cone 2p, so beyond n ~ 4p its entropy at
    # the centre cannot change with n at all.
    r.append(
        check(
            abs(s_hva[-1] - s_hva[-2]) < 0.02,
            "the circuit's entropy has saturated (light cone)",
        )
    )
    r.append(
        check(s_gs[-1] - s_gs[-2] > 0.03, "the ground state's entropy is still growing")
    )
    r.append(
        check(
            s_hva[-1] < s_gs[-1],
            "the circuit is under-entangled compared with the truth",
        )
    )


def check_local(r, ctx):
    """Check 6: bond energies in the middle of the chain."""
    n = ctx["sizes"][-1]
    log.info("=== Check 6: local bond energies in the middle (n=%d) ===", n)
    # Twelve central bonds are far enough from both ends (the open-end
    # alternation of Lab 25 has faded) to represent the bulk.
    lo, hi = n // 2 - 6, n // 2 + 6
    b_gs = ctx["gs"][n][1].copy().bond_energies()
    b_hva = ctx["hva"][n].bond_energies()
    m_gs, m_hva = np.mean(b_gs[lo:hi]), np.mean(b_hva[lo:hi])
    log.info(
        "mean of the 12 central bonds: DMRG %.4f | HVA %.4f | " "difference %.4f",
        m_gs,
        m_hva,
        m_hva - m_gs,
    )
    r.append(
        check(
            0 < m_hva - m_gs < 0.06,
            "the HVA central bonds are close to (and above) the true " "ones",
        )
    )
    r.append(
        check(
            abs((m_hva - m_gs) - ctx["bulk_error"]) < 0.02,
            "the central-bond error matches the bulk error per site",
        )
    )


def check_finetune(r, ctx):
    """Check 7: gradient and training with the MPS inside the optimiser."""
    n = ctx["tune_n"]
    log.info("=== Check 7: training with the MPS inside the optimiser (n=%d) " "===", n)
    h = sparse_hamiltonian(n)
    psi0 = singlet_vector(n)
    e_gs = float(eigsh(h, k=1, which="SA", return_eigenvectors=False)[0].real)
    x0 = ctx["angles"]
    # (a) The finite-difference MPS gradient must match the exact adjoint one.
    f_mps, g_mps = mps_energy_and_grad(x0, n, 64)
    f_sv, g_sv = sv_energy_and_grad(x0, n, h, psi0)
    gap = float(np.max(np.abs(g_mps - g_sv)))
    r.append(
        check(
            abs(f_mps - f_sv) < 1e-4 and gap < 5e-3,
            f"MPS energy and gradient match the adjoint method "
            f"(energy {abs(f_mps - f_sv):.1e}, gradient {gap:.1e})",
        )
    )
    # (b) Same optimiser, same start, same iteration budget: the two runs
    # should follow nearly the same path and end at the same energy.
    iters = ctx["tune_iters"]
    t0 = time.perf_counter()
    res_mps = minimize(
        mps_energy_and_grad,
        x0,
        args=(n, 64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": iters},
    )
    secs = time.perf_counter() - t0
    res_sv = minimize(
        sv_energy_and_grad,
        x0,
        args=(n, h, psi0),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": iters},
    )
    log.info(
        "n=%d, %d iterations: error %.5f -> MPS-trained %.5f | "
        "state-vector-trained %.5f (MPS run took %.1f s)",
        n,
        iters,
        f_sv - e_gs,
        res_mps.fun - e_gs,
        res_sv.fun - e_gs,
        secs,
    )
    r.append(
        check(
            res_mps.fun < f_mps - 1e-5,
            f"training on the MPS lowers the n={n} energy by "
            f"{f_mps - res_mps.fun:.1e}",
        )
    )
    r.append(
        check(
            abs(res_mps.fun - res_sv.fun) < 1e-4,
            "the MPS-trained energy matches the state-vector-trained one",
        )
    )
    r.append(
        check(res_mps.fun > e_gs - 1e-7, "and it is still above the true ground energy")
    )
    # The tiny gain also says something physical: angles trained at n = 12
    # are already almost optimal for longer chains.
    log.info(
        "angles trained at n=%d are within %.1e of the n=%d optimum " "found here",
        ctx["train_n"],
        f_sv - res_sv.fun,
        n,
    )


def main(argv=None):
    """Run every Lab 26 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 26: HVA judged by DMRG")
    ap.add_argument("--layers", type=int, default=6, help="HVA layers 2..8")
    ap.add_argument("--chi", type=int, default=32, help="DMRG bond dimension")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if not 2 <= args.layers <= 8:
        ap.error(f"--layers must be 2..8. Got {args.layers}.")
    if not 16 <= args.chi <= 128:
        ap.error(f"--chi must be 16..128. Got {args.chi}.")
    quick = args.quick
    # Everything the checks share lives in one dict so each check function
    # keeps a short signature.
    ctx = {
        "rng": np.random.default_rng(args.seed),
        "layers": min(args.layers, 4) if quick else args.layers,
        "chi": min(args.chi, 24) if quick else args.chi,
        "train_n": 10 if quick else 12,
        "sizes": [16, 24, 32] if quick else [16, 24, 32, 48],
        "tune_n": 12 if quick else 16,
        "tune_iters": 6 if quick else 15,
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_training(r, ctx)
    check_judge(r, ctx)
    check_fidelity(r, ctx)
    check_entropy(r, ctx)
    check_local(r, ctx)
    check_finetune(r, ctx)
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
