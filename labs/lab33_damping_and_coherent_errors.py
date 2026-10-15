#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab33_damping_and_coherent_errors.py

Purpose
-------
Labs 29-31 used random Pauli errors. Real devices also have two other kinds of
error that behave quite differently, and this lab studies both on the Heisenberg
Hamiltonian Variational Ansatz.

PART A. AMPLITUDE DAMPING (T1 decay). The channel rho -> K0 rho K0^+ + K1 rho
K1^+ with K0 = diag(1, sqrt(1-g)) and K1 = sqrt(g) |0><1| is NOT unitary. On a
matrix product state it is simulated by quantum jumps: a site jumps |1> -> |0>
with probability g * P1, otherwise it evolves with K0, and the state is
renormalised either way (noiselib.damp_site). Findings:
  * the energy bias is large (it pulls the chain towards the all-|0> state);
  * every jump changes total S_z by one unit and nothing changes it back, while
    K0 is diagonal and conserves S_z, so S_z verification keeps exactly the
    no-jump trajectories and removes almost all of the bias.

PART B. COHERENT ERRORS: small unitary mistakes that are the SAME every time
(a frozen detuning, a mis-scaled gate angle). They add in amplitude, not in
probability, which changes how they respond to the mitigation of Lab 30:
  * a uniform over-rotation is a global rotation, which leaves the singlet
    state alone, so it is far milder than a random error of equal strength
    (and the axis does not matter: SU(2) symmetry);
  * a frozen non-uniform detuning grows like fold^2 under unitary folding,
    random errors like fold^1, so linear extrapolation makes it WORSE;
  * a mis-scaled gate angle is invisible to folding, because U(a) U(-a) = 1
    exactly: extrapolation returns the wrong energy unchanged.

The lab verifies:
  1. The jump machinery against closed-form cases and an independent
     state-vector population.
  2. Trajectories against the exact density matrix (n=6) and linear scaling
     of the bias with the damping rate.
  3. S_z verification of damped states: heralding of the no-jump branch, exact
     and sampled.
  4. Coherent errors on the MPS against the exact density matrix; the
     uniform over-rotation results.
  5. Fold scaling and extrapolation of frozen detuning versus random errors.
  6. Fold-blindness and quadratic scaling of gate-angle errors.
  7. Long chain, damping with trajectories, raw versus verified.
  8. Long chain, coherent errors (deterministic, no sampling).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab33_damping_and_coherent_errors.py
    python3 labs/lab33_damping_and_coherent_errors.py --trajectories 400
    python3 labs/lab33_damping_and_coherent_errors.py --quick

Log files
---------
logs/lab33/lab33_<YYYYmmdd_HHMMSS>.log

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
from mpslib import MPS, PAULI, run_hva, sparse_hamiltonian, train_angles
from noiselib import (
    amplitude_damping_kraus,
    coherent_mps,
    damp_site,
    damped_mps,
    dm_expectation,
    dm_kraus_state,
    dm_sz_from_rho,
    mean_se,
    richardson_weights,
    rotation,
    sz_ratio,
    sz_verified,
)

log = get_logger("lab33")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def dephasing_kraus(angle):
    """Random Z error with the same single-gate infidelity as a rotation.

    A rotation exp(-i a Z / 2) leaves the ideal state with probability
    cos^2(a/2). A Z flip with probability sin^2(a/2) has the same infidelity,
    so the two are a fair "coherent versus random" pair.
    """
    p = np.sin(angle / 2) ** 2
    return [np.sqrt(1 - p) * PAULI["I"], np.sqrt(p) * PAULI["Z"]]


def slope(small, large, ratio):
    """Local power-law exponent: error(large) / error(small) = ratio^slope."""
    return float(np.log(large / small) / np.log(ratio))


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_machinery(r, ctx):
    """Check 1: the jump step against closed-form cases."""
    log.info("=== Check 1: the jump machinery ===")
    rng = np.random.default_rng(ctx["seed"])
    # Every check starts from a case whose answer is known without simulation.
    ks = amplitude_damping_kraus(0.3)
    # Trace preservation: the Kraus operators must satisfy sum K^+ K = 1.
    total = sum(k.conj().T @ k for k in ks)
    r.append(check(np.allclose(total, np.eye(2)), "sum of K^+ K is the identity"))
    # Full damping (g = 1) must send every |1> to |0>, and nothing else.
    neel = MPS.neel(4, 8)
    ones = int(sum(np.argmax(np.abs(t[0, :, 0])) for t in neel.a))
    jumps = sum(damp_site(neel, q, 1.0, rng) for q in range(4))
    vec = neel.to_vector()
    r.append(
        check(
            abs(abs(vec[0]) - 1) < 1e-12 and jumps == ones,
            f"g=1 sends the Neel state to |0000> with {jumps} jumps",
        )
    )
    # No damping: never a jump, and the state is unchanged.
    # A realistic entangled state, so P1 is not trivially 0 or 1.
    state = run_hva(6, ctx["params6"][2], 32)
    before = state.to_vector()
    jumps = sum(damp_site(state, q, 0.0, rng) for q in range(6))
    r.append(
        check(
            jumps == 0 and abs(np.vdot(before, state.to_vector())) > 1 - 1e-12,
            "g=0 never jumps and leaves the state unchanged",
        )
    )
    # The jump probability g * P1, with P1 taken independently from the
    # state vector (site q is bit q of the basis index).
    g, site, draws = 0.4, 2, ctx["draws"]
    prob = np.abs(before) ** 2
    p1 = float(prob[(np.arange(64) >> site) & 1 == 1].sum())
    hits = 0
    for _ in range(draws):
        hits += damp_site(run_hva(6, ctx["params6"][2], 32), site, g, rng)
    want = g * p1
    se = (want * (1 - want) / draws) ** 0.5
    log.info("jump rate %.4f expected %.4f +- %.4f", hits / draws, want, se)
    r.append(check(abs(hits / draws - want) < 4 * se, "jump rate equals g * P1"))
    # A whole damped trajectory stays a normalised state.
    # Renormalising after every step is what keeps the unravelling valid.
    traj = damped_mps(6, ctx["params6"][2], 32, 0.05, rng)
    r.append(check(abs(traj.norm() - 1) < 1e-10, "a damped trajectory has norm 1"))
    # The exact density matrix keeps unit trace and reduces to the clean state.
    rho = dm_kraus_state(6, ctx["params6"][2], lambda q: ks)
    clean = run_hva(6, ctx["params6"][2], 32).energy()
    rho0 = dm_kraus_state(6, ctx["params6"][2], lambda q: None)
    r.append(
        check(
            abs(np.trace(rho.reshape(64, 64)) - 1) < 1e-10
            and abs(dm_expectation(rho0, 6) - clean) < 1e-9,
            "density matrix: unit trace, and no channel reproduces the clean energy",
        )
    )


def check_exact_damping(r, ctx):
    """Check 2: trajectories against the exact density matrix."""
    n, params = 6, ctx["params6"][3]
    g, count = ctx["g_small"], ctx["small_count"]
    log.info("=== Check 2: damping, n=6, g=%g, N=%d ===", g, count)
    # The density matrix is the ground truth that no sampling noise touches.
    clean = run_hva(n, params, 64).energy()
    rho = dm_kraus_state(n, params, lambda q: amplitude_damping_kraus(g))
    exact = dm_expectation(rho, n)
    # Each seed offset gives an independent stream, so checks do not share draws.
    rng = np.random.default_rng(ctx["seed"] + 2)
    energies = [damped_mps(n, params, 64, g, rng).energy() for _ in range(count)]
    mean, se = mean_se(np.array(energies))
    log.info("exact %.5f | trajectories %.5f +- %.5f", exact, mean, se)
    # The unravelled average must reproduce the channel (within noise).
    r.append(
        check(abs(mean - exact) < 4 * se, "trajectory mean equals the exact channel")
    )
    r.append(
        check(exact - clean > 0.1, f"the energy bias is large ({exact - clean:.3f})")
    )
    # Small rates: the bias is first order in g, so doubling g doubles it.
    bias = (
        lambda gg: dm_expectation(
            dm_kraus_state(n, params, lambda q: amplitude_damping_kraus(gg)), n
        )
        - clean
    )
    ratio = bias(2 * g) / bias(g)
    log.info("bias(2g) / bias(g) = %.3f", ratio)
    r.append(check(1.8 < ratio < 2.1, "the bias is linear in the damping rate"))
    ctx["exact_damped"] = (rho, exact, clean, g)


def check_verification_exact(r, ctx):
    """Check 3: S_z verification of damped states, exact and sampled."""
    n, params = 6, ctx["params6"][3]
    rho, exact, clean, g = ctx["exact_damped"]
    log.info("=== Check 3: S_z verification of damped states ===")
    # Project the exact state onto S_z = 0 and renormalise by the acceptance.
    ver, acc = dm_sz_from_rho(rho, n)
    log.info(
        "raw bias %.5f | verified bias %.2e | acceptance %.4f",
        exact - clean,
        ver - clean,
        acc,
    )
    r.append(
        check(
            abs(ver - clean) < 0.01 * (exact - clean),
            "verification removes more than 99% of the bias (exact)",
        )
    )
    # Sampled version: reuse the same trajectories' structure.
    # Sampled version of the same projection, one trajectory at a time.
    rng = np.random.default_rng(ctx["seed"] + 3)
    nums, accs, jump_acc = [], [], []
    for _ in range(ctx["small_count"]):
        mps = damped_mps(n, params, 64, g, rng)
        num, a = sz_verified(mps)
        nums.append(num)
        accs.append(a)
        if mps.flips:
            jump_acc.append(a)
    got, _ = sz_ratio(nums, accs)
    mean_acc = float(np.mean(accs))
    se_acc = (mean_acc * (1 - mean_acc) / len(accs)) ** 0.5
    log.info("sampled verified energy %.5f, acceptance %.4f", got, mean_acc)
    r.append(
        check(abs(got - ver) < 1e-3, "sampled verified energy matches the exact one")
    )
    r.append(check(abs(mean_acc - acc) < 4 * se_acc, "and so does the acceptance"))
    # The mechanism: any trajectory with at least one jump is rejected outright,
    # because jumps all move S_z the same way and nothing moves it back.
    r.append(
        check(
            len(jump_acc) > 0 and max(jump_acc) < 1e-9,
            f"all {len(jump_acc)} trajectories with a jump are rejected",
        )
    )


def check_coherent_exact(r, ctx):
    """Check 4: coherent errors against the density matrix; uniform case."""
    n, params = 6, ctx["params6"][3]
    clean = run_hva(n, params, 64).energy()
    log.info("=== Check 4: coherent errors, n=6 ===")
    # Small frozen angles (about 1 degree), the perturbative regime.
    errs = np.random.default_rng(ctx["seed"]).normal(0, 0.02, n)
    worst = 0.0
    for fold in (1, 3):
        mps = coherent_mps(n, params, 64, errs, "Z", fold)
        ops = lambda q: [rotation("Z", errs[q])]
        dm = dm_expectation(dm_kraus_state(n, params, ops, fold), n)
        worst = max(worst, abs(mps.energy() - dm))
    r.append(
        check(worst < 1e-9, f"MPS equals the density matrix (max diff {worst:.1e})")
    )
    # Uniform over-rotation: the axis is irrelevant for a singlet-based circuit.
    # One shared angle on every qubit: a global rotation of the whole chain.
    eps = 0.1
    z = coherent_mps(n, params, 64, np.full(n, eps), "Z").energy() - clean
    x = coherent_mps(n, params, 64, np.full(n, eps), "X").energy() - clean
    r.append(check(abs(z - x) < 1e-9, f"Z and X over-rotations agree ({z:.5f})"))
    # The random twin has the same single-gate infidelity but acts incoherently.
    twin = dm_expectation(dm_kraus_state(n, params, lambda q: dephasing_kraus(eps)), n)
    log.info("coherent %.5f vs random twin %.5f", z, twin - clean)
    r.append(
        check(
            z < 0.5 * (twin - clean),
            "a uniform coherent error is milder than its random twin",
        )
    )
    big = coherent_mps(n, params, 64, np.full(n, 2 * eps), "Z").energy() - clean
    r.append(
        check(3.5 < big / z < 4.5, f"the coherent bias is quadratic (x{big / z:.2f})")
    )


def check_fold_scaling(r, ctx):
    """Check 5: frozen detuning versus random errors under folding."""
    n, params = 6, ctx["params6"][3]
    clean = run_hva(n, params, 64).energy()
    log.info("=== Check 5: fold scaling and extrapolation, n=6 ===")
    errs = np.random.default_rng(ctx["seed"] + 5).normal(0, 0.02, n)
    # Odd fold factors multiply the gate count; see Lab 30.
    folds = (1, 3, 5)
    coh, rnd = [], []
    for fold in folds:
        coh.append(coherent_mps(n, params, 64, errs, "Z", fold).energy() - clean)
        ops = lambda q: dephasing_kraus(errs[q])
        rnd.append(dm_expectation(dm_kraus_state(n, params, ops, fold), n) - clean)
    s_coh, s_rnd = slope(coh[0], coh[2], 5), slope(rnd[0], rnd[2], 5)
    log.info(
        "bias at folds 1,3,5: coherent %s, random %s",
        np.round(coh, 4),
        np.round(rnd, 4),
    )
    log.info("growth exponent: coherent %.2f, random %.2f", s_coh, s_rnd)
    r.append(check(s_coh > 1.5, f"coherent bias grows like fold^{s_coh:.2f}"))
    r.append(check(0.9 < s_rnd < 1.1, f"random bias grows like fold^{s_rnd:.2f}"))
    # Extrapolate to zero noise with a line (folds 1, 3) and a parabola (1, 3, 5).
    # Two points allow a straight line, three allow a parabola.
    w2, w3 = richardson_weights(folds[:2]), richardson_weights(folds)
    lin, quad = float(w2 @ coh[:2]), float(w3 @ coh)
    log.info(
        "remaining coherent bias: raw %.4f, linear %.4f, quadratic %.4f",
        coh[0],
        lin,
        quad,
    )
    r.append(
        check(
            abs(lin) > abs(coh[0]), "linear extrapolation makes the coherent bias worse"
        )
    )
    r.append(check(abs(quad) < abs(lin), "a quadratic fit does better than a line"))
    lin_rnd = float(w2 @ rnd[:2])
    r.append(
        check(
            abs(lin_rnd) < 0.1 * abs(rnd[0]),
            "while a line removes 90% of the random-error bias",
        )
    )


def check_gate_error(r, ctx):
    """Check 6: a mis-scaled gate angle is invisible to folding."""
    n, params = 6, ctx["params6"][3]
    clean = run_hva(n, params, 64).energy()
    log.info("=== Check 6: gate-angle error, n=6 ===")
    vals = [
        coherent_mps(n, params, 64, None, "Z", f, 0.05).energy() - clean
        for f in (1, 3, 5)
    ]
    # Identical to rounding: the extra gates are exact inverses of each other.
    spread = max(vals) - min(vals)
    log.info("bias at folds 1,3,5: %s", np.round(vals, 6))
    r.append(
        check(vals[0] > 1e-3 and spread < 1e-9, "folding leaves the bias unchanged")
    )
    r.append(
        check(
            abs(float(richardson_weights([1, 3]) @ vals[:2]) - vals[0]) < 1e-9,
            "so extrapolation returns the raw bias",
        )
    )
    # Quadruple the angle error and compare: a power law has a fixed exponent.
    small = coherent_mps(n, params, 64, None, "Z", 1, 0.025).energy() - clean
    large = coherent_mps(n, params, 64, None, "Z", 1, 0.1).energy() - clean
    s = slope(small, large, 4)
    r.append(
        check(
            1.8 < s < 2.5,
            f"the bias is quadratic in the angle error (exponent {s:.2f})",
        )
    )


def check_long_damping(r, ctx):
    """Check 7: amplitude damping on the long chain."""
    n, chi, count, g = ctx["n"], ctx["chi"], ctx["count"], ctx["gamma"]
    params = ctx["params8"][3]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 7: n=%d, 4 layers, g=%g, N=%d ===", n, g, count)
    # Damping trajectories are never skipped: every one carries a continuous weight.
    rng = np.random.default_rng(ctx["seed"] + 7)
    raw, nums, accs = [], [], []
    for _ in range(count):
        mps = damped_mps(n, params, chi, g, rng)
        raw.append(mps.energy())
        num, acc = sz_verified(mps)
        nums.append(num)
        accs.append(acc)
    m_raw, se_raw = mean_se(np.array(raw))
    m_ver, _ = sz_ratio(nums, accs)
    acceptance = float(np.mean(accs))
    log.info(
        "raw bias %.4f +- %.4f | verified bias %.2e | acceptance %.3f",
        m_raw - clean,
        se_raw,
        m_ver - clean,
        acceptance,
    )
    r.append(check(m_raw - clean > 2 * se_raw, "the raw damped energy is biased"))
    r.append(
        check(
            abs(m_ver - clean) < 0.01 * (m_raw - clean),
            "verification removes more than 99% of it",
        )
    )
    r.append(
        check(
            0.7 < acceptance < 1.0, f"at the cost of {1 - acceptance:.0%} of the shots"
        )
    )


def check_long_coherent(r, ctx):
    """Check 8: coherent errors on the long chain (no sampling needed)."""
    n, chi = ctx["n"], ctx["chi"]
    params = ctx["params8"][3]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 8: n=%d coherent errors, deterministic ===", n)
    # No trajectories here: a coherent error is one deterministic MPS run.
    errs = np.random.default_rng(ctx["seed"] + 8).normal(0, 0.02, n)
    vals = [
        coherent_mps(n, params, chi, errs, "Z", f).energy() - clean for f in (1, 3, 5)
    ]
    s = slope(vals[0], vals[2], 5)
    log.info(
        "frozen detuning bias at folds 1,3,5: %s (exponent %.2f)", np.round(vals, 4), s
    )
    r.append(check(vals[0] > 0 and s > 1.4, f"the bias grows like fold^{s:.2f}"))
    gate = [
        coherent_mps(n, params, chi, None, "Z", f, 0.05).energy() - clean
        for f in (1, 3, 5)
    ]
    spread = (max(gate) - min(gate)) / gate[0]
    log.info("gate-angle bias at folds 1,3,5: %s", np.round(gate, 6))
    r.append(
        check(
            gate[0] > 1e-3 and spread < 0.01,
            "a gate-angle error is still invisible to folding",
        )
    )


def main(argv=None):
    """Run every Lab 33 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 33: damping and coherent errors")
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
    # Quick mode shortens the chain and samples fewer trajectories.
    ctx = {
        "seed": args.seed,
        "n": 12 if quick else 16,
        "chi": 32,
        "count": 80 if quick else args.trajectories,
        # Fewer trajectories in quick mode are compensated by a stronger rate.
        "gamma": 2 * args.gamma if quick else args.gamma,
        "g_small": 0.01,
        "small_count": 400 if quick else 1200,
        "draws": 1000 if quick else 3000,
    }
    # Angles are trained on small chains, exactly as in Labs 28-31.
    # Training is cheap at these sizes; the angles transfer to the long chain.
    ctx["params6"], ctx["params8"] = [], []
    for size, key in ((6, "params6"), (8, "params8")):
        e0 = float(
            eigsh(sparse_hamiltonian(size), k=1, which="SA", return_eigenvectors=False)[
                0
            ].real
        )
        trained = train_angles(
            size, e0, 4, np.random.default_rng(args.seed), extra_starts=0
        )
        ctx[key] = [x for _, x in trained]
    # r collects every check's boolean so the exit code reflects all of them.
    # Order matters: the machinery check comes first so later failures are meaningful.
    r = []
    check_machinery(r, ctx)
    check_exact_damping(r, ctx)
    check_verification_exact(r, ctx)
    check_coherent_exact(r, ctx)
    check_fold_scaling(r, ctx)
    check_gate_error(r, ctx)
    check_long_damping(r, ctx)
    check_long_coherent(r, ctx)
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
