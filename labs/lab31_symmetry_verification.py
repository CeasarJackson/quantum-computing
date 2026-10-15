#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab31_symmetry_verification.py

Purpose
-------
Lab 30's zero-noise extrapolation needs no knowledge of the circuit but costs
many shots. The Heisenberg Hamiltonian Variational Ansatz offers a cheaper
check: it CONSERVES the total spin projection S_z (every gate is built from
XX + YY + ZZ), and it starts in an S_z = 0 state. A Pauli X or Y error flips
one spin and moves the state out of S_z = 0; a Pauli Z error does not.
SYMMETRY VERIFICATION measures S_z at the end and keeps only the S_z = 0 shots.

Because the Hamiltonian commutes with S_z, the energy of the kept shots is
    E_verified = Tr(H P0 rho) / Tr(P0 rho),
where P0 projects onto S_z = 0 and Tr(P0 rho) is the acceptance rate.

What it can and cannot do depends on the noise channel, so the lab studies
three channels with the same total error rate p:
    depolarizing  (X, Y, Z equally likely)   -> 2/3 of errors are detectable
    dephasing     (Z only)                   -> none is detectable
    bit-flip type (X and Y only)             -> all are detectable
and then combines the check with zero-noise extrapolation.

The lab verifies:
  1. The MPS projector machinery (rotation average) against a state-vector
     projection; a clean HVA state is always accepted; a single X error is
     always rejected and a single Z error always accepted.
  2. Exact channel study on a density matrix (n=6): verification is useless
     against dephasing, removes about two thirds of the depolarizing bias and
     nearly all of the bit-flip bias.
  3. Sampled trajectories (n=6) reproduce the exact post-selected energy.
  4. Complementarity: extrapolation fixes dephasing, verification does not;
     doing both beats either alone for depolarizing noise.
  5. On 16 sites with trajectories, verification cuts the bias with only a
     small loss of shots (the acceptance rate) and no extra circuits.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab31_symmetry_verification.py
    python3 labs/lab31_symmetry_verification.py --trajectories 2000 --rate 1e-3
    python3 labs/lab31_symmetry_verification.py --quick

Log files
---------
logs/lab31/lab31_<YYYYmmdd_HHMMSS>.log

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
    apply_pauli,
    dm_energy,
    dm_sz_verified,
    mean_se,
    noisy_mps,
    richardson_weights,
    sz_ratio,
    sz_verified,
)

log = get_logger("lab31")

# The three channels: relative probabilities of an X, Y or Z error.
CHANNELS = {
    "depolarizing": None,
    "dephasing": (0, 0, 1),
    "bit-flip type": (1, 1, 0),
}


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_machinery(r, ctx):
    """Check 1: the S_z projector on an MPS."""
    log.info("=== Check 1: S_z projector on the MPS (n=6) ===")
    n = 6
    params = ctx["angles"][2]
    mps = run_hva(n, params, 64)
    # Build a state with several errors, then compare with an explicit
    # projection of its state vector onto the S_z = 0 amplitudes.
    err = mps.copy()
    # An even number of flips, so part of the state stays in S_z = 0.
    for site, name in ((1, "X"), (3, "Y"), (4, "Z")):
        apply_pauli(err, site, name)
    err.flips = 2
    num, acc = sz_verified(err)
    vec = err.to_vector()
    ones = np.array([bin(i).count("1") for i in range(2**n)])
    proj = np.where(ones == n // 2, vec, 0)
    h = sparse_hamiltonian(n)
    want_acc = float(np.vdot(vec, proj).real)
    want_num = float(np.vdot(vec, h @ proj).real)
    r.append(check(0.01 < acc < 0.99, f"the test state is partly accepted ({acc:.3f})"))
    log.info(
        "acceptance %.6f (state vector %.6f); numerator %.6f (%.6f)",
        acc,
        want_acc,
        num,
        want_num,
    )
    # Agreement to 1e-10 means the rotation average is an exact projector.
    r.append(
        check(
            abs(acc - want_acc) < 1e-10 and abs(num - want_num) < 1e-10,
            "rotation-average projector equals the state-vector projection",
        )
    )
    # Clean state on the long chain: HVA conserves S_z.
    # The Hamiltonian commutes with S_z, so an error-free circuit never leaves it.
    big = run_hva(ctx["n"], ctx["angles"][3], ctx["chi"])
    big.flips = 0
    acc_clean = sz_verified(big)[1]
    r.append(
        check(abs(acc_clean - 1) < 1e-9, f"noiseless n={ctx['n']} state is S_z = 0")
    )
    # Same site, two different errors: X flips the spin, Z only changes a phase.
    x_hit, z_hit = mps.copy(), mps.copy()
    apply_pauli(x_hit, 2, "X")
    apply_pauli(z_hit, 2, "Z")
    x_hit.flips = z_hit.flips = 1
    acc_x, acc_z = sz_verified(x_hit)[1], sz_verified(z_hit)[1]
    # Zero and one are exact outcomes here, so the tolerances are tiny.
    r.append(
        check(
            abs(acc_x) < 1e-10 and abs(acc_z - 1) < 1e-10,
            f"one X error is always rejected ({acc_x:.0e}), one Z always "
            f"accepted ({acc_z:.6f})",
        )
    )


def check_channels(r, ctx):
    """Check 2: exact results for three noise channels (n=6)."""
    n, p = 6, ctx["p_exact"]
    params = ctx["angles"][2]
    # Noiseless energy at chi=64 is exact for 6 sites; biases are measured from it.
    clean = run_hva(n, params, 64).energy()
    log.info("=== Check 2: exact channel study (n=6, p=%g) ===", p)
    # For each channel: raw bias, verified bias, and the acceptance rate.
    rows = {}
    for name, weights in CHANNELS.items():
        raw = dm_energy(n, params, p, weights=weights) - clean
        ver, acc = dm_sz_verified(n, params, p, weights=weights)
        rows[name] = (raw, ver - clean, acc)
        log.info(
            "%-14s raw bias %.4f | verified bias %.4f (removes %3.0f%%) | "
            "acceptance %.3f",
            name,
            raw,
            ver - clean,
            100 * (1 - (ver - clean) / raw),
            acc,
        )
    ctx["rows"] = rows
    raw, ver, acc = rows["dephasing"]
    # Z errors only change phases, so the state stays inside S_z = 0 exactly.
    r.append(
        check(
            abs(raw - ver) < 1e-9 and abs(acc - 1) < 1e-9,
            "dephasing: verification changes nothing (Z errors keep S_z = 0)",
        )
    )
    raw, ver, acc = rows["bit-flip type"]
    r.append(check(ver < 0.1 * raw, "bit-flip noise: over 90% of the bias removed"))
    raw, ver, acc = rows["depolarizing"]
    frac = 1 - ver / raw
    # X and Y are two of the three depolarizing outcomes: about 2/3 detectable.
    r.append(
        check(
            0.5 < frac < 0.8,
            f"depolarizing: about two thirds removed ({frac:.0%}), "
            f"the X and Y share of the errors",
        )
    )


def check_trajectories_small(r, ctx):
    """Check 3: sampled trajectories against the exact post-selected value."""
    n, p, count = 6, 0.02, ctx["small_count"]
    params = ctx["angles"][2]
    log.info("=== Check 3: trajectories versus exact (n=6, p=%g) ===", p)
    exact, exact_acc = dm_sz_verified(n, params, p)
    # Own seed offset so this check's random stream is independent of the others.
    rng = np.random.default_rng(ctx["seed"] + 7)
    nums, accs = [], []
    # Larger p here (0.02) so that many trajectories actually contain errors.
    for _ in range(count):
        mps = noisy_mps(n, params, 64, p, rng)
        num, acc = sz_verified(mps)
        nums.append(num)
        accs.append(acc)
    mean, se = sz_ratio(nums, accs)
    log.info(
        "exact %.5f (acceptance %.3f) | trajectories %.5f +- %.5f (acceptance %.3f)",
        exact,
        exact_acc,
        mean,
        se,
        float(np.mean(accs)),
    )
    # The ratio estimator needs the delta-method standard error, not a plain mean.
    r.append(
        check(
            abs(mean - exact) < 4 * se,
            f"post-selected energy within 4 standard errors "
            f"({abs(mean - exact) / se:.1f})",
        )
    )
    r.append(
        check(
            abs(np.mean(accs) - exact_acc) < 4 * np.std(accs, ddof=1) / count**0.5,
            "acceptance rate agrees as well",
        )
    )


def check_combination(r, ctx):
    """Check 4: extrapolation and verification are complementary (exact)."""
    n, p = 6, ctx["p_exact"]
    params = ctx["angles"][2]
    clean = run_hva(n, params, 64).energy()
    # Weights for lambda = 1, 3, 5, the same extrapolation as Lab 30.
    w = richardson_weights([1, 3, 5])
    log.info("=== Check 4: verification and extrapolation together (n=6) ===")
    out = {}
    for name in ("dephasing", "depolarizing"):
        weights = CHANNELS[name]
        raw = [dm_energy(n, params, p, f, weights) - clean for f in (1, 3, 5)]
        ver = [dm_sz_verified(n, params, p, f, weights)[0] - clean for f in (1, 3, 5)]
        out[name] = (raw[0], ver[0], float(w @ raw), float(w @ ver))
        log.info(
            "%-13s raw %.4f | verified %.4f | extrapolated %.4f | both %.4f",
            name,
            *out[name],
        )
    raw, ver, zne, both = out["dephasing"]
    # Verification cannot touch dephasing, but extrapolation still removes it.
    r.append(
        check(
            abs(zne) < abs(raw) / 30 and abs(ver - raw) < 1e-9,
            "dephasing: extrapolation works where verification does nothing",
        )
    )
    raw, ver, zne, both = out["depolarizing"]
    r.append(
        check(
            abs(both) < abs(zne) and abs(both) < abs(ver) / 30,
            f"depolarizing: both together ({abs(both):.4f}) beat extrapolation "
            f"alone ({abs(zne):.4f}) and verification alone ({abs(ver):.4f})",
        )
    )


def check_long_chain(r, ctx):
    """Check 5: trajectories on the long chain, raw versus verified."""
    n, chi, count, p = ctx["n"], ctx["chi"], ctx["count"], ctx["rate"]
    params = ctx["angles"][3]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 5: n=%d, 4 layers, p=%g, N=%d ===", n, p, count)
    rng = np.random.default_rng(ctx["seed"] + 8)
    # Parallel lists: raw energy, numerator <H P0> and acceptance <P0> per trajectory.
    raw, nums, accs = [], [], []
    # error-free trajectories are skipped inside noisy_mps and handled below.
    for _ in range(count):
        mps = noisy_mps(n, params, chi, p, rng, skip_clean=True)
        if mps is None:
            # No error drawn: the ideal state, always accepted.
            raw.append(clean)
            nums.append(clean)
            accs.append(1.0)
            continue
        raw.append(mps.energy())
        num, acc = sz_verified(mps)
        nums.append(num)
        accs.append(acc)
    m_raw, se_raw = mean_se(np.array(raw))
    m_ver, se_ver = sz_ratio(nums, accs)
    acceptance = float(np.mean(accs))
    log.info(
        "raw bias %.4f +- %.4f | verified bias %.4f +- %.4f | acceptance %.3f",
        m_raw - clean,
        se_raw,
        m_ver - clean,
        se_ver,
        acceptance,
    )
    r.append(check(m_raw - clean > 3 * se_raw, "the raw noisy energy is biased"))
    # A modest bias cut with almost no discarded shots: the cheap-mitigation case.
    r.append(
        check(
            m_ver - clean < 0.7 * (m_raw - clean),
            f"verification cuts the bias ({m_raw - clean:.3f} -> {m_ver - clean:.3f})",
        )
    )
    r.append(
        check(
            0.8 < acceptance < 1.0,
            f"only {1 - acceptance:.0%} of the shots are discarded",
        )
    )
    r.append(check(se_ver < 1.5 * se_raw, "and the statistical error barely grows"))


def main(argv=None):
    """Run every Lab 31 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 31: symmetry verification")
    ap.add_argument("--trajectories", type=int, default=1000, help="long chain")
    ap.add_argument("--rate", type=float, default=5e-4, help="error rate p")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.trajectories < 50:
        ap.error(f"--trajectories must be at least 50. Got {args.trajectories}.")
    if not 1e-5 <= args.rate <= 5e-3:
        ap.error(f"--rate must be between 1e-5 and 5e-3. Got {args.rate}.")
    quick = args.quick
    # Quick mode shortens the chain and uses fewer, noisier trajectories.
    ctx = {
        "seed": args.seed,
        "n": 12 if quick else 16,
        "chi": 32,
        "count": 150 if quick else args.trajectories,
        "rate": 1e-3 if quick else args.rate,
        "p_exact": 0.003,
        "small_count": 400 if quick else 1500,
    }
    n_train = 8
    e0 = float(
        eigsh(sparse_hamiltonian(n_train), k=1, which="SA", return_eigenvectors=False)[
            0
        ].real
    )
    trained = train_angles(
        n_train, e0, 4, np.random.default_rng(args.seed), extra_starts=0
    )
    ctx["angles"] = [x for _, x in trained]
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_machinery(r, ctx)
    check_channels(r, ctx)
    check_trajectories_small(r, ctx)
    check_combination(r, ctx)
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
