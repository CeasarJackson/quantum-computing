#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab28_deeper_hva.py

Purpose
-------
Lab 26 showed that a 6-layer Hamiltonian Variational Ansatz (HVA) is off by
about 0.012 energy units per site in the bulk of the Heisenberg chain, and that
its entanglement saturates. Here the circuit is made deeper (up to 12 layers)
and the improvement is measured against DMRG, two ways, for comparison:

  A. TRANSFER (cheap). Train the angles with the exact state-vector adjoint
     method on a short chain (default 12 qubits), then run the same angles on
     a long chain with the MPS simulator. This works because the circuit is
     local: a depth-p circuit only feels a window of about 4p sites, so the
     bulk of a long chain sees the same physics as a short one.
  B. DIRECT (expensive, --full). Train the angles on a longer chain (n=20)
     with the MPS adjoint gradient of Lab 27, starting from the transferred
     angles, and see how much the extra cost buys.

The lab verifies:
  1. Training on the short chain: the error falls with depth (warm starts make
     it non-increasing) and 12 layers beat 6 layers by a clear factor.
  2. Transfer: the same angles on a longer chain give an energy above the DMRG
     ground energy (variational) that also falls with depth.
  3. Bulk error per site (difference of two long chains, so the surface term
     cancels) is smaller than the 6-layer value of Lab 26.
  4. Entanglement: the circuit's half-chain entropy keeps growing with depth
     instead of saturating, and stays below the true ground state's.
  5. Truncation: the bond dimension used is large enough (energy unchanged
     when chi grows).
  6. (--full) Direct MPS training: never worse than its transferred start;
     the gain and the cost are reported next to the transfer method.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab28_deeper_hva.py
    python3 labs/lab28_deeper_hva.py --layers 10 --train-n 12
    python3 labs/lab28_deeper_hva.py --quick
    python3 labs/lab28_deeper_hva.py --full      # adds the slow direct method

Log files
---------
logs/lab28/lab28_<YYYYmmdd_HHMMSS>.log

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
    heisenberg_mpo,
    mps_energy_and_adjoint_grad,
    run_hva,
    sparse_hamiltonian,
    train_angles,
)

log = get_logger("lab28")

# Bulk error per site of the 6-layer circuit, measured in Lab 26.
LAB26_BULK_ERROR = 0.012


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def dmrg_reference(n, chi, cache):
    """Ground energy and half-chain entropy of an n-site chain, cached by n."""
    if n not in cache:
        solver = DMRG(heisenberg_mpo(n), chi)
        energy = solver.run(max_sweeps=30, tol=1e-9)[-1]
        cache[n] = (energy, solver.mid_entropy)
    return cache[n]


def bulk_error(params, n1, n2, chi, ref):
    """Energy error per site in the bulk, from two chain lengths.

    The surface term is the same for both lengths, so it cancels in
    (E(n2) - E(n1)) / (n2 - n1); subtracting the same quantity for DMRG leaves
    the error the circuit makes per extra site.
    """
    e1 = run_hva(n1, params, chi).energy()
    e2 = run_hva(n2, params, chi).energy()
    return ((e2 - e1) - (ref[n2][0] - ref[n1][0])) / (n2 - n1)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_training(r, ctx):
    """Check 1: state-vector training on the short chain, depth by depth."""
    n, layers, ref = ctx["train_n"], ctx["layers"], ctx["ref_depth"] - 1
    log.info("=== Check 1: training on n=%d, 1..%d layers ===", n, layers)
    e0 = float(
        eigsh(sparse_hamiltonian(n), k=1, which="SA", return_eigenvectors=False)[0].real
    )
    t0 = time.perf_counter()
    # No random extra starts: the warm start from the previous depth already
    # contains the shallower solution, and extra starts dominate the runtime.
    # Warm starts make the error non-increasing: depth p contains depth p-1.
    trained = train_angles(n, e0, layers, ctx["rng"], extra_starts=0)
    ctx["train_time"] = time.perf_counter() - t0
    errs = [e for e, _ in trained]
    ctx["angles"] = [x for _, x in trained]
    log.info("error by depth: %s", " ".join(f"{e:.1e}" for e in errs))
    log.info("training time %.0f s", ctx["train_time"])
    # A tiny tolerance allows for optimiser noise, nothing more.
    r.append(
        check(
            all(b <= a + 1e-6 for a, b in zip(errs, errs[1:])),
            "training error never increases with depth (warm start)",
        )
    )
    # The reference depth is 6 normally; quick mode uses 4 because it stops at 8.
    r.append(
        check(
            errs[-1] < errs[ref] / 2,
            f"{layers} layers beat {ref + 1} layers by more than 2x on n={n}",
        )
    )


def check_transfer(r, ctx):
    """Check 2: run the short-chain angles on a longer chain."""
    n, chi = ctx["eval_n"], ctx["chi"]
    e_ref = dmrg_reference(n, 32, ctx["ref"])[0]
    log.info("=== Check 2: transfer to n=%d (DMRG E0 = %.5f) ===", n, e_ref)
    errs = []
    # Same angles, longer chain: the MPS handles lengths the vector cannot.
    for p, params in enumerate(ctx["angles"], 1):
        errs.append(run_hva(n, params, chi).energy() - e_ref)
    ctx["transfer_errors"] = errs
    log.info("error by depth: %s", " ".join(f"{e:.4f}" for e in errs))
    log.info("error per site: %s", " ".join(f"{e / n:.4f}" for e in errs))
    # Variational principle: no trial state can lie below the true ground energy.
    r.append(
        check(
            min(errs) > -1e-6,
            "energy never goes below the DMRG ground energy (variational)",
        )
    )
    # The 2e-3 slack covers plateaus where a new layer starts at the identity.
    r.append(
        check(
            all(b <= a + 2e-3 for a, b in zip(errs, errs[1:])),
            "transferred error falls (within 2e-3) as depth grows",
        )
    )
    shallow = ctx["ref_depth"] - 1
    r.append(
        check(
            errs[-1] < errs[shallow] / 2,
            f"{len(errs)} layers cut the error of {shallow + 1} layers "
            f"by more than 2x ({errs[shallow]:.3f} -> {errs[-1]:.3f})",
        )
    )


def check_bulk(r, ctx):
    """Check 3: bulk error per site by depth, against Lab 26."""
    n1, n2, chi = ctx["bulk_n"][0], ctx["bulk_n"][1], ctx["chi"]
    dmrg_reference(n1, 32, ctx["ref"])
    dmrg_reference(n2, 32, ctx["ref"])
    log.info("=== Check 3: bulk error per site (n=%d and n=%d) ===", n1, n2)
    # Two long chains share the surface term, so the difference isolates the bulk.
    errs = [bulk_error(x, n1, n2, chi, ctx["ref"]) for x in ctx["angles"]]
    ctx["bulk_errors"] = errs
    log.info("bulk error by depth: %s", " ".join(f"{e:.1e}" for e in errs))
    # Bulk error should fall with depth; a small slack allows for flat steps.
    r.append(
        check(
            all(b <= a + 1e-3 for a, b in zip(errs, errs[1:])),
            "bulk error per site falls (within 1e-3) as depth grows",
        )
    )
    # Limit set from Lab 26: roughly 0.6 of the 6-layer bulk error.
    r.append(
        check(
            errs[-1] < ctx["bulk_limit"],
            f"{len(errs)} layers: bulk error {errs[-1]:.1e} is below "
            f"{ctx['bulk_limit']:.1e} (6 layers in Lab 26: {LAB26_BULK_ERROR})",
        )
    )


def check_entanglement(r, ctx):
    """Check 4: half-chain entropy of the circuit versus depth."""
    n, chi = ctx["eval_n"], ctx["chi"]
    s_true = ctx["ref"][n][1]
    ent = []
    # Entropy across the middle cut; DMRG gives the true value to compare with.
    for params in ctx["angles"]:
        ent.append(run_hva(n, params, chi).entropy(n // 2 - 1))
    log.info("=== Check 4: half-chain entropy at n=%d ===", n)
    log.info(
        "circuit entropy by depth: %s (DMRG %.3f)",
        " ".join(f"{s:.3f}" for s in ent),
        s_true,
    )
    # Lab 26 saw the entropy saturate near 0.5; deeper circuits should pass it.
    r.append(
        check(
            ent[-1] > ent[ctx["ref_depth"] - 1] + 0.05,
            f"circuit entanglement keeps growing past {ctx['ref_depth']} layers",
        )
    )
    r.append(
        check(
            ent[-1] < s_true + 0.05,
            "and stays at or below the true ground state's entropy",
        )
    )


def check_truncation(r, ctx):
    """Check 5: the bond dimension is not what limits the result."""
    n = ctx["eval_n"]
    params = ctx["angles"][-1]
    # Same circuit at two bond dimensions: if they agree, chi is not the limit.
    low = run_hva(n, params, ctx["chi"])
    high = run_hva(n, params, 2 * ctx["chi"])
    gap = abs(low.energy() - high.energy())
    log.info("=== Check 5: truncation (n=%d, deepest circuit) ===", n)
    log.info(
        "chi=%d vs chi=%d: energy difference %.1e, discarded weight %.1e",
        ctx["chi"],
        2 * ctx["chi"],
        gap,
        low.discarded,
    )
    r.append(
        check(
            gap < 1e-3,
            f"doubling chi changes the energy by only {gap:.1e}",
        )
    )


def check_direct(r, ctx):
    """Check 6 (--full): train on a longer chain with the MPS adjoint."""
    n, chi = ctx["direct_n"], ctx["chi"]
    e_ref = dmrg_reference(n, 32, ctx["ref"])[0]
    log.info("=== Check 6: direct MPS training on n=%d (chi=%d) ===", n, chi)
    rows = []
    # Start from the transferred angles so L-BFGS only has to polish them.
    for p in ctx["direct_depths"]:
        start = ctx["angles"][p - 1]
        t0 = time.perf_counter()
        res = minimize(
            lambda x: mps_energy_and_adjoint_grad(x, n, chi, 2 * chi),
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": ctx["direct_iters"]},
        )
        cost = time.perf_counter() - t0
        e_start = run_hva(n, start, chi).energy() - e_ref
        e_end = res.fun - e_ref
        rows.append((p, e_start, e_end, cost))
        log.info(
            "depth %2d: transfer error %.4f -> direct %.4f (gain %.4f, %.0f s, "
            "%d gradients)",
            p,
            e_start,
            e_end,
            e_start - e_end,
            cost,
            res.nfev,
        )
    # L-BFGS only accepts steps that lower the energy, so this must hold.
    r.append(
        check(
            all(end <= start + 1e-6 for _, start, end, _ in rows),
            "direct training is never worse than its transferred start",
        )
    )
    r.append(
        check(
            all(end > -1e-6 for _, _, end, _ in rows),
            "and stays above the DMRG ground energy",
        )
    )
    # Gain is how much energy the extra cost bought over plain transfer.
    gain = sum(s - e for _, s, e, _ in rows)
    total = sum(c for *_, c in rows)
    log.info(
        "summary: total gain %.4f energy units for %.0f s of direct training; "
        "the transfer method cost %.0f s for all depths",
        gain,
        total,
        ctx["train_time"],
    )


def main(argv=None):
    """Run every Lab 28 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 28: deeper HVA judged by DMRG")
    ap.add_argument("--layers", type=int, default=12, help="maximum depth 6..12")
    ap.add_argument("--train-n", type=int, default=12, help="short training chain")
    ap.add_argument("--chi", type=int, default=48, help="MPS bond dimension")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    ap.add_argument("--full", action="store_true", help="add direct MPS training")
    args = ap.parse_args(argv)
    if not 6 <= args.layers <= 12:
        ap.error(f"--layers must be 6..12. Got {args.layers}.")
    if not 8 <= args.train_n <= 14:
        ap.error(f"--train-n must be 8..14. Got {args.train_n}.")
    quick = args.quick
    # Quick mode trains on a shorter chain, to fewer layers, and evaluates on
    # shorter chains so the whole lab takes well under a minute.
    ctx = {
        "rng": np.random.default_rng(args.seed),
        "layers": 8 if quick else args.layers,
        "train_n": 10 if quick else args.train_n,
        "eval_n": 16 if quick else 24,
        "bulk_n": (16, 24) if quick else (24, 32),
        "chi": 32 if quick else args.chi,
        "ref": {},
        # Depth that the deepest circuit must beat, and the bulk-error ceiling.
        "ref_depth": 4 if quick else 6,
        "bulk_limit": 0.011 if quick else 0.0075,
        "direct_n": 16 if quick else 20,
        "direct_depths": [8] if quick else [8, 10, 12],
        "direct_iters": 10 if quick else 15,
    }
    ctx["direct_depths"] = [p for p in ctx["direct_depths"] if p <= ctx["layers"]]
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_training(r, ctx)
    check_transfer(r, ctx)
    check_bulk(r, ctx)
    check_entanglement(r, ctx)
    check_truncation(r, ctx)
    if args.full:
        check_direct(r, ctx)
    else:
        log.info("Check 6 (direct MPS training) skipped; add --full to run it")
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
