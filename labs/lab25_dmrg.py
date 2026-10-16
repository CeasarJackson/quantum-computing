#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab25_dmrg.py

Purpose
-------
Lab 23 found exact ground energies by Lanczos on the full 2^n space, which
stopped at n = 18. Lab 24 showed that low-entanglement states fit in a matrix
product state. Lab 25 closes the loop: the DENSITY-MATRIX RENORMALISATION
GROUP (DMRG) finds the ground state of the Heisenberg chain INSIDE the MPS
manifold, so the exact reference now reaches n = 64 and beyond.

Everything is written from scratch in labs/mpslib.py:
  * the Hamiltonian as a bond-dimension-5 matrix product operator (MPO),
  * left and right "environments" that summarise the rest of the chain,
  * a two-site optimisation: build the effective Hamiltonian of two
    neighbouring sites, find its lowest eigenvector with Lanczos, split it
    with an SVD that keeps the chi largest singular values,
  * sweeps left and right until the energy stops changing.

The lab verifies:
  1. The MPO equals the sparse Hamiltonian of Lab 23.
  2. DMRG reproduces the exact ground energy and state for n = 8..16.
  3. The energy only falls from sweep to sweep (a variational method).
  4. Energy error and discarded weight fall as the bond dimension chi grows.
  5. A fit of E0(n) over n = 16..64 gives the Bethe-ansatz energy per site,
     1 - 4 ln 2, to better than 1e-4 (3e-4 in --quick mode).
  6. Half-chain entanglement grows like (1/6) ln n (a critical chain, c = 1).
  7. Bond energies alternate near the open ends and settle toward the bulk.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab25_dmrg.py
    python3 labs/lab25_dmrg.py --chi 64 --max-sites 96
    python3 labs/lab25_dmrg.py --quick

Log files
---------
logs/lab25/lab25_<YYYYmmdd_HHMMSS>.log

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
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import BOND, DMRG, E_INF, heisenberg_mpo, mpo_to_dense, overlap

log = get_logger("lab25")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def sparse_hamiltonian(n):
    """H = sum over bonds of XX+YY+ZZ as a scipy sparse matrix (Lab 23)."""
    h = sp.csr_matrix((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        left = sp.identity(2 ** (n - q - 2), format="csr")
        right = sp.identity(2**q, format="csr")
        h = h + sp.kron(sp.kron(left, sp.csr_matrix(BOND)), right, format="csr")
    return h


def exact_ground(n):
    """Lowest eigenvalue and eigenvector of the n-site chain (Lanczos)."""
    vals, vecs = eigsh(sparse_hamiltonian(n), k=1, which="SA")
    return float(vals[0].real), vecs[:, 0]


def solve(n, chi):
    """Run DMRG to convergence; return (solver, energy history, seconds)."""
    t0 = time.perf_counter()
    solver = DMRG(heisenberg_mpo(n), chi)
    history = solver.run(max_sweeps=30, tol=1e-9)
    return solver, history, time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_mpo(r):
    """Check 1: the MPO is the Heisenberg Hamiltonian."""
    log.info("=== Check 1: the MPO versus the sparse Hamiltonian ===")
    # The MPO is only trustworthy if contracting it back to a matrix gives the
    # same operator the earlier labs built from Kronecker products.
    for n in (2, 4, 6, 8):
        dense = mpo_to_dense(heisenberg_mpo(n))
        ref = sparse_hamiltonian(n).toarray()
        err = np.abs(dense - ref).max()
        r.append(
            check(err < 1e-12, f"n={n}: MPO equals H (max difference " f"{err:.1e})")
        )


def check_exact(r, sizes):
    """Check 2: DMRG against exact diagonalisation."""
    log.info("=== Check 2: DMRG versus exact diagonalisation ===")
    for n in sizes:
        e_ex, v_ex = exact_ground(n)
        # chi = 2^(n/2) is the largest Schmidt rank on any cut, so nothing is
        # truncated: DMRG must agree with Lanczos to rounding error.
        solver, hist, secs = solve(n, 2 ** (n // 2))
        # |<exact|DMRG>| = 1 up to a global phase means the same state.
        fid = abs(np.vdot(v_ex, solver.mps.copy().to_vector()))
        log.info(
            "n=%2d: exact %.10f | DMRG %.10f | %d sweeps, %.1f s | " "overlap %.10f",
            n,
            e_ex,
            hist[-1],
            len(hist),
            secs,
            fid,
        )
        r.append(
            check(
                abs(hist[-1] - e_ex) < 1e-8 and fid > 1 - 1e-8,
                f"n={n}: DMRG energy and state equal the exact ones",
            )
        )


def check_variational(r, n, chi):
    """Check 3: energy never rises from one sweep to the next."""
    log.info("=== Check 3: convergence of the sweeps (n=%d, chi=%d) ===", n, chi)
    solver = DMRG(heisenberg_mpo(n), chi)
    history = []
    # One call to sweep() is a full left-to-right plus right-to-left pass.
    for _ in range(5):
        history.append(solver.sweep())
    log.info("energy after each sweep: %s", " ".join(f"{e:.6f}" for e in history))
    r.append(
        check(
            all(b <= a + 1e-8 for a, b in zip(history, history[1:])),
            "the energy never rises from sweep to sweep",
        )
    )
    r.append(
        check(
            abs(history[-1] - history[-2]) < 1e-5,
            f"converged: last change {abs(history[-1] - history[-2]):.1e}",
        )
    )
    # Recompute the energy independently, from the bond energies of the final
    # state, to prove the solver's number belongs to the state it returns.
    mps = solver.mps.copy()
    r.append(
        check(
            abs(mps.energy() - history[-1]) < 1e-7
            and abs(overlap(mps, mps) - 1) < 1e-9,
            "the final MPS is normalised and its energy equals the " "sweep energy",
        )
    )


def check_chi(r, n, chis, ref_chi, tol):
    """Check 4: error against bond dimension, relative to a larger chi."""
    log.info("=== Check 4: error versus bond dimension (n=%d) ===", n)
    # The reference is a run with a larger chi than any tested value.
    ref = solve(n, ref_chi)[1][-1]
    errs, lost = [], []
    for c in chis:
        solver, hist, _ = solve(n, c)
        errs.append(hist[-1] - ref)
        lost.append(solver.discarded)
        log.info(
            "chi=%2d: E - E(chi=%d) = %.2e, discarded weight %.1e",
            c,
            ref_chi,
            errs[-1],
            lost[-1],
        )
    r.append(
        check(
            all(e > -1e-9 for e in errs),
            "variational: no smaller chi beats the largest chi",
        )
    )
    r.append(
        check(
            all(b < a for a, b in zip(errs, errs[1:])),
            "the energy error falls at every step in chi",
        )
    )
    r.append(
        check(
            all(b < a for a, b in zip(lost, lost[1:])),
            "the discarded weight falls at every step in chi",
        )
    )
    r.append(
        check(
            errs[-1] < tol,
            f"chi={chis[-1]} is within {tol:.0e} of " f"chi={ref_chi} ({errs[-1]:.1e})",
        )
    )


def check_bethe(r, sizes, chi, tol):
    """Checks 5-7: bulk energy, entanglement and bond energies from DMRG."""
    log.info("=== Check 5: bulk energy from DMRG (n up to %d) ===", sizes[-1])
    energies, entropies, solvers = {}, {}, {}
    # Solve every chain once and keep the solver: later checks reuse the
    # states instead of paying for DMRG again.
    for n in sizes:
        solver, hist, secs = solve(n, chi)
        energies[n], entropies[n], solvers[n] = hist[-1], solver.mid_entropy, solver
        log.info(
            "n=%3d: E0 = %.8f, E0/n = %.6f, S_half = %.4f, discarded " "%.1e (%.1f s)",
            n,
            hist[-1],
            hist[-1] / n,
            entropies[n],
            solver.discarded,
            secs,
        )
    # Open chain: E0(n) = e n + b + c/n, with e the bulk energy per site, b a
    # surface energy from the two ends, and c a Casimir-type finite-size term.
    mat = np.array([[n, 1.0, 1.0 / n] for n in sizes])
    (e_bulk, surf, cas), *_ = np.linalg.lstsq(
        mat, [energies[n] for n in sizes], rcond=None
    )
    log.info(
        "fit E0(n) = %.6f n %+.5f %+.4f/n | Bethe ansatz %.6f n",
        e_bulk,
        surf,
        cas,
        E_INF,
    )
    r.append(
        check(
            abs(e_bulk - E_INF) < tol,
            f"fitted energy per site {e_bulk:.6f} matches 1 - 4 ln 2 = "
            f"{E_INF:.6f} within {tol:.0e}",
        )
    )
    r.append(
        check(cas < 0, "the 1/n (Casimir) term is negative, as for a " "gapless chain")
    )
    log.info("=== Check 6: entanglement grows like (c/6) ln n ===")
    # Calabrese-Cardy: S = (c/6) ln n + const for an open critical chain with
    # central charge c. The Heisenberg chain has c = 1 (plus log corrections).
    slope = np.polyfit(np.log(sizes), [entropies[n] for n in sizes], 1)[0]
    log.info("S_half = %.4f ln n + const (c = 1 predicts 1/6 = 0.1667)", slope)
    r.append(
        check(
            all(
                b > a
                for a, b in zip(
                    [entropies[n] for n in sizes], [entropies[n] for n in sizes][1:]
                )
            ),
            "half-chain entropy grows with n",
        )
    )
    r.append(
        check(0.12 < slope < 0.25, f"entropy slope {slope:.3f} is near c/6 for c = 1")
    )
    log.info("=== Check 7: bond energies near the open end ===")
    n = 32
    # Energy of every individual bond: open ends break translation symmetry,
    # so strong and weak bonds alternate, with the alternation fading inward.
    be = solvers[n].mps.copy().bond_energies()
    log.info("n=%d, first bonds: %s", n, " ".join(f"{e:.3f}" for e in be[:8]))
    mid = float(np.mean(be[n // 2 - 6 : n // 2 + 6]))
    log.info("mean of the 12 central bonds: %.5f (bulk %.5f)", mid, E_INF)
    r.append(
        check(
            all(be[2 * k] < be[2 * k + 1] for k in range(4)),
            "even bonds are stronger than odd ones at the edge "
            "(open-end dimerisation)",
        )
    )
    r.append(check(be[0] < be[n // 2], "the end bond is stronger than the middle bond"))
    r.append(
        check(
            abs(mid - E_INF) < 5e-3,
            "central bonds average to the bulk energy within 5e-3",
        )
    )
    r.append(
        check(
            abs(sum(be) - energies[n]) < 1e-7,
            "the bond energies sum to the DMRG energy",
        )
    )
    return energies


def main(argv=None):
    """Run every Lab 25 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 25: DMRG for the Heisenberg " "chain")
    ap.add_argument("--chi", type=int, default=32, help="bond dimension 16..128")
    ap.add_argument("--max-sites", type=int, default=64, help="longest chain, 32..128")
    # Bounds protect against settings that would run for hours or give no
    # information (chi below 16 truncates too hard for the checks).
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if not 16 <= args.chi <= 128:
        ap.error(f"--chi must be 16..128. Got {args.chi}.")
    if not 32 <= args.max_sites <= 128:
        ap.error(f"--max-sites must be 32..128. Got {args.max_sites}.")
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_mpo(r)
    chi = args.chi
    if args.quick:
        check_exact(r, [8, 12])
        check_variational(r, 24, min(chi, 24))
        check_chi(r, 24, (4, 8, 16), 24, 1e-3)
        check_bethe(r, [12, 16, 24, 32], min(chi, 24), 3e-4)
    else:
        check_exact(r, [8, 12, 16])
        check_variational(r, 32, chi)
        check_chi(r, 32, (4, 8, 16, 32), 48, 1e-4)
        sizes = [n for n in (16, 24, 32, 48, 64, 96, 128) if n <= args.max_sites]
        check_bethe(r, sizes, chi, 1e-4)
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
