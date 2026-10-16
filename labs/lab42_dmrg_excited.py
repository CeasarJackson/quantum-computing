#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab42_dmrg_excited.py

Purpose
-------
Lab 25 found the GROUND state of the Heisenberg chain with two-site DMRG. This
lab uses the symmetries of the Hamiltonian to reach more of the spectrum.

  * S_z SECTORS. H conserves total S_z, so the spectrum splits into sectors.
    A plain two-site DMRG does not respect that: the two centre sites span all
    four spin states, so the optimiser can slide to another sector. Giving
    every bond index a charge (the S_z to its left), allowing only the
    two-site entries whose charges add up to the target, and doing the SVD
    block by block fixes the sector exactly. The ground state of the S_z = m
    sector has total spin S = m, so a chain of 10 spins gives the whole ladder
    of S = 0, 1, 2, 3 ground states from four cheap runs.
  * EXCITED STATES. Inside one sector the next level is the ground state of
    H + w |0><0|, where |0> is the state already found and w is larger than the
    gap. Repeating this gives levels 0, 1, 2, ... Each is checked against exact
    diagonalization: energy, <S^2> (an integer S(S + 1)), and the reflection
    parity (does the mirror image of the chain give the state back or minus it?).
  * SU(2) MULTIPLETS. A spin-S level appears in every sector S_z = -S..S with
    the same energy. The lab finds the S_z = 0 and S_z = 1 ladders SEPARATELY
    and shows that the S >= 1 levels of the first are exactly the second.
  * GAP SCALING. The triplet gap of the open chain falls like 1/n; n times the
    gap approaches the conformal value 2 pi^2 (in these sigma.sigma units) from
    below, slowly because of logarithmic corrections.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab42_dmrg_excited.py
    python3 labs/lab42_dmrg_excited.py --seed 7
    python3 labs/lab42_dmrg_excited.py --quick

Log files
---------
logs/lab42/lab42_<YYYYmmdd_HHMMSS>.log

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
from dmrgx import DMRGX, mpo_energy, reflect, sector_bits
from mpslib import heisenberg_mpo, overlap, sparse_hamiltonian
from noiselib import s2_matrix

log = get_logger("lab42")

# Conformal-field-theory limit of n times the triplet gap, in sigma.sigma units.
GAP_LIMIT = 2 * np.pi**2


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def popcount(n):
    """Number of 1 bits (down spins) of every basis state of n qubits."""
    return np.array([bin(i).count("1") for i in range(2**n)])


def exact_sector(n, twice_sz, k):
    """Lowest k levels of one S_z sector by exact diagonalization.

    Returns (energies, full-space eigenvectors as columns). The sector is the
    set of basis states with the right number of down spins; the Hamiltonian
    is sliced to it, so no other sector can contaminate the result.
    """
    # Slice the sparse Hamiltonian to the sector before any diagonalization.
    ham = sparse_hamiltonian(n).tocsr()
    idx = np.flatnonzero(n - 2 * popcount(n) == twice_sz)
    block = ham[idx][:, idx]
    if len(idx) <= 400:
        vals, vecs = np.linalg.eigh(block.toarray())
        vals, vecs = vals[:k], vecs[:, :k]
    else:
        vals, vecs = eigsh(block, k=k, which="SA")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    full = np.zeros((2**n, k), dtype=complex)
    full[idx] = vecs
    return vals.real, full


def mirror_parity(vec, n):
    """Reflection parity <psi|R|psi> of a full-space vector (R reverses sites)."""
    basis = np.arange(2**n)
    # Reverse the bit order of every basis index.
    rev = np.zeros_like(basis)
    for q in range(n):
        rev |= ((basis >> q) & 1) << (n - 1 - q)
    return float(np.vdot(vec, vec[rev]).real)


def ladder(mpo, n, twice_sz, levels, chi, start=None, weight=20.0, tol=1e-11):
    """Find the lowest `levels` states of one sector, each orthogonal to the last.

    Returns (energies, list of MPS). `start` overrides the product start state
    (it must lie in the sector).
    """
    bits = sector_bits(n, twice_sz) if start is None else start
    # Each new run is penalised against every state found so far.
    found, energies = [], []
    for _ in range(levels):
        dmrg = DMRGX(mpo, chi, start_bits=bits, previous=found, weight=weight)
        dmrg.run(max_sweeps=40, tol=tol)
        energies.append(mpo_energy(mpo, dmrg.mps))
        found.append(dmrg.mps)
    return np.array(energies), found


def vector_of(mps):
    """Normalised full-space vector of an MPS (little-endian, as to_vector)."""
    vec = mps.to_vector()
    return vec / np.linalg.norm(vec)


def check_sectors(r, ctx):
    """Check 1: every S_z sector is reachable and stays exact."""
    log.info("=== Check 1: S_z sectors and their ground states ===")
    n = 8
    pc = popcount(n)
    # sector_bits must hit every allowed S_z and refuse impossible ones.
    ok = all(
        sum(1 - 2 * b for b in sector_bits(n, s)) == s for s in range(-n, n + 1, 2)
    )
    r.append(check(ok, "sector_bits reaches every S_z of 8 spins"))
    refused = 0
    for bad in (1, n + 2):
        try:
            sector_bits(n, bad)
        except ValueError:
            refused += 1
    r.append(check(refused == 2, "odd 2S_z and |S_z| > n/2 are refused"))
    # The run must stay inside its sector: no weight on any other S_z.
    mpo = heisenberg_mpo(n)
    worst = 0.0
    for s in (0, 2, 4):
        dmrg = DMRGX(mpo, 32, start_bits=sector_bits(n, s))
        dmrg.run(max_sweeps=30, tol=1e-11)
        vec = vector_of(dmrg.mps)
        outside = float(np.sum(np.abs(vec[n - 2 * pc != s]) ** 2))
        worst = max(worst, outside)
    log.info("weight outside the target sector: %.1e", worst)
    r.append(check(worst < 1e-20, "charge-blocked DMRG never leaves its S_z sector"))
    # Ground state of S_z = m has S = m: energy, <S^2> = m(m+1) from exact.
    n = ctx["n"]
    mpo = heisenberg_mpo(n)
    s2 = s2_matrix(n)
    for m in range(4):
        e_dmrg, found = ladder(mpo, n, 2 * m, 1, ctx["chi"])
        e_exact, _ = exact_sector(n, 2 * m, 1)
        vec = vector_of(found[0])
        spin2 = float((vec.conj() @ s2 @ vec).real)
        log.info(
            "n=%d S_z=%d: E=%.10f (exact %.10f)  <S^2>=%.6f",
            n,
            m,
            e_dmrg[0],
            e_exact[0],
            spin2,
        )
        r.append(check(abs(e_dmrg[0] - e_exact[0]) < 1e-9, f"S_z={m} ground energy"))
        r.append(check(abs(spin2 - m * (m + 1)) < 1e-6, f"S_z={m} ground has S={m}"))


def check_excited(r, ctx):
    """Check 2: five levels of the S_z = 0 sector against exact diagonalization."""
    log.info("=== Check 2: excited states by orthogonality penalty ===")
    n, levels = ctx["n"], 5
    mpo = heisenberg_mpo(n)
    # chi=32 equals the full bond dimension of 10 spins, so DMRG is exact here.
    energies, found = ladder(mpo, n, 0, levels, ctx["chi"])
    e_exact, v_exact = exact_sector(n, 0, levels)
    s2 = s2_matrix(n)
    spins, parities = [], []
    for k, mps in enumerate(found):
        vec = vector_of(mps)
        spin2 = float((vec.conj() @ s2 @ vec).real)
        # Parity from the MPS itself: overlap with its mirror image.
        par = overlap(reflect(mps), mps) / overlap(mps, mps)
        spins.append(spin2)
        parities.append(par.real)
        # The exact eigenvector has the same parity if the level is isolated.
        par_exact = mirror_parity(v_exact[:, k], n)
        log.info(
            "level %d: E=%.10f (exact %.10f) <S^2>=%.5f parity %+.0f (exact %+.0f)",
            k,
            energies[k],
            e_exact[k],
            spin2,
            par.real,
            par_exact,
        )
        r.append(check(abs(par.real - par_exact) < 1e-6, f"level {k} parity"))
    err = float(np.abs(energies - e_exact).max())
    r.append(check(err < 1e-9, f"all {levels} energies match exact (max {err:.1e})"))
    # Total spin must be a good quantum number: <S^2> = S(S+1) for integer S.
    # S(S+1) for S = 0..5.
    allowed = np.array([0.0, 2.0, 6.0, 12.0, 20.0, 30.0])
    dev = max(float(np.min(np.abs(allowed - s))) for s in spins)
    r.append(check(dev < 1e-6, f"every level has integer spin (max dev {dev:.1e})"))
    r.append(
        check(bool(np.all(np.diff(energies) > 0)), "levels are in ascending order")
    )
    # The levels are mutually orthogonal.
    worst = 0.0
    for a in range(levels):
        for b in range(a):
            ov = abs(overlap(found[b], found[a])) / np.sqrt(
                overlap(found[a], found[a]).real * overlap(found[b], found[b]).real
            )
            worst = max(worst, float(ov))
    r.append(check(worst < 1e-8, f"levels are orthogonal (max overlap {worst:.1e})"))
    ctx["sz0"] = (energies, spins)


def check_multiplets(r, ctx):
    """Check 3: levels found in S_z = 1 are the S >= 1 levels found in S_z = 0."""
    log.info("=== Check 3: SU(2) multiplets across sectors ===")
    n = ctx["n"]
    mpo = heisenberg_mpo(n)
    e0, s0 = ctx["sz0"]
    # The S_z=1 ladder is computed from its own start state, independently.
    e1, _ = ladder(mpo, n, 2, 3, ctx["chi"])
    # Levels of S_z=0 with S >= 1 (<S^2> > 1), in order.
    # Keep only the S_z=0 levels that belong to multiplets with S >= 1.
    triplets = e0[np.array(s0) > 1.0]
    m = min(len(triplets), len(e1))
    log.info("S_z=0 levels with S>=1: %s", np.round(triplets, 8))
    log.info("S_z=1 ladder:           %s", np.round(e1, 8))
    err = float(np.abs(triplets[:m] - e1[:m]).max())
    r.append(check(m >= 2, f"found {m} levels to compare"))
    r.append(
        check(err < 1e-8, f"S>=1 levels of S_z=0 equal the S_z=1 ladder ({err:.1e})")
    )
    # The lowest S=2 level shows up as the ground state of S_z = 2.
    # Eight S_z=0 levels reach past the S=2 ground energy of this chain.
    e2, _ = ladder(mpo, n, 4, 1, ctx["chi"])
    full, _ = ladder(mpo, n, 0, 8, ctx["chi"])
    r.append(
        check(bool(np.min(np.abs(full - e2[0])) < 1e-8), "S=2 ground is also in S_z=0")
    )


def check_robust(r, ctx):
    """Check 4: the answer does not depend on the start state or on the penalty."""
    log.info("=== Check 4: start state and penalty weight ===")
    n = 10
    mpo = heisenberg_mpo(n)
    e_exact, _ = exact_sector(n, 0, 4)
    rng = np.random.default_rng(ctx["seed"])
    worst = 0.0
    # Random product states with the same S_z must reach the same four levels.
    for _ in range(3):
        bits = [int(b) for b in rng.permutation(sector_bits(n, 0))]
        energies, _ = ladder(mpo, n, 0, 4, 32, start=bits)
        worst = max(worst, float(np.abs(energies - e_exact).max()))
    r.append(
        check(worst < 1e-9, f"3 random product starts agree (max err {worst:.1e})")
    )
    # Any penalty larger than the gap works; the result must not depend on it.
    errs = []
    for weight in (8.0, 20.0, 60.0):
        energies, _ = ladder(mpo, n, 0, 3, 32, weight=weight)
        errs.append(float(np.abs(energies - e_exact[:3]).max()))
    log.info("max error for w = 8, 20, 60: %s", ["%.1e" % e for e in errs])
    r.append(check(max(errs) < 1e-9, "penalty weights 8, 20, 60 agree"))
    # A penalty much SMALLER than the gap does not push the state away.
    weak, _ = ladder(mpo, n, 0, 2, 32, weight=0.05)
    log.info("weak penalty: E1 = %.6f, true E1 = %.6f", weak[1], e_exact[1])
    r.append(
        check(weak[1] < e_exact[1] - 0.5, "a penalty below the gap fails (as it must)")
    )


def check_chi(r, ctx):
    """Check 5: the energy error shrinks with the bond dimension chi."""
    log.info("=== Check 5: convergence with chi (16 spins, S_z = 1) ===")
    n = 16
    mpo = heisenberg_mpo(n)
    exact, _ = exact_sector(n, 2, 1)
    # Compare the true error with the SVD's own truncation estimate.
    errs, discards = [], []
    for chi in (4, 8, 16, 32):
        dmrg = DMRGX(mpo, chi, start_bits=sector_bits(n, 2))
        dmrg.run(max_sweeps=30, tol=1e-11)
        errs.append(mpo_energy(mpo, dmrg.mps) - exact[0])
        discards.append(dmrg.discarded)
        log.info(
            "chi=%2d: error %.2e, discarded weight %.2e", chi, errs[-1], discards[-1]
        )
    r.append(check(bool(np.all(np.diff(errs) < 0)), "error falls at every chi"))
    # DMRG is variational, so the error can never be negative.
    r.append(check(min(errs) > -1e-9, "energy never below the exact value"))
    # The error is a modest multiple of the discarded weight (about 25-65 here).
    ratio = max(e / d for e, d in zip(errs, discards))
    r.append(check(ratio < 150, f"error / discarded weight <= {ratio:.0f}"))
    r.append(check(errs[-1] < 1e-8, f"chi=32 is converged ({errs[-1]:.1e})"))


def check_gap(r, ctx):
    """Check 6: the triplet gap times n, for chains beyond exact reach."""
    log.info("=== Check 6: triplet gap versus chain length ===")
    # Chains beyond exact reach for a dense solver; n=16 is still checked.
    sizes = ctx["gap_sizes"]
    scaled = []
    for n in sizes:
        mpo = heisenberg_mpo(n)
        e0 = ladder(mpo, n, 0, 1, 48, tol=1e-9)[0][0]
        e1 = ladder(mpo, n, 2, 1, 48, tol=1e-9)[0][0]
        scaled.append(n * (e1 - e0))
        log.info("n=%2d: E0=%.8f E1=%.8f n*gap=%.4f", n, e0, e1, scaled[-1])
        if n == 16:
            ex0, _ = exact_sector(n, 0, 1)
            ex1, _ = exact_sector(n, 2, 1)
            r.append(
                check(
                    abs((e1 - e0) - (ex1[0] - ex0[0])) < 1e-7,
                    "n=16 gap matches exact diagonalization",
                )
            )
    r.append(check(bool(np.all(np.diff(scaled) > 0)), "n*gap rises with n"))
    r.append(
        check(max(scaled) < GAP_LIMIT, f"n*gap stays below 2 pi^2 = {GAP_LIMIT:.3f}")
    )
    r.append(check(min(scaled) > 0.5 * GAP_LIMIT, "n*gap is within a factor 2 of it"))


def main(argv=None):
    """Run every Lab 42 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 42: DMRG sectors and excited states")
    # The seed fixes the random product states of Check 4.
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="skip the longest chain")
    args = ap.parse_args(argv)
    ctx = {
        "seed": args.seed,
        "n": 10,
        "chi": 32,
        "gap_sizes": (16, 24) if args.quick else (16, 24, 32),
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_sectors(r, ctx)
    check_excited(r, ctx)
    check_multiplets(r, ctx)
    check_robust(r, ctx)
    check_chi(r, ctx)
    check_gap(r, ctx)
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
