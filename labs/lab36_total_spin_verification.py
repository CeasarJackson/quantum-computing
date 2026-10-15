#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab36_total_spin_verification.py

Purpose
-------
Lab 31 discarded runs whose total S_z is wrong. That only catches errors that
FLIP spins (X and Y); a Z error leaves S_z alone. The Heisenberg model has a
much bigger symmetry: the full SU(2) rotation symmetry, i.e. total spin S. The
start state is a singlet (S = 0) and every gate commutes with S^2, so an ideal
run stays at S = 0 for ever. This lab post-selects on S = 0.

Why it is stronger. A Pauli on one qubit is a spin-1 (vector) operator. Acting
on a singlet it can only produce S = 1, never S = 0, and the (S-conserving)
rest of the circuit cannot undo that. So EVERY single Pauli error, X, Y or Z,
is detected with certainty. Only pairs of errors can hide, so the residual bias
is O(p^2) instead of O(p).

The lab verifies (exact density matrices, so there is no sampling noise):
  1. The S^2 operator: its eigenvalues are s(s+1), the singlet space has
     dimension Catalan(n/2), the noiseless circuit never leaves it.
  2. The theorem: all 3K single-Pauli errors have exactly zero overlap with
     S = 0, whereas S_z misses every Z error (one third of them).
  3. Two errors can leak back into S = 0 (the O(p^2) residue).
  4. The scaling with p: raw and S_z verified bias grow like p, S^2 verified
     like p^2 (fitted log-log slope).
  5. The price: acceptance is about (1 - p)^K, so about 4% is discarded here.
  6. Noise type: pure Z (dephasing) noise fools S_z verification completely;
     S^2 verification does not care which Pauli it is.
  7. Combination with extrapolation: the verified data scale like p^2, so the
     straight-line fit of Lab 30 is the WRONG model; a quadratic fit works.
  8. A longer chain (n=10, n=8 in quick mode).

Honest limit. On hardware S_z comes free from the measured bit strings, but S^2
does not: it needs extra measurement circuits (ancilla-based). This lab is the
ideal, an upper bound on what total-spin verification can give.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab36_total_spin_verification.py
    python3 labs/lab36_total_spin_verification.py --rate 2e-3
    python3 labs/lab36_total_spin_verification.py --quick

Log files
---------
logs/lab36/lab36_<YYYYmmdd_HHMMSS>.log

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
from mpslib import layer_plan, run_hva, sparse_hamiltonian, train_angles
from noiselib import (
    dm_energy,
    dm_s2_verified,
    dm_sz_verified,
    dm_with_errors,
    richardson_weights,
    s2_matrix,
    singlet_projector,
    touched_qubits,
)

log = get_logger("lab36")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def locations(n, params):
    """Number K of noisy locations: touched qubits summed over every layer."""
    return sum(len(touched_qubits(n, parity)) for parity, _ in layer_plan(params))


def sz_weight(rho, n):
    """Probability that the state has total S_z = 0 (site-order basis)."""
    # Only the diagonal matters: S_z is diagonal in the computational basis.
    diag = np.real(np.diag(rho.reshape(2**n, 2**n)))
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    return float(diag[bits.sum(axis=1) == n // 2].sum())


def biases(n, params, clean, p, proj, weights=None, fold=1):
    """Return ((raw, sz, s2) biases, (sz, s2) acceptances) at one rate."""
    # All three numbers come from the same noisy density matrix, so the
    # comparison between them is exact, with no sampling noise.
    raw = dm_energy(n, params, p, fold, weights) - clean
    e_sz, a_sz = dm_sz_verified(n, params, p, fold, weights)
    e_s2, a_s2 = dm_s2_verified(n, params, p, fold, weights, proj)
    return (raw, e_sz - clean, e_s2 - clean), (a_sz, a_s2)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_machinery(r, ctx):
    """Check 1: the S^2 operator and the noiseless circuit."""
    log.info("=== Check 1: the total-spin operator ===")
    # Catalan numbers count the singlet states of n spins: C_3 = 5, C_4 = 14.
    for n, catalan in ((6, 5), (8, 14)):
        w = np.linalg.eigvalsh(s2_matrix(n))
        # s(s+1) for s = 0..n/2 are 0, 2, 6, 12, ...
        allowed = np.array([s * (s + 1) for s in range(n // 2 + 1)], dtype=float)
        ok = all(np.min(np.abs(allowed - x)) < 1e-9 for x in w)
        r.append(check(ok, f"n={n}: every eigenvalue of S^2 is s(s+1)"))
        r.append(
            check(
                int((w < 0.5).sum()) == catalan,
                f"n={n}: the singlet space has dimension Catalan({n // 2}) = {catalan}",
            )
        )
    n, params = 6, ctx["params6"]
    proj = ctx["proj6"]
    e, a = dm_s2_verified(n, params, 0.0, proj=proj)
    r.append(
        check(
            abs(a - 1) < 1e-12, "noiseless: acceptance is 1 (the circuit stays at S=0)"
        )
    )
    r.append(
        check(abs(e - ctx["clean6"]) < 1e-10, "noiseless: verification changes nothing")
    )
    # A wrong projector would silently give wrong acceptances everywhere below.
    # P is a projector: idempotent and symmetric.
    r.append(
        check(np.allclose(proj @ proj, proj), "the singlet projector is idempotent")
    )


def check_theorem(r, ctx):
    """Check 2: every single Pauli error leaves S = 0 completely."""
    log.info("=== Check 2: one error is always caught ===")
    n, params, proj = 6, ctx["params6"], ctx["proj6"]
    # Walk every noisy location the density-matrix simulator would use, and insert
    # each of the three Paulis there, one at a time.
    worst_s2 = 0.0
    # Counters of the errors that S_z lets through, by Pauli type.
    sz_pass = {"X": 0, "Y": 0, "Z": 0}
    total = 0
    for step, (parity, _) in enumerate(layer_plan(params)):
        for q in touched_qubits(n, parity):
            for name in "XYZ":
                rho = dm_with_errors(n, params, {(step, q): name})
                rho = rho.reshape(2**n, 2**n)
                worst_s2 = max(worst_s2, float(np.trace(proj @ rho @ proj).real))
                sz_pass[name] += int(sz_weight(rho, n) > 0.5)
                total += 1
    # The weight is the probability of S = 0, i.e. of surviving verification.
    log.info("single errors tested: %d; largest S=0 weight %.2e", total, worst_s2)
    log.info("errors passing S_z verification: %s", sz_pass)
    r.append(check(total == 3 * locations(n, params), "tested all 3K single errors"))
    r.append(check(worst_s2 < 1e-12, "no single error keeps any weight at S=0"))
    k = locations(n, params)
    r.append(
        check(sz_pass["X"] == 0 and sz_pass["Y"] == 0, "S_z catches every X and Y")
    )
    r.append(check(sz_pass["Z"] == k, "but S_z lets every Z error through"))


def check_two_errors(r, ctx):
    """Check 3: pairs of errors can return to S = 0."""
    log.info("=== Check 3: two errors can hide ===")
    n, params, proj = 6, ctx["params6"], ctx["proj6"]
    # Pairs on the same gate layer with the same Pauli are the likeliest to
    # return to S = 0 (they act like a rotation of two spins together).
    best = 0.0
    for step, (parity, _) in enumerate(layer_plan(params)):
        qs = touched_qubits(n, parity)
        for a in qs:
            for b in qs:
                if a < b:
                    for name in "XYZ":
                        rho = dm_with_errors(
                            n, params, {(step, a): name, (step, b): name}
                        )
                        rho = rho.reshape(2**n, 2**n)
                        best = max(best, float(np.trace(proj @ rho @ proj).real))
    log.info("largest S=0 weight after a pair of equal Paulis: %.3f", best)
    r.append(check(best > 0.1, "a pair of errors can return to S=0 (the p^2 leak)"))


def check_scaling(r, ctx):
    """Check 4 and 5: bias against p, and the acceptance price."""
    log.info("=== Check 4: bias scaling with p ===")
    n, params, proj, clean = 6, ctx["params6"], ctx["proj6"], ctx["clean6"]
    # Four decades-apart-ish rates give a stable log-log slope.
    rates = [1e-4, 3e-4, 1e-3, 3e-3]
    rows, accs = [], []
    k = locations(n, params)
    for p in rates:
        b, a = biases(n, params, clean, p, proj)
        rows.append(b)
        accs.append(a)
        log.info(
            "p=%.0e  raw %+.2e  S_z %+.2e  S^2 %+.2e | accept S_z %.4f S^2 %.4f",
            p,
            *b,
            *a,
        )
    # The slope of log|bias| against log p is the order of the leading error:
    # 1 means single errors survive, 2 means only pairs do.
    rows = np.array(rows)
    slopes = [
        np.polyfit(np.log(rates), np.log(np.abs(rows[:, c])), 1)[0] for c in range(3)
    ]
    log.info("fitted slopes: raw %.2f, S_z %.2f, S^2 %.2f", *slopes)
    r.append(check(0.9 < slopes[0] < 1.1, "raw bias grows like p (slope ~1)"))
    r.append(check(0.9 < slopes[1] < 1.1, "S_z verified bias also grows like p"))
    r.append(
        check(1.8 < slopes[2] < 2.2, "S^2 verified bias grows like p^2 (slope ~2)")
    )
    r.append(
        check(
            all(abs(x[2]) < abs(x[1]) < abs(x[0]) for x in rows),
            "S^2 < S_z < raw at every rate",
        )
    )
    # Reduction at the working rate p = 1e-3.
    # Reduction at the working rate p = 1e-3.
    mid = rows[2]
    r.append(
        check(
            abs(mid[2]) < 0.02 * abs(mid[0]), "at p=1e-3 S^2 removes >98% of the bias"
        )
    )
    log.info("=== Check 5: the price (acceptance) ===")
    for p, (a_sz, a_s2) in zip(rates, accs):
        # Acceptance is the chance that NO error happened, to second order.
        # Every error is flagged by S^2, two thirds by S_z.
        r.append(
            check(
                abs(a_s2 - (1 - p) ** k) < (k * p) ** 2,
                f"p={p:.0e}: S^2 acceptance equals (1-p)^K to second order",
            )
        )
        r.append(check(a_s2 <= a_sz + 1e-12, f"p={p:.0e}: S^2 discards more than S_z"))


def check_noise_types(r, ctx):
    """Check 6: the Pauli type of the noise does not matter to S^2."""
    log.info("=== Check 6: noise type ===")
    n, params, proj, clean = 6, ctx["params6"], ctx["proj6"], ctx["clean6"]
    p = 1e-3
    for label, w in (("dephasing (Z)", (0, 0, 1)), ("bit-flip type (X,Y)", (1, 1, 0))):
        (raw, sz, s2), (a_sz, _) = biases(n, params, clean, p, proj, weights=w)
        log.info("%s: raw %+.2e  S_z %+.2e  S^2 %+.2e", label, raw, sz, s2)
        # A Z error does not change S_z, so S_z verification cannot see it at all.
        if w == (0, 0, 1):
            r.append(check(abs(a_sz - 1) < 1e-10, "Z noise: S_z accepts every run"))
            r.append(
                check(abs(sz - raw) < 1e-10, "Z noise: S_z verification does nothing")
            )
        r.append(
            check(abs(s2) < 0.05 * abs(raw), f"{label}: S^2 removes >95% of the bias")
        )


def check_extrapolation(r, ctx):
    """Check 7: verified data scale like p^2, so fit a quadratic."""
    log.info("=== Check 7: extrapolating verified data ===")
    n, params, proj, clean = 6, ctx["params6"], ctx["proj6"], ctx["clean6"]
    folds = [1, 3, 5]
    for p in (1e-3, 3e-3):
        # Same circuit with 1, 3 and 5 times the gates: the noise scales, the
        # ideal state does not.
        v = [dm_s2_verified(n, params, p, f, proj=proj)[0] - clean for f in folds]
        # A straight line through folds 1 and 3, as in Lab 30.
        line = (3 * v[0] - v[1]) / 2
        # The verified bias goes as p^2, hence as fold^2: three points fit it.
        quad = float(richardson_weights(folds) @ np.array(v))
        log.info(
            "p=%.0e: single %+.2e  line %+.2e  quadratic %+.2e", p, v[0], line, quad
        )
        r.append(
            check(abs(line) > abs(v[0]), f"p={p:.0e}: the straight line makes it WORSE")
        )
        r.append(
            check(
                abs(quad) < 0.5 * abs(v[0]),
                f"p={p:.0e}: the quadratic fit halves the verified bias or better",
            )
        )


def check_long_chain(r, ctx):
    """Check 8: a longer chain."""
    n = ctx["n_long"]
    log.info("=== Check 8: n=%d ===", n)
    params = ctx["params8"]
    # Angles trained on 8 qubits are reused on 10: the bias is measured against
    # this chain's own error-free energy, so they need not be optimal.
    clean = run_hva(n, params, 64).energy()
    proj = singlet_projector(n)
    p = ctx["rate"]
    (raw, sz, s2), (a_sz, a_s2) = biases(n, params, clean, p, proj)
    log.info("p=%.0e: raw %+.2e  S_z %+.2e  S^2 %+.2e", p, raw, sz, s2)
    log.info("acceptance S_z %.3f  S^2 %.3f (cost: x%.2f runs)", a_sz, a_s2, 1 / a_s2)
    r.append(check(abs(raw) > 20 * abs(s2), "S^2 beats the raw bias by more than 20x"))
    r.append(check(abs(s2) < 0.5 * abs(sz), "and beats S_z by at least 2x"))
    r.append(check(1 / a_s2 < 1.5, "at a cost of less than 1.5x more runs"))


def main(argv=None):
    """Run every Lab 36 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 36: total-spin verification")
    ap.add_argument("--rate", type=float, default=1e-3, help="rate for the long chain")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if not 1e-4 <= args.rate <= 5e-3:
        ap.error(f"--rate must be between 1e-4 and 5e-3. Got {args.rate}.")
    ctx = {
        "seed": args.seed,
        # The density matrix for n=10 takes seconds; quick mode uses n=8.
        "n_long": 8 if args.quick else 10,
        "rate": args.rate,
    }
    # Angles are trained on small chains, exactly as in Labs 28-35.
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
    # Every bias is measured against this error-free energy.
    ctx["clean6"] = run_hva(6, ctx["params6"], 64).energy()
    # Built once: the eigen-decomposition is the slow step for larger n.
    ctx["proj6"] = singlet_projector(6)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_machinery(r, ctx)
    check_theorem(r, ctx)
    check_two_errors(r, ctx)
    check_scaling(r, ctx)
    check_noise_types(r, ctx)
    check_extrapolation(r, ctx)
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
