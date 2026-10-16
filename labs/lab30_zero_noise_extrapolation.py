#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab30_zero_noise_extrapolation.py

Purpose
-------
Lab 29 showed that gate noise destroys the advantage of a deeper variational
circuit. Error MITIGATION recovers part of it without error correction. Zero-
noise extrapolation (ZNE) runs the same circuit at several amplified noise
levels and extrapolates the measured energy back to the zero-noise limit.

Noise is amplified by UNITARY FOLDING: each gate layer U becomes
U (U^dagger U)^k. The ideal circuit is unchanged (U^dagger U = 1) but there
are 1 + 2k times as many noisy layers, so the noise scale is lambda = 1 + 2k.
The measured energies E(1), E(3), E(5) are fitted with a polynomial in lambda
and evaluated at lambda = 0 (Richardson extrapolation).

The lab verifies:
  1. Folding leaves the ideal energy unchanged and rejects even factors.
  2. Folded noise matches simply scaling the error rate (small p).
  3. On an exact density matrix (n=6) a linear and then a quadratic fit cut
     the noise bias by a large factor each.
  4. The leftover bias of the quadratic fit grows faster with p than the raw
     bias does (extrapolation works only while noise is small).
  5. On a 16-site chain with trajectories, the mitigated energy is much closer
     to the ideal one than the raw noisy energy.
  6. The price: the statistical error of the mitigated value is larger, so
     more trajectories (shots) are needed; the overhead is reported.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab30_zero_noise_extrapolation.py
    python3 labs/lab30_zero_noise_extrapolation.py --trajectories 800 --rate 5e-4
    python3 labs/lab30_zero_noise_extrapolation.py --quick

Log files
---------
logs/lab30/lab30_<YYYYmmdd_HHMMSS>.log

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
    dm_energy,
    fold_steps,
    mean_se,
    richardson_weights,
    sample,
)

log = get_logger("lab30")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def extrapolate(scales, means, ses):
    """Zero-noise estimate and its standard error from measurements."""
    w = richardson_weights(scales)
    value = float(w @ np.asarray(means))
    error = float(np.sqrt(np.sum((w * np.asarray(ses)) ** 2)))
    return value, error


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_folding(r, ctx):
    """Check 1: folding is an identity on the ideal circuit."""
    log.info("=== Check 1: unitary folding (n=6) ===")
    params = ctx["angles"][2]
    base = dm_energy(6, params, 0.0)
    gaps = [abs(dm_energy(6, params, 0.0, fold=f) - base) for f in (3, 5)]
    # Folded circuits do the same ideal computation, so energies must agree.
    r.append(
        check(
            max(gaps) < 1e-9, f"folded circuits are ideal-equivalent ({max(gaps):.1e})"
        )
    )
    # Layer counts are exact, so this is an equality check, not a tolerance.
    steps = [(1, 0.3), (0, -0.2)]
    r.append(
        check(
            len(fold_steps(steps, 5)) == 5 * len(steps), "factor 5 gives 5x the layers"
        )
    )
    try:
        fold_steps(steps, 2)
        rejected = False
    except ValueError:
        rejected = True
    r.append(check(rejected, "an even folding factor is rejected"))


def check_scaling(r, ctx):
    """Check 2: folding by lambda is the same as the rate lambda * p."""
    n, p = 6, ctx["p_exact"]
    params = ctx["angles"][2]
    clean = run_hva(n, params, 64).energy()
    log.info("=== Check 2: folded noise versus scaled rate (n=6, p=%g) ===", p)
    worst = 0.0
    # Compare against the rate lambda*p: equal to first order in p.
    for lam in (3, 5):
        folded = dm_energy(n, params, p, fold=lam) - clean
        scaled = dm_energy(n, params, lam * p) - clean
        worst = max(worst, abs(folded - scaled) / folded)
        log.info(
            "lambda=%d: folded bias %.4f | scaled-rate bias %.4f", lam, folded, scaled
        )
    r.append(check(worst < 0.03, f"they agree within 3% ({worst:.1%})"))
    ctx["clean6"] = clean


def exact_curve(ctx, p):
    """Exact biases E(lambda) - E(0) for lambda = 1, 3, 5 at rate p (n=6)."""
    params = ctx["angles"][2]
    return np.array(
        [dm_energy(6, params, p, fold=lam) - ctx["clean6"] for lam in (1, 3, 5)]
    )


def check_exact_zne(r, ctx):
    """Checks 3 and 4: extrapolation on the exact density matrix."""
    p = ctx["p_exact"]
    log.info("=== Check 3: extrapolation without sampling noise (n=6) ===")
    bias = exact_curve(ctx, p)
    raw = bias[0]
    lin = extrapolate([1, 3], bias[:2], [0, 0])[0]
    # Three points fix a parabola; the zero-noise value is its intercept.
    quad = extrapolate([1, 3, 5], bias, [0, 0, 0])[0]
    log.info(
        "p=%g: raw bias %.4f | linear fit %.4f | quadratic fit %.4f", p, raw, lin, quad
    )
    # Factors are generous: the measured values are about 20x, 150x.
    r.append(
        check(abs(lin) < abs(raw) / 5, "a linear fit removes over 80% of the bias")
    )
    r.append(
        check(
            abs(quad) < abs(lin) / 4, "a quadratic fit leaves under a quarter of that"
        )
    )
    r.append(
        check(abs(quad) < abs(raw) / 30, "the bias falls by a factor of 30 or more")
    )
    log.info("=== Check 4: leftover bias versus error rate ===")
    # A larger rate shows how fast the leftover bias grows with noise.
    bias2 = exact_curve(ctx, 2.5 * p)
    quad2 = extrapolate([1, 3, 5], bias2, [0, 0, 0])[0]
    grow_raw, grow_quad = bias2[0] / raw, quad2 / quad
    log.info(
        "rate x2.5: raw bias x%.2f, quadratic-fit leftover x%.1f", grow_raw, grow_quad
    )
    r.append(
        check(
            grow_quad > 2 * grow_raw,
            "mitigated bias grows faster than raw bias (valid only at low noise)",
        )
    )


def check_trajectories(r, ctx):
    """Checks 5 and 6: ZNE with sampled trajectories on a longer chain."""
    n, chi, count, p = ctx["n"], ctx["chi"], ctx["count"], ctx["rate"]
    params = ctx["angles"][3]
    clean = run_hva(n, params, chi).energy()
    log.info("=== Check 5: n=%d, 4 layers, p=%g, N=%d per scale ===", n, p, count)
    # Own seed offset, so changing --seed changes every sampled number.
    rng = np.random.default_rng(ctx["seed"] + 5)
    means, ses = [], []
    # lambda = 1, 3, 5 are the odd folding factors the circuit can realise.
    for lam in (1, 3, 5):
        # Error-free trajectories reuse the ideal energy (folding keeps it).
        es = sample(n, params, chi, p, count, rng, clean, fold=lam)
        m, s = mean_se(es)
        means.append(m)
        ses.append(s)
        log.info("lambda=%d: bias %.4f +- %.4f", lam, m - clean, s)
    # Standard errors combine in quadrature with the squared weights.
    value, error = extrapolate([1, 3, 5], means, ses)
    raw_bias, mit_bias = means[0] - clean, value - clean
    log.info(
        "raw bias %.4f +- %.4f | mitigated bias %.4f +- %.4f",
        raw_bias,
        ses[0],
        mit_bias,
        error,
    )
    r.append(check(raw_bias > 3 * ses[0], "the raw noisy energy is clearly biased"))
    # The statistical test is the point: the raw value is excluded by many
    # standard errors, the mitigated value is not excluded at all.
    # Compare to the ideal MPS energy, which a real device would not know.
    r.append(
        check(
            abs(mit_bias) < 3 * error,
            f"the mitigated energy is consistent with the ideal one "
            f"({abs(mit_bias) / error:.1f} standard errors; raw: "
            f"{raw_bias / ses[0]:.1f})",
        )
    )
    # Variance grows by the squared Richardson weights: the cost of ZNE.
    overhead = (error / ses[0]) ** 2
    log.info("=== Check 6: sampling cost of mitigation ===")
    log.info(
        "standard error x%.1f, so about %.0fx more trajectories for equal precision",
        error / ses[0],
        overhead,
    )
    r.append(
        check(
            error > 1.5 * ses[0], "the mitigated estimate is noisier than the raw one"
        )
    )


def main(argv=None):
    """Run every Lab 30 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 30: zero-noise extrapolation")
    ap.add_argument("--trajectories", type=int, default=1000, help="per noise scale")
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
        "p_exact": 0.002,
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
    check_folding(r, ctx)
    check_scaling(r, ctx)
    check_exact_zne(r, ctx)
    check_trajectories(r, ctx)
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
