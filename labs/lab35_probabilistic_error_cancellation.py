#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab35_probabilistic_error_cancellation.py

Purpose
-------
Labs 30 and 31 mitigate noise without knowing it (extrapolation) or by
discarding bad runs (symmetry verification). PROBABILISTIC ERROR CANCELLATION
(PEC) does the opposite: it uses a measured noise model to undo the noise,
exactly, on average.

For depolarizing noise of rate p on one qubit, the inverse channel is a signed
combination of Paulis: q_I * (identity) + q_P * (X, Y, Z) with q_P < 0. Because
the q's are not probabilities ("quasi-probabilities") it cannot be run as a
circuit, but it can be SAMPLED: draw a Pauli with probability |q|/gamma, run the
circuit with it inserted, and multiply the result by sign(q) * gamma, where
gamma = |q_I| + 3 |q_P| >= 1. Over a whole circuit the weights multiply, giving
gamma_total = gamma^K for K noisy locations. The weighted mean is UNBIASED; the
price is variance, larger by about gamma_total^2, which grows exponentially with
circuit size and noise.

The lab verifies:
  1. The inverse channel algebra and the per-location overhead gamma.
  2. Exactness on a density matrix (n=6): noise followed by the quasi-
     probability inverse gives the ideal energy to rounding error, at any
     depth.
  3. The overhead: gamma_total grows like exp(2 p K); a table for n=16.
  4. Sampling (n=6): the weighted mean equals the clean energy; the raw noisy
     mean does not; E[weight] = 1.
  5. The variance lesson: weighting the raw energy gives a large variance
     (it multiplies a number near -20 by weights of size gamma_total); shifting
     the observable to H - c, which is legal for any constant c, removes most
     of it.
  6. Model error: PEC with a wrong rate leaves a bias proportional to the
     relative error (exact); the noise TYPE hardly matters here, because this
     circuit's energy depends on the noise strength.
  7. Long chain (n=16): PEC recovers the clean energy where the raw run is far
     off.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab35_probabilistic_error_cancellation.py
    python3 labs/lab35_probabilistic_error_cancellation.py --samples 600
    python3 labs/lab35_probabilistic_error_cancellation.py --quick

Log files
---------
logs/lab35/lab35_<YYYYmmdd_HHMMSS>.log

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
    dm_expectation,
    dm_pec_state,
    inverse_depolarizing,
    mean_se,
    noisy_energy,
    pec_mps,
    touched_qubits,
)

log = get_logger("lab35")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def locations(n, params, fold=1):
    """Number K of noisy locations: touched qubits summed over every layer."""
    return fold * sum(
        len(touched_qubits(n, parity)) for parity, _ in layer_plan(params)
    )


def pec_sample(n, params, chi, p, count, rng, clean, shift=0.0, p_model=None):
    """Weighted samples w * (E - shift); returns (estimate - clean, se, weights)."""
    # Parallel lists: weighted values for the mean, bare weights for E[w].
    values, weights = [], []
    for _ in range(count):
        mps, weight = pec_mps(n, params, chi, p, rng, p_model)
        # None means no Pauli was applied anywhere: the ideal state.
        energy = clean if mps is None else mps.energy()
        values.append(weight * (energy - shift))
        weights.append(weight)
    mean, se = mean_se(np.array(values))
    # E[weight] = 1 exactly, so adding `shift` back restores the energy.
    return mean + shift - clean, se, np.array(weights)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_algebra(r, ctx):
    """Check 1: the inverse of the depolarizing channel."""
    log.info("=== Check 1: the inverse channel ===")
    # A rate far above realistic hardware, so that rounding cannot hide an error.
    p = 0.02
    q_i, q_p, gamma = inverse_depolarizing(p)
    r.append(check(abs(q_i + 3 * q_p - 1) < 1e-12, "the quasi-probabilities sum to 1"))
    r.append(check(q_p < 0 < q_i, "the Pauli weights are negative: not a probability"))
    # Apply the noise, then the inverse, to a random one-qubit state.
    # A random pure state: the identity must hold for every input, not just |0>.
    rng = np.random.default_rng(ctx["seed"])
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    rho = np.outer(v, v.conj())
    paulis = [
        np.array(m, dtype=complex)
        for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]])
    ]
    noisy = (1 - p) * rho + sum(p / 3 * m @ rho @ m for m in paulis)
    back = q_i * noisy + sum(q_p * m @ noisy @ m for m in paulis)
    r.append(
        check(
            np.allclose(back, rho, atol=1e-12), "noise then inverse returns the state"
        )
    )
    # lambda is the factor by which depolarizing noise shrinks X, Y and Z.
    lam = 1 - 4 * p / 3
    r.append(check(abs(gamma - (3 / lam - 1) / 2) < 1e-12, "gamma = (3/lambda - 1)/2"))
    # Realistic rates are tiny, where gamma is well approximated by 1 + 2p.
    small = 1e-4
    r.append(
        check(
            abs(inverse_depolarizing(small)[2] - 1 - 2 * small) < 1e-7,
            "gamma is 1 + 2p for small p",
        )
    )
    r.append(
        check(
            inverse_depolarizing(0.0) == (1.0, 0.0, 1.0), "no noise: nothing to cancel"
        )
    )


def check_exact(r, ctx):
    """Check 2: noise plus quasi-probability inverse is the identity (exact)."""
    # Exactness is checked on the density matrix, where no sampling noise exists.
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    log.info("=== Check 2: exact cancellation, n=6 ===")
    for p in (0.003, 0.01):
        raw = dm_energy(n, params, p) - clean
        rho = dm_pec_state(n, params, p)
        res = dm_expectation(rho, n) - clean
        log.info("p=%g: raw bias %.4f, after PEC %.1e", p, raw, res)
        r.append(
            check(
                raw > 0.3 and abs(res) < 1e-9,
                f"p={p:g}: bias {raw:.3f} removed to {abs(res):.0e}",
            )
        )
    # The circuit depth does not matter to exactness: fold 3 has 3x the noise.
    # Folding triples the number of noisy locations without changing the ideal circuit.
    raw3 = dm_energy(n, params, 0.003, fold=3) - clean
    res3 = dm_expectation(dm_pec_state(n, params, 0.003, fold=3), n) - clean
    r.append(
        check(raw3 > 1 and abs(res3) < 1e-9, f"also at 3x the depth (bias {raw3:.2f})")
    )
    trace = float(np.trace(dm_pec_state(n, params, 0.003).reshape(64, 64)).real)
    r.append(check(abs(trace - 1) < 1e-9, "and the trace stays 1"))


def check_overhead(r, ctx):
    """Check 3: the sampling overhead gamma_total."""
    log.info("=== Check 3: sampling overhead ===")
    # K counts every place where an error can strike: qubit times layer.
    k = locations(ctx["n"], ctx["params8"])
    k2 = locations(ctx["n"], ctx["params8"], fold=2)
    log.info("n=%d, 4 layers: K=%d noisy locations", ctx["n"], k)
    # Overheads are products over locations, so they are exponential in K.
    totals = []
    for p in (1e-3, 3e-3, 1e-2):
        gamma = inverse_depolarizing(p)[2]
        totals.append(gamma**k)
        log.info(
            "p=%g: gamma=%.5f, gamma_total=%.2f, variance factor %.1f",
            p,
            gamma,
            gamma**k,
            gamma ** (2 * k),
        )
    r.append(check(totals[0] < 1.5, "p=1e-3: the overhead is mild"))
    r.append(
        check(
            totals[2] > 5 and totals == sorted(totals),
            "p=1e-2: gamma_total > 5 and growing with p",
        )
    )
    # Twice the depth means twice the locations, hence gamma_total squared.
    g = inverse_depolarizing(3e-3)[2]
    r.append(
        check(abs(g**k2 - (g**k) ** 2) < 1e-9, "doubling the depth squares gamma_total")
    )
    r.append(
        check(
            abs(np.log(g**k) - 2 * 3e-3 * k) < 0.01 * k * 3e-3,
            "log gamma_total is about 2 p K",
        )
    )


def check_sampling(r, ctx):
    """Check 4 and 5: sampled PEC on n=6, and the variance lesson."""
    n, params, clean, p = 6, ctx["params6"], ctx["clean6"], 0.003
    log.info("=== Check 4: sampled PEC, n=6, p=%g, N=%d ===", p, ctx["small"])
    # The pilot batch fixes the shift c; any constant keeps the estimator unbiased.
    rng = np.random.default_rng(ctx["seed"] + 2)
    pilot = np.mean(
        [noisy_energy(n, params, 64, p, rng, e_clean=clean) for _ in range(50)]
    )
    # Weighted samples w * (E - c) with E[w] = 1 give c + mean = the clean energy.
    est, se, weights = pec_sample(
        n, params, 64, p, ctx["small"], rng, clean, shift=pilot
    )
    raw = [
        noisy_energy(n, params, 64, p, rng, e_clean=clean) for _ in range(ctx["small"])
    ]
    raw_mean, raw_se = mean_se(np.array(raw))
    log.info(
        "PEC %.4f +- %.4f | raw bias %.4f +- %.4f", est, se, raw_mean - clean, raw_se
    )
    r.append(check(raw_mean - clean > 3 * raw_se, "the raw noisy mean is biased"))
    r.append(check(abs(est) < 4 * se, "the PEC mean equals the clean energy"))
    wm, wse = mean_se(weights)
    r.append(check(abs(wm - 1) < 4 * wse, f"the weights average to 1 ({wm:.4f})"))
    log.info("=== Check 5: why the observable is shifted ===")
    # Shift 0: the raw energy near -10 is multiplied by weights of size gamma_total.
    plain, plain_se, _ = pec_sample(n, params, 64, p, ctx["small"], rng, clean)
    log.info(
        "without the shift: %.4f +- %.4f; with: %.4f +- %.4f", plain, plain_se, est, se
    )
    r.append(
        check(
            plain_se > 1.3 * se,
            f"the unshifted error bar is {plain_se / se:.1f}x larger",
        )
    )
    r.append(check(abs(plain) < 4 * plain_se, "but its mean is still unbiased"))


def check_mismatch(r, ctx):
    """Check 6: a wrong noise model leaves a bias (exact)."""
    n, params, clean, p = 6, ctx["params6"], ctx["clean6"], 0.003
    log.info("=== Check 6: model error, exact ===")
    # factor is the assumed rate divided by the true rate.
    raw = dm_energy(n, params, p) - clean
    for factor in (0.5, 0.8, 1.2):
        res = dm_expectation(dm_pec_state(n, params, p, p_model=factor * p), n) - clean
        want = 1 - factor
        log.info(
            "assumed rate x%.1f: residual is %.3f of the raw bias (1 - x = %.2f)",
            factor,
            res / raw,
            want,
        )
        r.append(
            check(
                abs(res / raw - want) < 0.03,
                f"rate x{factor}: the residual fraction is 1 - x",
            )
        )
    # A wrong noise TYPE: dephasing or bit-flip noise corrected as if depolarizing.
    # Weights (0,0,1) is pure dephasing, (1,1,0) bit flips, (1,0,0) X only.
    worst = 0.0
    for weights in ((0, 0, 1), (1, 1, 0), (1, 0, 0)):
        raw_w = dm_energy(n, params, p, weights=weights) - clean
        res = dm_expectation(dm_pec_state(n, params, p, weights=weights), n) - clean
        worst = max(worst, abs(res / raw_w))
    log.info("wrong noise type: worst residual fraction %.4f", worst)
    r.append(
        check(
            worst < 0.02, "a wrong noise type matters little for this circuit's energy"
        )
    )


def check_long_chain(r, ctx):
    """Check 7: the long chain."""
    n, chi, count, p = ctx["n"], 32, ctx["samples"], ctx["rate"]
    params = ctx["params8"]
    clean = run_hva(n, params, chi).energy()
    k = locations(n, params)
    # The predicted cost of PEC at this size and noise, before running anything.
    gamma_total = inverse_depolarizing(p)[2] ** k
    log.info(
        "=== Check 7: n=%d, p=%g, N=%d, gamma_total=%.3f ===", n, p, count, gamma_total
    )
    rng = np.random.default_rng(ctx["seed"] + 7)
    pilot = np.mean(
        [noisy_energy(n, params, chi, p, rng, e_clean=clean) for _ in range(50)]
    )
    est, se, _ = pec_sample(n, params, chi, p, count, rng, clean, shift=pilot)
    # A fresh raw batch gives the baseline bias and error bar.
    raw = [noisy_energy(n, params, chi, p, rng, e_clean=clean) for _ in range(count)]
    raw_mean, raw_se = mean_se(np.array(raw))
    log.info(
        "raw bias %.4f +- %.4f | PEC %.4f +- %.4f", raw_mean - clean, raw_se, est, se
    )
    r.append(check(raw_mean - clean > 3 * raw_se, "the raw noisy energy is biased"))
    r.append(check(abs(est) < 4 * se, "PEC recovers the clean energy"))
    r.append(check(se < 8 * raw_se, f"at an error bar {se / raw_se:.1f}x the raw one"))
    # The same estimator without the shift: a much larger error bar here.
    plain, plain_se, _ = pec_sample(n, params, chi, p, count, rng, clean)
    log.info(
        "unshifted: %.4f +- %.4f (%.1fx the shifted error bar)",
        plain,
        plain_se,
        plain_se / se,
    )
    r.append(
        check(
            plain_se > 2 * se,
            "the shift cuts the error bar more than 2x on the long chain",
        )
    )


def main(argv=None):
    """Run every Lab 35 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 35: probabilistic error cancellation")
    ap.add_argument("--samples", type=int, default=400, help="long chain")
    ap.add_argument("--rate", type=float, default=5e-4, help="depolarizing rate p")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.samples < 50:
        ap.error(f"--samples must be at least 50. Got {args.samples}.")
    if not 1e-5 <= args.rate <= 3e-3:
        ap.error(f"--rate must be between 1e-5 and 3e-3. Got {args.rate}.")
    quick = args.quick
    # Quick mode shortens the chain and uses fewer samples.
    ctx = {
        "seed": args.seed,
        "n": 12 if quick else 16,
        "samples": 150 if quick else args.samples,
        # Quick mode has fewer samples, so it uses a stronger noise rate.
        "rate": 2 * args.rate if quick else args.rate,
        "small": 600 if quick else 2000,
    }
    # Angles are trained on small chains, exactly as in Labs 28-34.
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
    # Every bias in the lab is measured against this error-free energy.
    ctx["clean6"] = run_hva(6, ctx["params6"], 64).energy()
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_algebra(r, ctx)
    check_exact(r, ctx)
    check_overhead(r, ctx)
    check_sampling(r, ctx)
    check_mismatch(r, ctx)
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
