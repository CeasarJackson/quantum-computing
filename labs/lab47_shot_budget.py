#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab47_shot_budget.py

Purpose
-------
Lab 44 could not tell whether noise-aware angles beat ideal angles on ibm_fez:
a 3-sigma detection of the predicted depth-3 gain needed about 8400 shots per
basis and the lab capped shots at 4000. Lab 47 asks two questions with exact
numbers (no hardware, no network):

  1. CAN THE SAME ANSWER BE BOUGHT WITH FEWER SHOTS? Three ideas were on the
     table, and each is tested against the exact measurement distributions of
     the six Lab 44 circuits (noisy density matrices from the Lab 21 model):
       a. Measure only Z and use the SU(2) symmetry. A state that commutes
          with global rotations has <XX> = <YY> = <ZZ> on every bond, so 3 <ZZ>
          estimates the energy. RESULT: the variance is IDENTICAL to the three
          basis estimator (three independent estimates of one number, averaged,
          cost the same as one estimate with the same total shots), and the
          estimator is wrong as soon as the noise breaks the symmetry
          (amplitude damping: bias 0.06 at 1 % damping).
       b. Optimal shot allocation between X, Y and Z. RESULT: gain 1.0000 (the
          three variances are equal by the same symmetry).
       c. Classical shadows (a random Pauli basis on every qubit). RESULT:
          3.8 to 5.9 times MORE variance, because a bond term is measured
          only when both its qubits happen to share a basis (probability 1/3
          per bond and Pauli, not 1).
     My first guess was that the symmetry would save a factor of up to three;
     it does not, and the lab says so.
  2. WHAT DOES THE DECISIVE RUN COST? Shots per basis for a 3-sigma and 5-sigma
     detection, with the variance of the model and with the larger variance the
     hardware showed, for the depth-3 gain, and for the depth-1 and depth-2
     pairs (whose predicted gain is 0.0001 and 0.0023: unresolvable, so they
     are NULL controls). Then a plan: ideal, noise-trained and a sign-flipped
     copy of the ideal angles (the same noisy energy by time-reversal) at
     depth 3, with the QPU time that costs at the rate of the recorded run.
     A Monte-Carlo test checks that the shot formula really gives the power
     it promises, and that a null pair is flagged at the nominal rate.

Honest limit: the plan is a calculation from the recorded run, not a run. The
QPU seconds assume the rate seen once (16 s for 40000 shots); a real account
may bill differently.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab47_shot_budget.py                 # about 20 s
    python3 labs/lab47_shot_budget.py --quick         # fewer Monte-Carlo trials
    python3 labs/lab47_shot_budget.py --trials 1000 --seed 4

Log files
---------
logs/lab47/lab47_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import itertools
import math
import sys

import numpy as np

from applog import get_logger

log = get_logger("lab47")

try:
    from fezdata import (
        ANGLES,
        CIRCUITS,
        LAMBDA_FEZ,
        MEASURED,
        RUNNING_SECONDS,
        SHOTS,
        fit_lines,
        shortfalls,
    )
    from lab21_noise_aware_training import N, NoisyChain
    from lab44_noise_aware_on_qpu import energy_from_counts
except ImportError:  # pragma: no cover
    log.fail("qiskit is not installed: pip install qiskit qiskit-aer")
    sys.exit(2)

I2 = np.eye(2)
H1 = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
# Rotations that turn a Pauli measurement into a Z measurement. Y needs S-dagger
# first (the same gates as Lab 44's basis circuits).
ROT = {"Z": I2, "X": H1, "Y": H1 @ np.diag([1, -1j])}
BITS = (np.arange(2**N)[:, None] >> np.arange(N)) & 1  # bit q = qubit q
# Sum over bonds of the +/-1 correlator that one bit string gives for a basis.
F = sum((-1.0) ** (BITS[:, q] + BITS[:, q + 1]) for q in range(N - 1))


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Exact measurement distributions
# ----------------------------------------------------------------------------
def kron_all(mats):
    """Tensor product with mats[q] acting on qubit q (qubit 0 least significant)."""
    out = np.array([[1.0 + 0j]])
    for m in reversed(mats):
        out = np.kron(out, m)
    return out


def probs(rho, u):
    """Outcome probabilities diag(U rho U^dagger) without forming the product."""
    return np.real(np.einsum("ij,jk,ik->i", u, rho, u.conj()))


def basis_stats(rho):
    """{basis: (probabilities, mean, variance)} of the bond sum in X, Y and Z."""
    out = {}
    for a in "XYZ":
        p = probs(rho, kron_all([ROT[a]] * N))
        mean = float(p @ F)
        out[a] = (p, mean, float(p @ F**2) - mean**2)
    return out


def energy_var(rho):
    """(energy, V) where V = sum over bases of the per-shot variance.

    With n shots in EACH basis the standard error of the energy is sqrt(V / n).
    """
    st = basis_stats(rho)
    return sum(m for _, m, _ in st.values()), sum(v for _, _, v in st.values())


def damp(rho, gamma):
    """Amplitude damping of strength gamma on every qubit (breaks SU(2))."""
    k0 = np.diag([1.0, math.sqrt(1 - gamma)])
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    for q in range(N):
        ks = [kron_all([k if i == q else I2 for i in range(N)]) for k in (k0, k1)]
        rho = sum(k @ rho @ k.conj().T for k in ks)
    return rho


def shadow_stats(rho):
    """(mean, variance) per shot of the random-Pauli (classical shadow) estimator.

    Each shot picks X, Y or Z for every qubit at random. A bond term P_q P_(q+1)
    is read out only when both qubits drew P, and is then weighted by 9 so the
    expectation is right.
    """
    m1 = m2 = 0.0
    weight = 3.0**-N
    for setting in itertools.product("XYZ", repeat=N):
        p = probs(rho, kron_all([ROT[b] for b in setting]))
        f = sum(
            9.0
            * (setting[q] == setting[q + 1])
            * (-1.0) ** (BITS[:, q] + BITS[:, q + 1])
            for q in range(N - 1)
        )
        m1 += weight * float(p @ f)
        m2 += weight * float(p @ f**2)
    return m1, m2 - m1**2


# ----------------------------------------------------------------------------
# Shots needed
# ----------------------------------------------------------------------------
def shots_needed(delta, v_a, v_b, z):
    """Shots per basis so that a gain `delta` is z standard errors.

    The two energies are measured separately, so their variances add:
    sigma^2 = (v_a + v_b) / n and delta / sigma = z gives n = z^2 (v_a + v_b) / delta^2.
    """
    return z**2 * (v_a + v_b) / delta**2


def trial_stats(rng, probs_by_basis, n, trials):
    """Energy and standard error of `trials` simulated experiments (vectorised)."""
    e = np.zeros(trials)
    var = np.zeros(trials)
    for p, _, _ in probs_by_basis.values():
        counts = rng.multinomial(n, p, size=trials)
        mean = counts @ F / n
        # Per-basis sample variance, exactly as Lab 44's estimator forms it.
        v = counts @ F**2 / n - mean**2
        e += mean
        var += v / n
    return e, np.sqrt(var)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def build_states(chain):
    """Noisy density matrices at the Lab 44 strength: depth 0 and the six sets."""
    states = {(0, "ideal"): chain.state([], LAMBDA_FEZ)}
    for key, x in ANGLES.items():
        states[key] = chain.state(x, LAMBDA_FEZ)
    return states


def check_symmetry(r, ctx):
    """Check 1: the exact distributions and the SU(2) equalities."""
    log.info("=== Check 1: exact distributions and the SU(2) symmetry ===")
    chain, states = ctx["chain"], ctx["states"]
    err_e = err_m = err_v = 0.0
    for rho in states.values():
        st = basis_stats(rho)
        err_e = max(
            err_e, abs(sum(m for _, m, _ in st.values()) - chain_energy(chain, rho))
        )
        means = [st[a][1] for a in "XYZ"]
        variances = [st[a][2] for a in "XYZ"]
        err_m = max(err_m, max(means) - min(means))
        err_v = max(err_v, max(variances) - min(variances))
    log.info(
        "max |energy - Tr(rho H)| %.1e; spread of means %.1e, variances %.1e",
        err_e,
        err_m,
        err_v,
    )
    r.append(check(err_e < 1e-9, "X, Y, Z bond sums add up to Tr(rho H) for 7 states"))
    r.append(check(err_m < 1e-9, "<XX> = <YY> = <ZZ>: depolarizing noise keeps SU(2)"))
    r.append(check(err_v < 1e-9, "the three measurement variances are equal too"))
    # (energy, V) for every state; V sets the shot noise of everything below.
    ctx["ev"] = {k: energy_var(rho) for k, rho in states.items()}
    # Cross-check the variance formula through an independent code path: sample
    # counts and give them to Lab 44's own estimator.
    rho = states[(3, "ideal")]
    st = basis_stats(rho)
    n = 40000
    counts = {}
    for a in "XYZ":
        draw = ctx["rng"].multinomial(n, st[a][0])
        # Bit strings in Qiskit order (qubit 0 is the rightmost character).
        counts[a] = {format(i, f"0{N}b"): int(c) for i, c in enumerate(draw) if c}
    e_hat, se_hat = energy_from_counts(counts, np.ones(N))
    e, v = ctx["ev"][(3, "ideal")]
    log.info(
        "depth 3 ideal: sampled E %.4f +/- %.4f; exact %.4f +/- %.4f",
        e_hat,
        se_hat,
        e,
        math.sqrt(v / n),
    )
    r.append(check(abs(e_hat - e) < 4 * se_hat, "Lab 44's estimator is unbiased here"))
    r.append(
        check(
            abs(se_hat / math.sqrt(v / n) - 1) < 0.05,
            "standard error matches sqrt(V/n)",
        )
    )


def chain_energy(chain, rho):
    """Tr(rho H) for a density matrix."""
    return float(np.trace(rho @ chain.h).real)


def check_savers(r, ctx):
    """Check 2: three ways to save shots, tested."""
    log.info("=== Check 2: can fewer shots do the job? ===")
    # Depth 3 ideal angles: the state the decisive run cares about.
    rho = ctx["states"][(3, "ideal")]
    st = basis_stats(rho)
    v = [st[a][2] for a in "XYZ"]
    e, v_sum = ctx["ev"][(3, "ideal")]
    # (a) All 3n shots in Z, energy = 3 * mean(ZZ sum): variance 9 v_Z / (3 n).
    z_only = 9 * v[2] / 3
    log.info("(a) Z only: variance per n %.6f vs three bases %.6f", z_only, v_sum)
    r.append(
        check(abs(z_only - v_sum) < 1e-9 * v_sum, "Z only with the same shots: no gain")
    )
    # (b) Neyman allocation n_a proportional to sigma_a; equal sigmas give 1.
    gain = 3 * sum(v) / sum(math.sqrt(x) for x in v) ** 2
    log.info("(b) optimal allocation gain %.6f", gain)
    r.append(check(abs(gain - 1) < 1e-9, "optimal allocation between X, Y, Z: gain 1"))
    # Symmetry-breaking noise: amplitude damping on every qubit. Real devices
    # have some (T1 decay), so this shows what the Z-only shortcut would cost.
    for g in (0.01, 0.03):
        d = damp(rho, g)
        sd = basis_stats(d)
        ed = sum(m for _, m, _ in sd.values())
        bias = 3 * sd["Z"][1] - ed
        vd = [sd[a][2] for a in "XYZ"]
        gain_d = 3 * sum(vd) / sum(math.sqrt(x) for x in vd) ** 2
        log.info(
            "    damping %.2f: E %.4f, Z-only bias %+.4f, allocation gain %.5f",
            g,
            ed,
            bias,
            gain_d,
        )
        if g == 0.01:
            ctx["bias1"], ctx["gain_d"] = bias, gain_d
    r.append(check(ctx["bias1"] > 0.05, "Z only is biased once the symmetry breaks"))
    r.append(check(ctx["gain_d"] < 1.001, "even then, allocation gains under 0.1 %"))
    # (c) Classical shadows on depth 3 and on the depth-0 singlets.
    ratios = []
    for key in ((3, "ideal"), (0, "ideal")):
        m, var = shadow_stats(ctx["states"][key])
        e_k, v_k = ctx["ev"][key]
        ratio = var / (3 * v_k)  # direct scheme: 3 v_sum per shot (3n shots)
        ratios.append(ratio)
        log.info(
            "(c) shadows, depth %d: mean %.5f (exact %.5f), variance ratio %.2f",
            key[0],
            m,
            e_k,
            ratio,
        )
        if key[0] == 3:
            r.append(check(abs(m - e_k) < 1e-9, "shadow estimator is unbiased"))
    r.append(
        check(
            min(ratios) > 3,
            f"shadows need {min(ratios):.1f}-{max(ratios):.1f}x more shots",
        )
    )


def check_null(r, ctx):
    """Check 3: sign-flipped angles and what the recorded nulls looked like."""
    log.info("=== Check 3: the sign-flipped control and the recorded nulls ===")
    # Time reversal: exp(-i t H) at -t is the complex conjugate, and the real
    # Hamiltonian and the real noise channel give the same energy.
    chain = ctx["chain"]
    worst = 0.0
    for x in ANGLES.values():
        e1 = chain.energy(x, LAMBDA_FEZ)
        e2 = chain.energy([-v for v in x], LAMBDA_FEZ)
        worst = max(worst, abs(e1 - e2))
    log.info("noisy energy of angles x and -x differs by at most %.1e", worst)
    r.append(
        check(worst < 1e-9, "negating every angle leaves the noisy energy unchanged")
    )
    # The recorded depth-1 and depth-2 pairs have predicted gains of ~0: a null.
    # Chi-squared of the measured gaps against the predicted ones (about zero).
    chi = 0.0
    for p in (1, 2):
        (ei, si), (ea, sa) = MEASURED[(p, "ideal")], MEASURED[(p, "aware")]
        pred = chain.energy(ANGLES[(p, "ideal")], LAMBDA_FEZ) - chain.energy(
            ANGLES[(p, "aware")], LAMBDA_FEZ
        )
        z = (ei - ea - pred) / math.hypot(si, sa)
        chi += z * z
        log.info(
            "depth %d: measured gap %+.3f, predicted %+.4f, %+.1f sigma",
            p,
            ei - ea,
            pred,
            z,
        )
    log.info("null chi-squared %.1f for 2 degrees of freedom (95%% limit 5.99)", chi)
    r.append(check(chi < 9.21, f"recorded null pairs: chi2 {chi:.1f} < 9.21 (99 %)"))
    ctx["null_chi2"] = chi


def check_budget(r, ctx):
    """Check 4: shots, the Monte-Carlo test of the formula, QPU seconds."""
    log.info("=== Check 4: the shot budget ===")
    chain, ev = ctx["chain"], ctx["ev"]
    # The measured error bars show how noisy the real state is, which sets its
    # shot variance: V = sigma^2 * n from the recorded run (n = SHOTS per basis).
    v_hw = {k: MEASURED[k][1] ** 2 * SHOTS for k in MEASURED}
    ratio = np.mean([v_hw[k] / ev[k][1] for k in MEASURED])
    log.info("hardware variance / model variance: %.2f (mean of 6 circuits)", ratio)
    r.append(
        check(0.9 < ratio < 1.6, "the hardware shot variance is 1.0-1.6x the model")
    )
    mod, mea, sig = shortfalls(chain, LAMBDA_FEZ)
    _, _, beta, _, _ = fit_lines(mod, mea, sig)
    scale = float(beta[1])
    log.info("gate-noise scale from the recorded fit (Lab 46): %.2f", scale)
    # rows[depth] = (gain, shots at 3 sigma, at 5 sigma, 3 sigma with hw variance)
    rows = {}
    log.info(
        "%-9s %9s %12s %12s %12s", "depth", "gain", "3 sigma", "5 sigma", "(hw var)"
    )
    for p in (1, 2, 3):
        gain = chain.energy(ANGLES[(p, "ideal")], LAMBDA_FEZ) - chain.energy(
            ANGLES[(p, "aware")], LAMBDA_FEZ
        )
        vi, va = ev[(p, "ideal")][1], ev[(p, "aware")][1]
        hi, ha = v_hw[(p, "ideal")], v_hw[(p, "aware")]
        n3 = shots_needed(gain, vi, va, 3)
        n5 = shots_needed(gain, vi, va, 5)
        n3h = shots_needed(gain, hi, ha, 3)
        rows[p] = (gain, n3, n5, n3h)
        log.info("%-9d %9.4f %12.0f %12.0f %12.0f", p, gain, n3, n5, n3h)
    # Depth 3 with only `scale` of the modelled gain reaching the hardware.
    g3, _, _, n3h = rows[3]
    # Lab 44 quoted about 8400 from a simulator error bar at depth 2; the exact
    # depth-3 variance is a little larger, and the hardware's larger still.
    log.info(
        "Lab 44 quoted about 8400; exact depth 3: %.0f (model), %.0f (hardware)",
        rows[3][1],
        n3h,
    )
    # If only `scale` of the modelled gain survives on the machine, more shots.
    n3_scaled = shots_needed(scale * g3, v_hw[(3, "ideal")], v_hw[(3, "aware")], 3)
    log.info(
        "depth 3 at %.2f of the gain, hardware variance: %.0f shots per basis",
        scale,
        n3_scaled,
    )
    r.append(
        check(
            rows[2][1] > 1e6 and rows[1][1] > 1e8,
            "depth 1 and 2 pairs cannot be resolved: they are null controls",
        )
    )
    r.append(
        check(5e3 < rows[3][1] < 5e4, f"depth 3: {rows[3][1]:.0f} shots (3 sigma)")
    )
    # Monte-Carlo power of the formula, on the exact model distributions.
    rng, trials = ctx["rng"], ctx["trials"]
    n = int(math.ceil(rows[3][1]))
    pi = basis_stats(ctx["states"][(3, "ideal")])
    pa = basis_stats(ctx["states"][(3, "aware")])
    ei, si = trial_stats(rng, pi, n, trials)
    ea, sa = trial_stats(rng, pa, n, trials)
    zs = (ei - ea) / np.hypot(si, sa)
    power = float(np.mean(zs > 1.645))
    # The expected power: P(N(3, 1) > 1.645) = Phi(3 - 1.645) = 0.912.
    expect = 0.5 * (1 + math.erf((3 - 1.645) / math.sqrt(2)))
    tol = 4 * math.sqrt(expect * (1 - expect) / trials)
    log.info(
        "n = %d per basis: detection rate %.3f (expected %.3f +/- %.3f)",
        n,
        power,
        expect,
        tol,
    )
    r.append(check(abs(power - expect) < tol, "the shot formula delivers its power"))
    # A null pair has zero true gain, so only chance can produce z > 1.645.
    # Null: the same state against itself (a sign-flipped copy) at that n.
    ei2, si2 = trial_stats(rng, pi, n, trials)
    zn = (ei - ei2) / np.hypot(si, si2)
    fp = float(np.mean(zn > 1.645))
    tol_fp = 4 * math.sqrt(0.05 * 0.95 / trials)
    log.info("null pair: false-positive rate %.3f (nominal 0.050 +/- %.3f)", fp, tol_fp)
    r.append(
        check(abs(fp - 0.05) < tol_fp, "a null pair is flagged at the nominal rate")
    )
    # QPU time at the rate of the recorded run.
    rate = RUNNING_SECONDS / (SHOTS * CIRCUITS)
    n_plan = int(math.ceil(n3_scaled / 1000.0) * 1000)
    # Three angle sets x three bases, plus the two readout calibrations.
    plan_circuits = 3 * 3 + 2  # ideal, noise-trained, sign-flipped x 3 bases + 2 cal
    plan_shots = plan_circuits * n_plan
    seconds = plan_shots * rate
    log.info(
        "recorded rate %.2f ms per shot; plan: %d circuits x %d shots = %d shots, "
        "about %.0f s of running time",
        1000 * rate,
        plan_circuits,
        n_plan,
        plan_shots,
        seconds,
    )
    log.info(
        "plan = depth-3 ideal angles, noise-trained angles and the ideal angles "
        "with every sign flipped (a null with the same predicted energy), plus "
        "calibration; check the account's remaining QPU minutes and the "
        "backend's maximum shots per circuit before submitting"
    )
    r.append(
        check(0.05e-3 < rate < 2e-3, "recorded rate is plausible (0.05-2 ms per shot)")
    )
    r.append(
        check(
            seconds < 300, f"the decisive plan fits in {seconds:.0f} s of running time"
        )
    )


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 47: shot and QPU-time budget")
    ap.add_argument("--trials", type=int, default=400, help="Monte-Carlo trials")
    ap.add_argument("--quick", action="store_true", help="150 trials")
    ap.add_argument("--seed", type=int, default=3, help="random seed")
    args = ap.parse_args(argv)
    if args.trials < 50:
        ap.error(f"--trials must be at least 50. Got {args.trials}.")
    chain = NoisyChain(N)
    ctx = {
        "chain": chain,
        "rng": np.random.default_rng(args.seed),
        "trials": 150 if args.quick else args.trials,
        "states": build_states(chain),
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_symmetry(r, ctx)
    check_savers(r, ctx)
    check_null(r, ctx)
    check_budget(r, ctx)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
