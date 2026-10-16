#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab39_spin_extrapolation.py

Purpose
-------
Lab 36 post-selected on total spin S = 0, which on hardware needs extra
measurement circuits. This lab finds the cheap half of that idea. The expected
total spin <S^2> needs NO ancilla: S^2 = (S^x)^2 + (S^y)^2 + (S^z)^2, and each
term is a function of the bit string when every qubit is measured in that
axis' basis. Those are the same three settings (all-X, all-Y, all-Z) that give
the whole Heisenberg energy, so <S^2> comes for free with the energy.

<S^2> is a measured noise meter. The ideal circuit has <S^2> = 0, and any single
Pauli error puts the state at exactly S = 1 (<S^2> = 2; Lab 36). The energy bias
then tracks the spin: bias = kappa <S^2>, with kappa about 2.4 for this circuit
whatever the noise type. So instead of extrapolating to zero noise in an assumed
variable (the fold factor, as in Lab 30), extrapolate to <S^2> = 0 in the
MEASURED one.

Honest limit: kappa is a property of the NOISE as well as the circuit. It is the
same for incoherent noise (depolarizing, dephasing, bit-flip, damping) and for
Pauli-twirled coherent noise, but a FROZEN coherent error drives the state
somewhere else (kappa 1.2-2.2 depending on the angles, drifting with the fold),
so there the S^2 line is only a partial fix, though a far better one than the
fold line.

The lab verifies:
  1. The three-setting measurement reproduces the exact energy and <S^2>.
  2. Single errors give <S^2> = 2; the noiseless circuit gives 0.
  3. <S^2>/2 matches the discarded fraction of Lab 36 (the first-order law).
  4. bias / <S^2> is the same constant for depolarizing, dephasing, bit-flip,
     damping and twirled coherent noise, and across folds (not for frozen
     coherent errors).
  5. Extrapolating in <S^2> removes more than 92% of the bias for every
     incoherent or twirled noise type. For FROZEN coherent errors it removes
     70-95% where the fold extrapolation of Lab 30 makes things worse (their
     bias grows like fold^2).
  6. With finite shots the S^2 extrapolation costs about what the fold one does.
  7. The same law on a longer chain.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab39_spin_extrapolation.py
    python3 labs/lab39_spin_extrapolation.py --shots 50000
    python3 labs/lab39_spin_extrapolation.py --quick

Log files
---------
logs/lab39/lab39_<YYYYmmdd_HHMMSS>.log

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
    basis_shot_values,
    dm_basis_probs,
    dm_compiled_state,
    dm_damping_state,
    dm_energy_and_s2,
    dm_expectation,
    dm_s2_verified,
    dm_state,
    dm_with_errors,
    richardson_weights,
    s2_matrix,
    singlet_projector,
    touched_qubits,
)

log = get_logger("lab39")

FOLDS = (1, 3, 5)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def noise_models(n, params):
    """Return {name: function(fold) -> density matrix} for every noise type."""
    # One common rate, so the noise types are compared like for like.
    rate = 1e-3
    return {
        "depolarizing": lambda f: dm_state(n, params, rate, f),
        "dephasing (Z)": lambda f: dm_state(n, params, rate, f, weights=(0, 0, 1)),
        "bit-flip (X,Y)": lambda f: dm_state(n, params, rate, f, weights=(1, 1, 0)),
        "amplitude damping": lambda f: dm_damping_state(n, params, rate, f),
        "frozen coherent": lambda f: dm_compiled_state(
            n, params, 0.02, 0.02, "none", f
        ),
        "twirled coherent": lambda f: dm_compiled_state(
            n, params, 0.02, 0.02, "full", f
        ),
    }


def table(n, params, clean, models, folds=FOLDS):
    """Exact (bias, <S^2>) at every fold for every noise model."""
    # bias and <S^2> come from the SAME density matrix at each fold.
    out = {}
    for name, make in models.items():
        pairs = [dm_energy_and_s2(make(f), n) for f in folds]
        out[name] = (
            np.array([e - clean for e, _ in pairs]),
            np.array([s for _, s in pairs]),
        )
        log.info(
            "%-18s <S^2> %s  bias %s  bias/<S^2> %s",
            name,
            np.round(out[name][1], 4),
            np.round(out[name][0], 5),
            np.round(out[name][0] / out[name][1], 3),
        )
    return out


def s2_line(bias, s2):
    """Bias extrapolated to <S^2> = 0 with a line through folds 1 and 3."""
    # Two points define the line; its value at <S^2> = 0 is the mitigated bias.
    slope = (bias[1] - bias[0]) / (s2[1] - s2[0])
    return float(bias[0] - slope * s2[0])


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_measurement(r, ctx):
    """Check 1 and 2: <S^2> from three settings; single errors; noiseless."""
    # Three product-basis settings give the whole energy AND the whole <S^2>.
    # Nothing here needs an ancilla qubit or a controlled gate.
    log.info("=== Check 1: the three measurement settings ===")
    n, params = 6, ctx["params6"]
    # A strong rate, so that rounding cannot hide an error in the bookkeeping.
    rho = dm_state(n, params, 3e-3)
    energy, s2 = dm_energy_and_s2(rho, n)
    exact_s2 = float(np.trace(s2_matrix(n) @ rho.reshape(2**n, 2**n)).real)
    r.append(
        check(
            abs(energy - dm_expectation(rho, n)) < 1e-10, "energy from three settings"
        )
    )
    r.append(
        check(abs(s2 - exact_s2) < 1e-10, "<S^2> from three settings = Tr(S^2 rho)")
    )
    # Each setting is a probability distribution over bit strings.
    probs = dm_basis_probs(rho, n)
    r.append(
        check(
            all(abs(p.sum() - 1) < 1e-12 for p in probs.values()),
            "probabilities sum to one",
        )
    )
    # With no noise the circuit must stay at S = 0 exactly.
    clean_rho = dm_state(n, params, 0.0)
    r.append(
        check(abs(dm_energy_and_s2(clean_rho, n)[1]) < 1e-10, "noiseless: <S^2> = 0")
    )
    energy_vals, s2_vals = basis_shot_values(n)
    r.append(
        check(
            len(energy_vals) == len(s2_vals) == 2**n,
            "per-outcome tables cover all bit strings",
        )
    )
    log.info("=== Check 2: a single error is exactly S = 1 ===")
    # Insert each Pauli at each noisy location, one at a time (as in Lab 36).
    worst = 0.0
    count = 0
    for step, (parity, _) in enumerate(layer_plan(params)):
        for q in touched_qubits(n, parity):
            for name in "XYZ":
                state = dm_with_errors(n, params, {(step, q): name})
                worst = max(worst, abs(dm_energy_and_s2(state, n)[1] - 2.0))
                count += 1
    log.info("single errors tested: %d; largest |<S^2> - 2| = %.1e", count, worst)
    r.append(check(worst < 1e-10, "every single Pauli error gives <S^2> = 2"))


def check_first_order(r, ctx):
    """Check 3: <S^2>/2 against the discarded fraction of Lab 36."""
    # Lab 36 discarded the runs with S >= 1; here every run is kept and the
    # same information is read out as an average instead.
    log.info("=== Check 3: <S^2>/2 and the discarded fraction ===")
    n, params = 6, ctx["params6"]
    # Lab 36's exact singlet projector gives the discarded fraction to compare with.
    proj = singlet_projector(n)
    for p in (3e-4, 1e-3, 3e-3):
        rho = dm_state(n, params, p)
        s2 = dm_energy_and_s2(rho, n)[1]
        discarded = 1 - dm_s2_verified(n, params, p, proj=proj)[1]
        log.info("p=%.0e: <S^2>/2 = %.5f, discarded = %.5f", p, s2 / 2, discarded)
        # Spin s costs s(s+1) >= 2, so <S^2>/2 can only overshoot the discard.
        r.append(check(s2 / 2 >= discarded - 1e-12, f"p={p:.0e}: <S^2>/2 >= discarded"))
        r.append(check(s2 / 2 < 1.1 * discarded, f"p={p:.0e}: and within 10% of it"))


def check_proportionality(r, ctx):
    """Check 4: bias = kappa <S^2> for every incoherent noise type."""
    log.info("=== Check 4: bias against <S^2> ===")
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    # The table is reused by the extrapolation check below.
    ctx["table6"] = table(n, params, clean, noise_models(n, params))
    kappas = {}
    for name, (bias, s2) in ctx["table6"].items():
        # kappa = bias / <S^2>: energy cost per unit of total spin.
        ratio = bias / s2
        kappas[name] = float(ratio[0])
        if name != "frozen coherent":
            r.append(
                check(np.ptp(ratio) < 0.04 * ratio[0], f"{name}: constant over folds")
            )
    # Frozen coherent errors are left out: their kappa is not universal.
    inco = {k: v for k, v in kappas.items() if k != "frozen coherent"}
    mean = float(np.mean(list(inco.values())))
    log.info("kappa by noise type: %s", {k: round(v, 3) for k, v in kappas.items()})
    r.append(check(2.0 < mean < 3.0, f"kappa is about 2.4 (mean {mean:.2f})"))
    spread = max(abs(v - mean) for v in inco.values())
    r.append(
        check(spread < 0.1 * mean, "and the same for Pauli, damping and twirled noise")
    )
    # A frozen coherent error steers the state elsewhere, so its kappa is its own.
    log.info("frozen coherent kappa %.2f (not universal)", kappas["frozen coherent"])


def check_extrapolation(r, ctx):
    """Check 5: extrapolating in <S^2> against extrapolating in the fold."""
    # Same two exact data points (folds 1 and 3) feed both extrapolations.
    log.info("=== Check 5: S^2 extrapolation against fold extrapolation ===")
    # Lagrange weights for the usual straight-line fold extrapolation.
    w2 = richardson_weights(FOLDS[:2])
    for name, (bias, s2) in ctx["table6"].items():
        fold_line = float(w2 @ bias[:2])
        s2_res = s2_line(bias, s2)
        log.info(
            "%-18s single %+.2e | fold line %+.2e | S^2 line %+.2e",
            name,
            bias[0],
            fold_line,
            s2_res,
        )
        if name == "frozen coherent":
            r.append(
                check(
                    abs(s2_res) < 0.3 * abs(bias[0]), f"{name}: S^2 line removes >70%"
                )
            )
            r.append(
                check(abs(fold_line) > abs(bias[0]), f"{name}: the fold line is WORSE")
            )
            r.append(
                check(abs(s2_res) < 0.2 * abs(fold_line), f"{name}: S^2 line 5x closer")
            )
        else:
            r.append(
                check(
                    abs(s2_res) < 0.08 * abs(bias[0]), f"{name}: S^2 line removes >92%"
                )
            )


def check_shots(r, ctx):
    """Check 6: finite shots on the same three settings."""
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    shots, reps = ctx["shots"], ctx["reps"]
    log.info("=== Check 6: %d shots per setting, %d repetitions ===", shots, reps)
    # Fresh stream for the shot noise, separate from the other checks.
    rng = np.random.default_rng(ctx["seed"] + 6)
    energy_vals, s2_vals = basis_shot_values(n)
    # Exact outcome probabilities of the three settings at folds 1 and 3.
    states = [dm_basis_probs(dm_state(n, params, 1e-3, f), n) for f in (1, 3)]

    # Sampling the three settings is all an experiment would have to do.
    def measure(probs):
        """One experiment: energy and <S^2> from multinomial counts."""
        energy = spin = 0.0
        for axis in "XYZ":
            freq = rng.multinomial(shots, probs[axis]) / shots
            energy += freq @ energy_vals
            spin += freq @ s2_vals
        return energy, spin

    # Each repetition is one complete experiment at both folds.
    rows = []
    for _ in range(reps):
        (e1, s1), (e3, s3) = measure(states[0]), measure(states[1])
        # The slope of energy against measured spin replaces the assumed fold law.
        kappa = (e3 - e1) / (s3 - s1)
        rows.append((e1, 1.5 * e1 - 0.5 * e3, e1 - kappa * s1))
    rows = np.array(rows) - clean
    # Mean over repetitions estimates the bias; the spread is the shot noise.
    mean, std = rows.mean(axis=0), rows.std(axis=0, ddof=1)
    se = std / np.sqrt(reps)
    # The standard error of the mean tells whether a bias is real or noise.
    for label, m, s, e in zip(("raw", "fold line", "S^2 line"), mean, std, se):
        log.info("%-10s bias %+.4f +- %.4f (spread %.4f)", label, m, e, s)
    r.append(
        check(
            abs(mean[2]) < 4 * se[2] + 0.02 * mean[0],
            "S^2 line is unbiased to within noise",
        )
    )
    r.append(check(abs(mean[1]) < 4 * se[1] + 0.02 * mean[0], "so is the fold line"))
    r.append(
        check(
            0.5 < std[2] / std[1] < 2.0, "and the two cost about the same statistically"
        )
    )
    r.append(
        check(std[2] > 1.2 * std[0], "both inflate the error bar over the raw energy")
    )


def check_longer(r, ctx):
    """Check 7: the same law on a longer chain."""
    n = ctx["n_long"]
    log.info("=== Check 7: n=%d ===", n)
    params = ctx["params8"]
    # Error-free energy of this chain from the matrix product state.
    clean = run_hva(n, params, 64).energy()
    models = {
        k: v
        for k, v in noise_models(n, params).items()
        if k in ("depolarizing", "amplitude damping", "frozen coherent")
    }
    # Two folds suffice for a straight line; the density matrix stays exact.
    tab = table(n, params, clean, models, folds=(1, 3))
    w2 = richardson_weights((1, 3))
    for name, (bias, s2) in tab.items():
        res = s2_line(bias, s2)
        fold_line = float(w2 @ bias)
        if name == "frozen coherent":
            r.append(
                check(abs(res) < 0.3 * abs(bias[0]), f"{name}: S^2 line removes >70%")
            )
            r.append(check(abs(fold_line) > abs(bias[0]), "and the fold line is worse"))
            r.append(
                check(abs(res) < 0.2 * abs(fold_line), "the S^2 line is 5x closer")
            )
        else:
            r.append(
                check(
                    abs(res) < 0.08 * abs(bias[0]),
                    f"{name}: S^2 line removes >92% at n={n}",
                )
            )
    ratios = {k: float(b[0] / s[0]) for k, (b, s) in tab.items()}
    log.info("bias/<S^2> at n=%d: %s", n, {k: round(v, 3) for k, v in ratios.items()})
    inco = [v for k, v in ratios.items() if k != "frozen coherent"]
    r.append(
        check(max(inco) / min(inco) < 1.1, "kappa agrees for depolarizing and damping")
    )


def main(argv=None):
    """Run every Lab 39 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 39: spin extrapolation")
    ap.add_argument("--shots", type=int, default=20000, help="per setting")
    # The seed fixes the trained angles and the shot noise.
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.shots < 1000:
        ap.error(f"--shots must be at least 1000. Got {args.shots}.")
    quick = args.quick
    # Quick mode keeps the exact checks and trims the repetitions.
    ctx = {
        "seed": args.seed,
        "shots": args.shots,
        "reps": 100 if quick else 300,
        "n_long": 8,
    }
    # Angles are trained on small chains, exactly as in Labs 28-38.
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
    # Every exact bias is measured against this error-free energy.
    ctx["clean6"] = run_hva(6, ctx["params6"], 64).energy()
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_measurement(r, ctx)
    check_first_order(r, ctx)
    check_proportionality(r, ctx)
    check_extrapolation(r, ctx)
    check_shots(r, ctx)
    check_longer(r, ctx)
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
