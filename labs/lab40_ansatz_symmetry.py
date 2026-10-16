#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab40_ansatz_symmetry.py

Purpose
-------
Labs 36-39 mitigated noise on the Heisenberg-chain HVA by using its symmetry:
the circuit conserves total spin S, so an error shows up as S > 0. This lab
asks what happens with OTHER ansatz families. Two ways to give a circuit more
parameters per layer, and two very different outcomes:

  * SYMMETRY-ADAPTED: one angle per bond instead of one per layer. Every gate
    is still exp(-i t (XX + YY + ZZ)), so S is conserved for ANY angles. The
    extra freedom is real: 100x lower energy error at the same depth, and
    fewer layers means fewer gates and less noise. All of Labs 36-39 still
    applies unchanged.
  * SYMMETRY-BREAKING: bond gates exp(-i (a XX + b YY + c ZZ)) with a, b, c
    free. The HVA optimum is a stationary point with a positive-definite
    Hessian, so the optimizer gains NOTHING from the extra directions; random
    starts end in worse local minima that leak a little spin. Breaking the
    symmetry costs energy and buys nothing.

If you do end up with a leaky ansatz, verification still works, with three
changes this lab measures: (1) the noise meter is <S^2> MINUS the ideal
circuit's <S^2>, (2) S^2 verification targets the projected ideal state, which
is usually BETTER than the ansatz (projection repairs part of the leakage),
and (3) the <S^2> line must end at the ideal <S^2>, not at zero: ending at zero
misses by kappa x (ideal <S^2>), the leakage itself, whatever the noise rate.
The ideal <S^2> is a classical, noiseless number.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab40_ansatz_symmetry.py
    python3 labs/lab40_ansatz_symmetry.py --rate 2e-3
    python3 labs/lab40_ansatz_symmetry.py --quick

Log files
---------
logs/lab40/lab40_<YYYYmmdd_HHMMSS>.log

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
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import group_terms, sparse_hamiltonian, train_angles
from noiselib import (
    aniso_steps,
    bond_unitary,
    dm_energy_and_s2,
    dm_state,
    free_steps,
    general_dm_state,
    general_energy,
    general_gate,
    general_sv,
    perbond_from_hva,
    perbond_steps,
    rho_projected,
    s2_matrix,
    singlet_projector,
    train_aniso,
    train_steps,
)

log = get_logger("lab40")

# Anisotropies of the symmetry-breaking families: (name, xi, zeta). ZZ keeps S_z
# and breaks S^2; XY breaks both. Two strengths each show the trend.
FAMILIES = (
    ("ZZ 0.3", 0.0, 0.3),
    ("ZZ 0.6", 0.0, 0.6),
    ("XY 0.3", 0.3, 0.0),
    ("XY 0.6", 0.6, 0.0),
)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def ground_energy(n):
    """Exact ground-state energy of the open Heisenberg chain."""
    return float(
        eigsh(sparse_hamiltonian(n), k=1, which="SA", return_eigenvectors=False)[0].real
    )


def sz_squared(n, steps):
    """<S_z^2> of the ideal circuit state: zero if S_z is conserved."""
    psi = general_sv(n, steps).reshape(-1)
    # Bit k of each basis index is site k's spin: 0 is up, 1 is down.
    bits = (np.arange(2**n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    # Site spins are +-1/2, so S_z is (number of ups - number of downs) / 2.
    sz = (n - 2 * bits.sum(axis=1)) / 2
    return float(np.sum(np.abs(psi) ** 2 * sz**2))


def ideal_s2(n, steps):
    """<S^2> of the ideal (noiseless) circuit state."""
    psi = general_sv(n, steps).reshape(-1)
    return float(np.vdot(psi, s2_matrix(n) @ psi).real)


def noisy_pair(n, steps, rate, fold):
    """Exact (energy, <S^2>, density matrix) of the noisy circuit at one fold."""
    # The density matrix is exact, so every number below is free of shot noise.
    rho = general_dm_state(n, steps, rate, fold)
    energy, s2 = dm_energy_and_s2(rho, n)
    return energy, s2, rho


def line_to(e_pair, s_pair, target):
    """Value at <S^2> = target of the line through folds 1 and 3."""
    # Two folds give two (S^2, E) points; the line is read off at the target.
    slope = (e_pair[1] - e_pair[0]) / (s_pair[1] - s_pair[0])
    return float(e_pair[0] + slope * (target - s_pair[0]))


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_consistency(r, ctx):
    """Check 1: the new circuit code reproduces the old one, folds included."""
    log.info("=== Check 1: general gates reproduce the HVA ===")
    n, xa = 6, ctx["hva6"][2]
    # a = b = c is exactly the Heisenberg bond gate of Labs 28-39.
    r.append(
        check(
            np.abs(general_gate(0.37, 0.37, 0.37) - bond_unitary(0.37)).max() < 1e-12,
            "general_gate(t, t, t) = bond_unitary(t)",
        )
    )
    # A random gate shows unitarity is not special to the trained angles.
    rng = np.random.default_rng(ctx["seed"])
    u = general_gate(*rng.normal(0, 0.7, 3))
    r.append(
        check(
            np.abs(u @ u.conj().T - np.eye(4)).max() < 1e-12,
            "a random (a, b, c) gate is unitary",
        )
    )
    # The noisy density matrix must match the established dm_state exactly.
    # Track the worst case over rate and fold so one bad corner cannot hide.
    worst = 0.0
    for rate, fold in ((0.0, 1), (1e-3, 1), (1e-3, 3)):
        new = general_dm_state(n, aniso_steps(xa), rate, fold)
        old = dm_state(n, xa, rate, fold)
        worst = max(worst, float(np.abs(new - old).max()))
    log.info("largest difference from dm_state: %.2e", worst)
    r.append(check(worst < 1e-12, "xi = zeta = 0 reproduces dm_state (folds 1, 3)"))
    # The per-bond family with equal angles per group is the same circuit.
    # Give every bond of a group the group's single HVA angle.
    equal = perbond_from_hva(n, xa)
    diff = np.abs(
        general_dm_state(n, perbond_steps(n, equal), 1e-3) - dm_state(n, xa, 1e-3)
    ).max()
    r.append(check(diff < 1e-12, "per-bond with equal angles is the HVA"))
    # Folding U -> U U^dagger U must leave every ideal state unchanged, whatever
    # the family, or the fold extrapolations of Labs 30-39 would not apply.
    for name, steps in (
        ("leaky", aniso_steps(xa, 0.3, 0.3)),
        ("per-bond", perbond_steps(n, equal * 1.1)),
    ):
        d = np.abs(
            general_dm_state(n, steps, 0.0, 3) - general_dm_state(n, steps, 0.0, 1)
        ).max()
        r.append(check(d < 1e-12, f"{name}: folding leaves the ideal state unchanged"))


def check_symmetry_adapted(r, ctx):
    """Check 2 and 3: one angle per bond conserves S and lowers the error."""
    log.info("=== Check 2: symmetry-adapted, more parameters ===")
    # A separate stream, so the draws here do not move other checks.
    rng = np.random.default_rng(ctx["seed"] + 1)
    # Conservation must hold for ARBITRARY angles, not just trained ones.
    worst_s2 = worst_sz = 0.0
    for _ in range(3):
        steps = perbond_steps(8, rng.normal(0, 0.6, 14))
        worst_s2 = max(worst_s2, abs(ideal_s2(8, steps)))
        worst_sz = max(worst_sz, sz_squared(8, steps))
    log.info(
        "random per-bond angles, n=8: <S^2> %.1e  <S_z^2> %.1e", worst_s2, worst_sz
    )
    r.append(check(worst_s2 < 1e-12, "random per-bond angles: <S^2> = 0 exactly"))
    r.append(check(worst_sz < 1e-12, "random per-bond angles: <S_z^2> = 0 exactly"))
    # Same depth for both families: only the parameter count differs.
    for n, key in ((6, 2), (8, 2)):
        hva, pb = ctx[f"hva_err{n}"][key], ctx[f"pb_err{n}"][key]
        log.info(
            "n=%d, 2 layers: HVA %d params error %.5f | per-bond %d params error %.2e",
            n,
            4,
            hva,
            len(ctx[f"pb{n}"][key]),
            pb,
        )
        r.append(check(pb < 0.02 * hva, f"n={n}: per-bond error < 2% of the HVA error"))
        r.append(
            check(
                abs(ideal_s2(n, perbond_steps(n, ctx[f"pb{n}"][key]))) < 1e-10,
                f"n={n}: the trained per-bond state is still a singlet",
            )
        )
    # Fewer layers beat more layers: per-bond at depth 2 against HVA at 3, 4.
    # Compare with the better of the deeper HVA circuits, the harder baseline.
    deeper = min(ctx["hva_err8"][3], ctx["hva_err8"][4])
    r.append(
        check(
            ctx["pb_err8"][2] < 0.5 * deeper,
            "n=8: per-bond at 2 layers (14 gates) beats the best HVA at 3 or 4 "
            "layers (21 or 28 gates) by 2x",
        )
    )
    log.info("=== Check 3: the payoff under noise (n=8) ===")
    n, rate = 8, ctx["rate"]
    # Per-bond at 2 layers uses 14 bond gates; the HVA at 3 layers uses 21.
    pb_steps = perbond_steps(n, ctx["pb8"][2])
    hva_steps = aniso_steps(ctx["hva8"][3])
    proj = ctx["proj8"]
    # Collect one row of numbers per circuit for the comparisons below.
    out = {}
    for name, steps in (("per-bond L=2", pb_steps), ("HVA L=3", hva_steps)):
        e1, s1, rho = noisy_pair(n, steps, rate, 1)
        e3, s3, _ = noisy_pair(n, steps, rate, 3)
        ev, acc = rho_projected(rho, n, "S2", proj)
        ideal_e, ideal_s = dm_energy_and_s2(general_dm_state(n, steps, 0.0), n)
        gates = sum(len(group_terms(n, par)) for par, _ in steps)
        out[name] = {
            "raw": e1 - ctx["e0_8"],
            "ver": ev - ctx["e0_8"],
            "kappa": (e1 - ideal_e) / (s1 - ideal_s),
            "line": line_to((e1, e3), (s1, s3), ideal_s) - ctx["e0_8"],
            "bias": e1 - ideal_e,
            "ideal": ideal_e - ctx["e0_8"],
            "acc": acc,
        }
        log.info(
            "%-13s error vs exact: raw %+.4f | S^2 verified %+.5f (kept %.3f) | "
            "S^2 line %+.5f | kappa %.3f%s",
            name,
            out[name]["raw"],
            out[name]["ver"],
            acc,
            out[name]["line"],
            out[name]["kappa"],
            f" | {gates} bond gates",
        )
    pb, hv = out["per-bond L=2"], out["HVA L=3"]
    r.append(check(pb["raw"] < hv["raw"], "raw: the shallower per-bond circuit wins"))
    r.append(
        check(
            pb["ver"] < 0.2 * hv["ver"],
            "S^2 verified: per-bond error is under 20% of the deeper HVA's",
        )
    )
    # The line is read at the ideal <S^2>, which is 0 for these symmetric circuits.
    for name, res in out.items():
        r.append(
            check(
                abs(res["line"] - res["ideal"]) < 0.05 * abs(res["bias"]),
                f"{name}: the <S^2> line removes >95% of the bias",
            )
        )
        r.append(
            check(2.1 < res["kappa"] < 2.6, f"{name}: kappa = bias/<S^2> is about 2.4")
        )
    r.append(
        check(
            abs(pb["kappa"] / hv["kappa"] - 1) < 0.1,
            "kappa is the same for both families (within 10%)",
        )
    )


def check_breaking_is_useless(r, ctx):
    """Check 4: the HVA optimum is a strict local minimum in every direction."""
    log.info("=== Check 4: extra symmetry-breaking parameters gain nothing ===")
    n = 6
    x0 = np.repeat(ctx["hva6"][2], 3)
    m = x0.size

    def f(x):
        """Noiseless energy of the free (a, b, c) family at flat angles x."""
        return general_energy(n, free_steps(x))

    # Energy of the symmetric optimum, the reference for every restart.
    e_sym = f(x0)
    # Central-difference step: small for accuracy, large enough that rounding
    # error stays negligible.
    h = 1e-4
    unit = np.eye(m)
    # Central finite differences; the symmetry argument predicts exact zeros.
    grad = np.array(
        [(f(x0 + h * unit[i]) - f(x0 - h * unit[i])) / (2 * h) for i in range(m)]
    )
    log.info(
        "largest gradient component over all %d parameters: %.1e", m, np.abs(grad).max()
    )
    r.append(
        check(
            np.abs(grad).max() < 1e-5,
            "the HVA optimum is stationary in all 12 (a, b, c) directions",
        )
    )
    hess = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            ei, ej = h * unit[i], h * unit[j]
            hess[i, j] = hess[j, i] = (
                f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)
            ) / (4 * h * h)
    eig = np.linalg.eigvalsh(hess)
    log.info("Hessian eigenvalues: %s", np.round(eig, 2))
    r.append(
        check(eig.min() > 0.5, "the Hessian is positive definite: a strict minimum")
    )
    # Random restarts: the extra freedom must not find anything better.
    rng = np.random.default_rng(ctx["seed"] + 2)
    found = []
    for _ in range(ctx["restarts"]):
        res = minimize(
            f,
            x0 + rng.normal(0, 0.4, m),
            method="L-BFGS-B",
            options={"maxiter": 600},
        )
        found.append((res.fun - e_sym, ideal_s2(n, free_steps(res.x))))
    found.sort()
    log.info(
        "restarts (error above the HVA optimum, <S^2>): %s",
        [(round(a, 4), round(b, 4)) for a, b in found],
    )
    r.append(check(found[0][0] > -1e-6, "no random start beats the symmetric optimum"))
    # A leaky local minimum has spin leakage AND a worse energy than the HVA.
    leaky = [a for a, s in found if s > 1e-3 and a > 1e-3]
    r.append(
        check(
            len(leaky) >= 2,
            "several starts end in WORSE local minima that leak spin (<S^2> > 0)",
        )
    )


def leaky_family(n, xa, xi, zeta):
    """Retrained leaky steps and their noiseless statistics."""
    x, _ = train_aniso(n, xa, xi, zeta)
    return aniso_steps(x, xi, zeta)


def check_leaky(r, ctx):
    """Check 5 and 6: an anisotropic ansatz leaks spin; verification adapts."""
    n, rate = 6, ctx["rate"]
    e0, xa = ctx["e0_6"], ctx["hva6"][2]
    proj = ctx["proj6"]
    log.info("=== Check 5: leaky ansatz families (n=%d, 2 layers) ===", n)
    # The symmetric ansatz's own error, which leaky families must exceed.
    base_err = ctx["hva_err6"][2]
    # One dictionary of noiseless statistics per anisotropic family.
    rows = {}
    for name, xi, zeta in FAMILIES:
        steps = leaky_family(n, xa, xi, zeta)
        rho0 = general_dm_state(n, steps, 0.0)
        e_id, s_id = dm_energy_and_s2(rho0, n)
        e_s2, acc2 = rho_projected(rho0, n, "S2", proj)
        e_sz, accz = rho_projected(rho0, n, "Sz")
        rows[name] = dict(
            steps=steps,
            err=e_id - e0,
            s2=s_id,
            sz2=sz_squared(n, steps),
            proj_err=e_s2 - e0,
            acc2=acc2,
            sz_err=e_sz - e0,
            e_id=e_id,
            e_s2=e_s2,
            e_sz=e_sz,
        )
        log.info(
            "%s: error %.4f <S^2> %.4f <S_z^2> %.4f | S^2 projection error %.4f "
            "(kept %.4f) | S_z projection error %.4f",
            name,
            rows[name]["err"],
            s_id,
            rows[name]["sz2"],
            rows[name]["proj_err"],
            acc2,
            rows[name]["sz_err"],
        )
    for fam in ("ZZ", "XY"):
        weak, strong = rows[f"{fam} 0.3"], rows[f"{fam} 0.6"]
        r.append(
            check(
                0 < weak["s2"] < strong["s2"],
                f"{fam}: more anisotropy leaks more spin (<S^2> grows)",
            )
        )
        r.append(
            check(
                base_err < weak["err"] < strong["err"],
                f"{fam}: and costs more energy than the HVA",
            )
        )
    r.append(
        check(
            max(rows["ZZ 0.3"]["sz2"], rows["ZZ 0.6"]["sz2"]) < 1e-12,
            "ZZ anisotropy conserves S_z exactly",
        )
    )
    r.append(
        check(
            min(rows["XY 0.3"]["sz2"], rows["XY 0.6"]["sz2"]) > 1e-3,
            "XY anisotropy breaks S_z too",
        )
    )
    r.append(
        check(
            all(v["proj_err"] < v["err"] for v in rows.values()),
            "projecting onto S = 0 LOWERS the error of every leaky ansatz",
        )
    )
    r.append(
        check(
            all(v["acc2"] > 0.98 for v in rows.values()),
            "while keeping more than 98% of the shots",
        )
    )
    for name in ("ZZ 0.3", "ZZ 0.6"):
        r.append(
            check(
                abs(rows[name]["e_sz"] - rows[name]["e_id"]) < 1e-10,
                f"{name}: S_z projection changes nothing (S_z is already exact)",
            )
        )
    for name in ("XY 0.3", "XY 0.6"):
        r.append(
            check(
                rows[name]["proj_err"] < rows[name]["sz_err"] < rows[name]["err"],
                f"{name}: S_z projection helps, S^2 projection helps more",
            )
        )
    log.info("=== Check 6: noise on the leaky ansatz (rate %.0e) ===", rate)
    for name, row in rows.items():
        # Folds 1 and 3 give the two points of the <S^2> line.
        folds = {f: noisy_pair(n, row["steps"], rate, f) for f in (1, 3)}
        e1, s1, rho1 = folds[1]
        e3, s3, _ = folds[3]
        # The noise meter is the EXCESS spin over what the ideal circuit has.
        kappa1 = (e1 - row["e_id"]) / (s1 - row["s2"])
        kappa3 = (e3 - row["e_id"]) / (s3 - row["s2"])
        e_ver, acc = rho_projected(rho1, n, "S2", proj)
        e_vz, _ = rho_projected(rho1, n, "Sz")
        line_ideal = line_to((e1, e3), (s1, s3), row["s2"])
        line_zero = line_to((e1, e3), (s1, s3), 0.0)
        # Noise-induced energy shift relative to the ideal, un-projected circuit.
        bias = e1 - row["e_id"]
        log.info(
            "%s: bias %+.4f kappa %.3f/%.3f | S^2 verified vs projected ideal %+.5f | "
            "S_z verified vs its ideal %+.4f | line to ideal <S^2> %+.5f, to 0 %+.4f",
            name,
            bias,
            kappa1,
            kappa3,
            e_ver - row["e_s2"],
            e_vz - row["e_sz"],
            line_ideal - row["e_id"],
            line_zero - row["e_id"],
        )
        r.append(
            check(
                2.1 < kappa1 < 2.6 and abs(kappa3 / kappa1 - 1) < 0.02,
                f"{name}: bias = kappa (<S^2> - ideal <S^2>), kappa about 2.4",
            )
        )
        r.append(
            check(
                abs(e_ver - row["e_s2"]) < 4e-4 * (rate / 1e-3) ** 2,
                f"{name}: S^2 verification recovers the projected ideal energy "
                "(residual grows as rate^2)",
            )
        )
        r.append(
            check(
                abs(e_ver - row["e_s2"]) < 0.02 * abs(e_vz - row["e_sz"]),
                f"{name}: and is 50x closer than S_z verification",
            )
        )
        r.append(
            check(
                abs(line_ideal - row["e_id"]) < 0.03 * abs(bias),
                f"{name}: the <S^2> line to the ideal <S^2> removes >97%",
            )
        )
        # Ending at <S^2> = 0 extrapolates past the true endpoint by exactly the
        # leakage, so it misses by kappa x (ideal <S^2>), independent of the rate.
        miss = abs(line_zero - row["e_id"])
        r.append(
            check(
                abs(miss - kappa1 * row["s2"]) < 0.1 * kappa1 * row["s2"]
                and miss > 10 * abs(line_ideal - row["e_id"]),
                f"{name}: a line to <S^2> = 0 misses by kappa x ideal <S^2>",
            )
        )


def check_longer(r, ctx):
    """Check 7: the same story on 8 sites, 3 layers."""
    n, rate, e0 = 8, ctx["rate"], ctx["e0_8"]
    log.info("=== Check 7: a leaky ansatz on n=%d ===", n)
    # Three layers on 8 sites, where the HVA is accurate enough to deform.
    xa = ctx["hva8"][3]
    proj = ctx["proj8"]
    for name, xi, zeta in (("ZZ 0.3", 0.0, 0.3), ("XY 0.3", 0.3, 0.0)):
        steps = leaky_family(n, xa, xi, zeta)
        rho0 = general_dm_state(n, steps, 0.0)
        e_id, s_id = dm_energy_and_s2(rho0, n)
        e_proj, acc = rho_projected(rho0, n, "S2", proj)
        e1, s1, _ = noisy_pair(n, steps, rate, 1)
        e3, s3, _ = noisy_pair(n, steps, rate, 3)
        bias = e1 - e_id
        # Residual of the line ended at the ideal <S^2> (the right endpoint).
        res_ideal = line_to((e1, e3), (s1, s3), s_id) - e_id
        res_zero = line_to((e1, e3), (s1, s3), 0.0) - e_id
        log.info(
            "%s: error %.4f <S^2> %.4f projected error %.4f (kept %.3f) | "
            "bias %+.4f kappa %.3f | line to ideal %+.5f, to 0 %+.4f",
            name,
            e_id - e0,
            s_id,
            e_proj - e0,
            acc,
            bias,
            bias / (s1 - s_id),
            res_ideal,
            res_zero,
        )
        r.append(check(e_proj < e_id, f"{name}: S^2 projection lowers the energy"))
        r.append(
            check(
                2.1 < bias / (s1 - s_id) < 2.6,
                f"{name}: kappa about 2.4 at n={n}",
            )
        )
        r.append(
            check(
                abs(res_ideal) < 0.03 * abs(bias),
                f"{name}: the line to the ideal <S^2> removes >97% at n={n}",
            )
        )
        # The same law: ending at zero misses by kappa x (ideal <S^2>).
        kappa = bias / (s1 - s_id)
        r.append(
            check(
                abs(abs(res_zero) - kappa * s_id) < 0.1 * kappa * s_id
                and abs(res_zero) > 10 * abs(res_ideal),
                f"{name}: a line to 0 misses by kappa x ideal <S^2> at n={n}",
            )
        )


def build_context(seed, rate, quick, starts=12):
    """Train every ansatz once: HVA, per-bond and the reference energies."""
    # Everything the checks share is trained once, here.
    ctx = {
        "seed": seed,
        "rate": rate,
        "restarts": 4 if quick else 6,
        "hva6": {},
        "hva8": {},
        "hva_err6": {},
        "hva_err8": {},
        "pb6": {},
        "pb8": {},
        "pb_err6": {},
        "pb_err8": {},
    }
    # Two sizes: 6 sites for the detailed study, 8 for the longer-chain check.
    for n in (6, 8):
        e0 = ground_energy(n)
        ctx[f"e0_{n}"] = e0
        # The HVA landscape has plateaus (Lab 28): one warm-started run can stop at
        # a poor minimum. For a FAIR baseline keep the best of several runs per
        # depth, so the comparison with richer ansatz families is not a straw man.
        # Best (error, angles) per depth over all runs.
        best = {}
        for offset in range(starts):
            trained = train_angles(
                n, e0, 4, np.random.default_rng(seed + offset), extra_starts=3
            )
            for layers in range(1, 5):
                err, angles = trained[layers - 1]
                if layers not in best or err < best[layers][0]:
                    best[layers] = (err, angles)
        for layers in range(1, 5):
            # Re-polish to full precision so gradients and errors are trustworthy.
            x, e = train_aniso(n, best[layers][1], 0.0, 0.0)
            ctx[f"hva{n}"][layers] = x
            ctx[f"hva_err{n}"][layers] = e - e0
        # Per-bond angles start from the HVA solution (a symmetric manifold).
        for layers in (1, 2, 3):
            x0 = perbond_from_hva(n, ctx[f"hva{n}"][layers])
            x, e = train_steps(n, x0, lambda v, n=n: perbond_steps(n, v))
            ctx[f"pb{n}"][layers] = x
            ctx[f"pb_err{n}"][layers] = e - e0
        ctx[f"proj{n}"] = singlet_projector(n)
    return ctx


def main(argv=None):
    """Run every Lab 40 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 40: ansatz families and symmetry")
    ap.add_argument("--rate", type=float, default=1e-3, help="depolarizing rate")
    # The seed fixes the HVA training and the random restarts.
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fewer random restarts")
    args = ap.parse_args(argv)
    if not 1e-4 <= args.rate <= 3e-3:
        ap.error(f"--rate must be between 1e-4 and 3e-3. Got {args.rate}.")
    ctx = build_context(args.seed, args.rate, args.quick)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_consistency(r, ctx)
    check_symmetry_adapted(r, ctx)
    check_breaking_is_useless(r, ctx)
    check_leaky(r, ctx)
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
