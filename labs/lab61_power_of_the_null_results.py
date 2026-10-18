#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab61_power_of_the_null_results.py

Purpose
-------
Labs 56, 58 and 59 reported NULL results ("nothing in the calibration ranks the
links", "links between chip rows lose no more than row links") and Lab 57 a
positive one ("the excess persists"). A null result is only as strong as the test
behind it. Each lab wrote down how small an effect it could see (Lab 58: a rank
correlation of about 0.7; Lab 59: a difference of about 0.029 at 3 sigma), but
those statements were never themselves tested. This lab does, by Monte Carlo
simulation with NO QPU time, NO network and NO qiskit: it generates links with
known properties, applies the exact rules of Labs 57, 58 and 59 to them, and
measures each rule's FALSE-POSITIVE RATE when there is no effect and its POWER
when there is one.

THE MODEL (the sizes are those of the real data, stated so they can be changed):
  SIG_TRUE   0.05   the spread of the true loss across links (the 13 pooled Lab 58
                    losses have a standard deviation of 0.048).
  SIG_ERR    0.014  Lab 58 error of a pooled loss; 0.02 for a single measurement
                    (Labs 56, 57, 59).
  n          13 links (Lab 58, Lab 57), 15 (Lab 56), 8 + 8 (Lab 59).
  Lab 58 / 56 : one feature has true correlation r with the true loss, 11 others
                are independent of it; the rule is the Lab 58 rule (ASSOCIATED at
                |rho| >= 0.70 and p < 0.05 / 12; SUGGESTIVE at |rho| >= 0.50 and
                p < 0.05; p from the exact permutation distribution).
  Lab 59      : vertical links have a true mean excess D above the row links; each
                link also has its own spread SIG_TRUE (links differ, which the
                Lab 59 error bar sigma_D, built from shot noise only, does not
                include); the rule is the Lab 59 rule.
  Lab 57      : each link has a stable loss part of size SIG_STABLE and a
                measurement error in each run; the rule is the Lab 57 persistence
                rule (PERSISTENT at rho >= 0.70 and p < 0.01).

THE RULES, fixed here before the simulation is run. For each procedure the lab
prints the false-positive rate (no effect) and the power over a grid of effects,
and the smallest effect on the grid with power >= 0.8 (the DETECTABLE EFFECT).
  Lab 58 / 56  ADEQUATE    false ASSOCIATED rate (any of the 12 features, no
                           effect) <= 0.05 and power to flag an r = 0.8 feature
                           as ASSOCIATED >= 0.8;
               UNDERPOWERED  the false rate is fine but the power at r = 0.8 is
                           below 0.8 (a null result is weak even for a strong
                           effect);
               OPTIMISTIC  the false rate exceeds 0.05.
  Lab 59       ADEQUATE    false VERTICAL HIGHER + VERTICAL LOWER rate <= 0.05 and
                           power at D = 0.06 >= 0.8;
               UNDERPOWERED  the false rate is fine but the power at D = 0.06 is
                           below 0.8;
               OPTIMISTIC  the false rate exceeds 0.05 (the error bar is too
                           small because it leaves out the differences between
                           links).
  Lab 57       ADEQUATE    false PERSISTENT rate (SIG_STABLE = 0) <= 0.01 and
                           power at SIG_STABLE = 0.05 >= 0.8; otherwise
                           OPTIMISTIC or UNDERPOWERED as above.

LIMITATIONS: a model, not the hardware (normal true losses, one feature at a
time, independent links); the true spread SIG_TRUE is taken from the data
itself; the Lab 58 features are assumed measured without error.

FIRST RUN (2026-10-18, 4000 data sets per cell, seed 61): Lab 58 and Lab 56
are UNDERPOWERED: false ASSOCIATED rate 0.044 and 0.040, but power to flag a
planted r = 0.8 feature only 0.48 and 0.51 (0.77 and 0.78 at r = 0.9). The
"about 0.7" detectable correlation quoted in Lab 58 was optimistic: a feature must
correlate at about 0.9 with the true loss to be flagged 8 times in 10, so those
null results exclude only very strong associations. Lab 57 is ADEQUATE (false
PERSISTENT 0.004, power 0.85 at a stable SD of 0.05). Lab 59 is OPTIMISTIC by the
pre-written rule: its error bar leaves out the spread between links, so with a
link spread of 0.05 the false HIGHER/LOWER rate is 0.27 (power at D = 0.06 is
0.87). NOT PRE-REGISTERED sensitivity: with link spread 0.02 the false rate is
0.03 and with 0.03 it is 0.10; the spread of the excess (loss minus implied)
across links was not measured here, so which applies is open. The actual Lab 59
verdict was NO DIFFERENCE, which this optimism does not affect; the lab could
detect D of about 0.06, not the 0.029 it quoted.

Modes
-----
  (default)  Run all three analyses (about a minute).
  --sims N   Number of simulated data sets per cell (default 4000).
  --seed S   Random seed (default 61).

Security
--------
No key, no network and no QPU: pure NumPy simulation. Nothing secret exists.

Usage
-----
    python3 labs/lab61_power_of_the_null_results.py

Log files
---------
logs/lab61/lab61_<YYYYmmdd_HHMMSS>.log (plain text)
logs/lab61/power_<YYYYmmdd_HHMMSS>.csv (one row per procedure and effect)

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from applog import get_logger

log = get_logger("lab61")

try:
    from lab51_herald_and_isolate import check
    from lab52_gate_count_orientation_idle import summarise
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

OUT_DIR = Path("logs/lab61")
SIG_TRUE, SIG_ERR, SIG_ERR_ONE = 0.05, 0.014, 0.02
N_FEATURES = 12
RHO_ASSOC, RHO_SUGG, P_SUGG = 0.70, 0.50, 0.05
P_ASSOC = 0.05 / N_FEATURES
RHO_PERSIST, P_PERSIST = 0.70, 0.01
D_MIN, SIGMA_HIGH, SIGMA_NONE = 0.03, 3.0, 2.0
N_PERM_NULL = 200_000
POWER_GOAL = 0.8
R_GRID = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
D_GRID = (0.0, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10)
STABLE_GRID = (0.0, 0.02, 0.03, 0.05, 0.08)


# ----------------------------------------------------------------------------
# Rank statistics in batches
# ----------------------------------------------------------------------------
def ranks_batch(x):
    """Ranks 1..n along the last axis (continuous data, no ties)."""
    return np.argsort(np.argsort(x, axis=-1), axis=-1) + 1.0


def spearman_batch(x, y):
    """Spearman rho of x and y along the last axis."""
    rx, ry = ranks_batch(x), ranks_batch(y)
    n = x.shape[-1]
    d2 = np.sum((rx - ry) ** 2, axis=-1)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1.0))


def null_abs_rho(n, rng, m=N_PERM_NULL):
    """Sorted |rho| of m random permutations: the exact null for n untied points."""
    perms = np.argsort(rng.random((m, n)), axis=1) + 1.0
    base = np.arange(1, n + 1, dtype=float)
    d2 = np.sum((perms - base) ** 2, axis=1)
    return np.sort(np.abs(1.0 - 6.0 * d2 / (n * (n * n - 1.0))))


def p_values(rho, null_sorted):
    """Two-sided permutation p of each rho against the sorted null |rho|."""
    m = len(null_sorted)
    ge = m - np.searchsorted(null_sorted, np.abs(rho) - 1e-12, side="left")
    return (ge + 1.0) / (m + 1.0)


# ----------------------------------------------------------------------------
# The three rules (vectorised)
# ----------------------------------------------------------------------------
def lab58_rule(rho, p):
    """Boolean arrays (associated, suggestive-or-better) for rho and p arrays."""
    assoc = (np.abs(rho) >= RHO_ASSOC) & (p < P_ASSOC)
    sugg = assoc | ((np.abs(rho) >= RHO_SUGG) & (p < P_SUGG))
    return assoc, sugg


def lab58_power(n, sig_err, r_true, sims, rng, null_sorted):
    """(P[planted feature ASSOCIATED], P[planted >= SUGGESTIVE], P[any false ASSOC])."""
    z = rng.normal(size=(sims, n))
    true = SIG_TRUE * z
    obs = true + sig_err * rng.normal(size=(sims, n))
    feat = r_true * z + math.sqrt(max(1.0 - r_true**2, 0.0)) * rng.normal(
        size=(sims, n)
    )
    rho = spearman_batch(feat, obs)
    a, s = lab58_rule(rho, p_values(rho, null_sorted))
    false_any = np.zeros(sims, dtype=bool)
    for _ in range(N_FEATURES - 1):
        other = rng.normal(size=(sims, n))
        ro = spearman_batch(other, obs)
        false_any |= lab58_rule(ro, p_values(ro, null_sorted))[0]
    return float(a.mean()), float(s.mean()), float(false_any.mean())


def lab59_rule(d, sigma):
    """Index of the Lab 59 verdict per simulation: 0 HIGHER, 1 LOWER, 2 NONE, 3 UNCLEAR."""
    out = np.full(d.shape, 3)
    out[np.abs(d) < SIGMA_NONE * sigma] = 2
    out[(d <= -D_MIN) & (d < -SIGMA_HIGH * sigma)] = 1
    out[(d >= D_MIN) & (d > SIGMA_HIGH * sigma)] = 0
    return out


def lab59_power(d_true, n_per, sims, rng, sig_err=SIG_ERR_ONE, sig_link=SIG_TRUE):
    """Fractions of (HIGHER, LOWER, NONE, UNCLEAR) when vertical links lose d_true more."""
    v = (
        d_true
        + sig_link * rng.normal(size=(sims, n_per))
        + sig_err * rng.normal(size=(sims, n_per))
    )
    r = sig_link * rng.normal(size=(sims, n_per)) + sig_err * rng.normal(
        size=(sims, n_per)
    )
    d = v.mean(axis=1) - r.mean(axis=1)
    sigma = math.sqrt(2.0 * n_per) * sig_err / n_per  # shot noise only, as in Lab 59
    verdict = lab59_rule(d, sigma)
    return tuple(float(np.mean(verdict == k)) for k in range(4))


def lab57_power(stable, n, sims, rng, null_sorted, sig_err=SIG_ERR_ONE):
    """P[PERSISTENT] when links have a stable loss part of size stable."""
    base = stable * rng.normal(size=(sims, n)) + SIG_TRUE * 0.0
    x = base + sig_err * rng.normal(size=(sims, n))
    y = base + sig_err * rng.normal(size=(sims, n))
    rho = spearman_batch(x, y)
    p = p_values(rho, null_sorted)
    return float(np.mean((rho >= RHO_PERSIST) & (p < P_PERSIST)))


def detectable(grid, powers, goal=POWER_GOAL):
    """The smallest grid effect whose power reaches the goal, or None."""
    for g, p in zip(grid, powers):
        if g > 0 and p >= goal:
            return g
    return None


def adequacy(false_rate, power_at_goal, false_limit):
    """ADEQUATE / UNDERPOWERED / OPTIMISTIC from the two numbers."""
    if false_rate > false_limit:
        return "OPTIMISTIC"
    if power_at_goal < POWER_GOAL:
        return "UNDERPOWERED"
    return "ADEQUATE"


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def run_all(sims, seed):
    """Run the three analyses; return (rows, summary dict)."""
    rng = np.random.default_rng(seed)
    rows, summary = [], {}
    for lab, n, sig in (("lab58", 13, SIG_ERR), ("lab56", 15, SIG_ERR_ONE)):
        null = null_abs_rho(n, rng)
        res = [lab58_power(n, sig, r, sims, rng, null) for r in R_GRID]
        for r, (pa, ps, pf) in zip(R_GRID, res):
            rows.append(
                {
                    "procedure": lab,
                    "effect": r,
                    "p_flagged": pa,
                    "p_suggestive": ps,
                    "p_false": pf,
                }
            )
            log.info(
                "%s n=%d r=%.1f: P(ASSOCIATED) %.2f  P(>=SUGGESTIVE) %.2f  P(false ASSOCIATED, 11 noise features) %.3f",
                lab,
                n,
                r,
                pa,
                ps,
                pf,
            )
        false_rate = res[0][2]
        p80 = res[R_GRID.index(0.8)][0]
        det = detectable(R_GRID, [x[0] for x in res])
        summary[lab] = (adequacy(false_rate, p80, 0.05), false_rate, p80, det)
    null13 = null_abs_rho(13, rng)
    for k in (8,):
        res = [lab59_power(d, k, sims, rng) for d in D_GRID]
        for d, (h, lo, no, un) in zip(D_GRID, res):
            rows.append(
                {
                    "procedure": "lab59",
                    "effect": d,
                    "p_flagged": h,
                    "p_suggestive": lo,
                    "p_false": no,
                }
            )
            log.info(
                "lab59 8+8 D=%.2f: HIGHER %.2f  LOWER %.2f  NO DIFFERENCE %.2f  UNCLEAR %.2f",
                d,
                h,
                lo,
                no,
                un,
            )
        false_rate = res[0][0] + res[0][1]
        p60 = res[D_GRID.index(0.06)][0]
        det = detectable(D_GRID, [x[0] for x in res])
        summary["lab59"] = (adequacy(false_rate, p60, 0.05), false_rate, p60, det)
    res = [lab57_power(s, 13, sims, rng, null13) for s in STABLE_GRID]
    for s, pw in zip(STABLE_GRID, res):
        rows.append(
            {
                "procedure": "lab57",
                "effect": s,
                "p_flagged": pw,
                "p_suggestive": float("nan"),
                "p_false": float("nan"),
            }
        )
        log.info("lab57 n=13 stable SD=%.2f: P(PERSISTENT) %.3f", s, pw)
    summary["lab57"] = (
        adequacy(res[0], res[STABLE_GRID.index(0.05)], 0.01),
        res[0],
        res[STABLE_GRID.index(0.05)],
        detectable(STABLE_GRID, res),
    )
    return rows, summary


def report(summary):
    """Print the verdict of each procedure."""
    names = {
        "lab58": "Lab 58 (n = 13, features vs pooled loss)",
        "lab56": "Lab 56 (n = 15, features vs single-run loss)",
        "lab59": "Lab 59 (8 + 8 links, vertical vs row)",
        "lab57": "Lab 57 (n = 13, persistence)",
    }
    for k in ("lab58", "lab56", "lab59", "lab57"):
        verdict, fr, pw, det = summary[k]
        log.info(
            "verdict %s: %s (false rate %.3f, power at the reference effect %.2f, detectable effect %s)",
            names[k],
            verdict,
            fr,
            pw,
            "none on the grid" if det is None else f"{det:g}",
        )


def write_csv(rows, path):
    """Write the rows to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_pure(r):
    """The batch statistics and the rule helpers on cases with known answers."""
    log.info("=== Check 1: the statistics on cases with known answers ===")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 13))
    r.append(
        check(
            np.allclose(spearman_batch(x, x**3), 1.0)
            and np.allclose(spearman_batch(x, -x), -1.0),
            "batch Spearman: monotone +1, reversed -1",
        )
    )
    null = null_abs_rho(13, rng)
    p_hi = float(p_values(np.array([0.9]), null)[0])
    p_lo = float(p_values(np.array([0.05]), null)[0])
    r.append(
        check(
            p_hi < 0.001 and p_lo > 0.5,
            f"exact null: p(0.9) {p_hi:.5f}, p(0.05) {p_lo:.2f}",
        )
    )
    thr = float(
        np.interp(
            0.05, [p_values(np.array([v]), null)[0] for v in (0.7, 0.5)], [0.7, 0.5]
        )
    )
    r.append(
        check(
            0.4 < thr < 0.8, f"n = 13: |rho| 0.5 and 0.7 straddle p = 0.05 ({thr:.2f})"
        )
    )
    v = lab59_rule(np.array([0.05, -0.05, 0.005, 0.025]), 0.01)
    r.append(
        check(list(v) == [0, 1, 2, 3], "Lab 59 rule: HIGHER, LOWER, NONE, UNCLEAR")
    )
    r.append(
        check(
            adequacy(0.2, 1.0, 0.05) == "OPTIMISTIC"
            and adequacy(0.01, 0.5, 0.05) == "UNDERPOWERED"
            and adequacy(0.01, 0.9, 0.05) == "ADEQUATE",
            "adequacy labels",
        )
    )
    pa, ps, pf = lab58_power(13, SIG_ERR, 0.99, 800, rng, null)
    pa0, _, pf0 = lab58_power(13, SIG_ERR, 0.0, 800, rng, null)
    r.append(
        check(
            pa > 0.9 and pa0 < 0.1,
            f"Lab 58 rule: planted r = 0.99 flagged {pa:.2f}, r = 0 flagged {pa0:.3f}",
        )
    )


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 61: power of the null results")
    ap.add_argument("--sims", type=int, default=4000, help="data sets per cell")
    ap.add_argument("--seed", type=int, default=61, help="random seed")
    args = ap.parse_args(argv)
    if args.sims < 200:
        ap.error("--sims must be at least 200.")
    r = []
    check_pure(r)
    log.info("=== Check 2: false-positive rates and power ===")
    rows, summary = run_all(args.sims, args.seed)
    r.append(check(len(rows) > 20, f"{len(rows)} cells simulated"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(rows, OUT_DIR / f"power_{stamp}.csv")
    report(summary)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
