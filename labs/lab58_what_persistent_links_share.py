#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab58_what_persistent_links_share.py

Purpose
-------
Lab 57 found that the excess singlet-preparation loss of a link repeats (Spearman
rho +0.95 over 13 links), so it is a stable property of the link that the
reported calibration does not carry. Lab 56 looked for a calibration number that
ranks the links and found only a weak hint (shorter T2, rho -0.52, p 0.049) on
the first run alone, which is noisy (about 0.02 per link). This lab asks the
question again with a better measurement and two structural features, at NO QPU
time and NO network: it reads the CSV files that Labs 56 and 57 already wrote.

THE TARGET: for each link, the inverse-variance mean of its loss in run 1 and in
the repeat (weights 1/err^2). Only links with both readout attenuations at least
0.95 in BOTH runs (Lab 57 cut, GOOD_ALPHA); this removes the noise-amplified
links. The pooled error is about 0.7 of the single-run error.

THE FEATURES (12): the ten of Lab 56 (cz_err, cz_ns, t1_min, t2_min, f_diff,
ro_mean, sx_mean, spectators, worst_nb_cz and the benchmark implied; cz_ns and
f_diff are constant or missing on these backends and are reported as not
testable) and two structural ones added here:
  gap         |a - b| of the two qubit numbers. On IBM heavy-hex layouts a gap of 1
              is a link inside a row of the chip and a larger gap is a link between
              two rows (a "vertical" coupler).
  qmin        the lower qubit number, a crude position on the chip.

THE RULES, fixed here before the pooled numbers are looked at (Spearman rho against
the pooled loss, two-sided permutation p from 20000 shuffles, Bonferroni over
the 12 features, P_ASSOC = 0.05 / 12):
  ASSOCIATED   |rho| >= 0.70 and p < P_ASSOC
  SUGGESTIVE   |rho| >= 0.50 and p < 0.05
  NONE         otherwise
  SUMMARY      any ASSOCIATED : the stable loss tracks a number we can name;
               only SUGGESTIVE : weak hints only; else : nothing in these 12
               numbers ranks the persistent links.
  REPLICATION  of Lab 56's T2 hint: t2_min on the pooled loss REPLICATES if
               rho <= -0.50 and p < 0.05, otherwise it does NOT replicate.
With n about 13 only |rho| of about 0.7 or more can be seen, so NONE is a weak
statement. Printed with no rule of their own: the pooled mean excess per device
and the mean pooled loss of gap = 1 links against gap > 1 links.

LIMITATIONS: n is small; the calibration features were read at 02:10 to 02:20,
after the first Lab 55 jobs and before the repeat; the target is two measurements
about two hours apart; a correlation is not a cause.

FIRST RUN (2026-10-18 04:48, on the Lab 56 CSV of 02:49 and the Lab 57 CSV of 03:39;
13 links, fez 4, kingston 6, marrakesh 3): no feature ASSOCIATED and none
SUGGESTIVE; summary: nothing in these 12 numbers ranks the persistent links.
  t2_min -0.42 (p 0.15), t1_min -0.26, cz_err -0.25, ro_mean -0.24, sx_mean +0.27,
  spectators -0.04, worst_nb_cz +0.05, implied -0.16 (the calibration's own implied
  loss again does not rank the links), qmin +0.12. cz_ns, f_diff and gap are
  constant or missing (gap: 12 of the 13 links have gap 1).
  Lab 56's T2 hint does NOT replicate (it was -0.52, p 0.049, one of ten tests).
Descriptive (no rule): mean pooled excess fez +0.051 (4), kingston -0.001 (6, one of
them the 14-15 link whose implied loss is 0.148), marrakesh +0.064 (3). The only
link with gap > 1 is marrakesh 43-56, which has the highest pooled loss (+0.154 +/-
0.014), but one link cannot test the vertical-coupler idea; it needs more such
links. READING: the pooled target halves the variance and still no named number
tracks it; with n = 13 only |rho| of about 0.7 or more could be seen.

Modes
-----
  (default)   DRY RUN on synthetic links (no files): tests the statistics.
  --data      Read logs/lab56 and logs/lab57 (the newest CSV of each, or the
              files given by --lab56-csv and --lab57-csv) and analyse. No QPU
              time, no network, no key.

Security
--------
No key, no network and no QPU: the lab reads two local CSV files that hold no
secrets, and writes one CSV and one log file.

Usage
-----
    python3 labs/lab58_what_persistent_links_share.py              # dry run
    python3 labs/lab58_what_persistent_links_share.py --data       # real data

Log files
---------
logs/lab58/lab58_<YYYYmmdd_HHMMSS>.log (plain text)
logs/lab58/pooled_<YYYYmmdd_HHMMSS>.csv (one row per link: pooled loss, features)

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (missing file, package, bad flag).
130 User interrupted execution.
===============================================================================
"""

import argparse
import csv
import glob
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from applog import get_logger

log = get_logger("lab58")

try:
    from lab51_herald_and_isolate import check
    from lab52_gate_count_orientation_idle import summarise
    from lab56_calibration_vs_loss import GOOD_ALPHA, N_PERM, perm_p, spearman
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

FEATURES = (
    ("cz_err", "reported CZ error"),
    ("cz_ns", "CZ duration (ns)"),
    ("t1_min", "shorter T1 (us)"),
    ("t2_min", "shorter T2 (us)"),
    ("f_diff", "|frequency difference| (GHz)"),
    ("ro_mean", "mean readout error"),
    ("sx_mean", "mean sx error"),
    ("spectators", "other qubits coupled to the pair"),
    ("worst_nb_cz", "worst neighbouring CZ error"),
    ("implied", "the calibration's implied loss (benchmark)"),
    ("gap", "|a - b| of the qubit numbers"),
    ("qmin", "lower qubit number"),
)
P_ASSOC = 0.05 / len(FEATURES)  # Bonferroni over the features
RHO_ASSOC, RHO_SUGG, P_SUGG = 0.70, 0.50, 0.05
RHO_REPL, P_REPL = -0.50, 0.05  # replication of the Lab 56 T2 hint
MIN_LINKS = 8


# ----------------------------------------------------------------------------
# Reading and joining the two earlier CSV files
# ----------------------------------------------------------------------------
def newest(pattern):
    """The newest file that matches the glob, or None."""
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def read_csv(path):
    """Rows of a CSV file as dicts."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(text):
    """A float from CSV text; nan when empty or not a number."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def pooled(l1, e1, l2, e2):
    """Inverse-variance mean of two measurements and its error."""
    w1, w2 = 1.0 / (e1 * e1), 1.0 / (e2 * e2)
    return (w1 * l1 + w2 * l2) / (w1 + w2), math.sqrt(1.0 / (w1 + w2))


def join_rows(rows56, rows57, min_alpha=GOOD_ALPHA):
    """Links in both files with good readout in both runs: pooled loss, features."""
    by57 = {(r["device"], r["link"]): r for r in rows57}
    out = []
    for r in rows56:
        q = by57.get((r["device"], r["link"]))
        if q is None:
            continue
        if min(num(q["alpha1"]), num(q["alpha2"])) < min_alpha:
            continue
        loss, err = pooled(
            num(q["loss1"]), num(q["err1"]), num(q["loss2"]), num(q["err2"])
        )
        a, b = (int(v) for v in r["link"].split("-"))
        row = {k: num(r.get(k)) for k, _ in FEATURES if k not in ("gap", "qmin")}
        row.update(
            {
                "device": r["device"],
                "link": r["link"],
                "loss": loss,
                "err": err,
                "implied": num(q["implied"]),
                "gap": float(abs(a - b)),
                "qmin": float(min(a, b)),
            }
        )
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------------
def assoc_verdict(rho, p):
    """The per-feature rule."""
    if not (math.isfinite(rho) and math.isfinite(p)):
        return "NONE"
    if abs(rho) >= RHO_ASSOC and p < P_ASSOC:
        return "ASSOCIATED"
    if abs(rho) >= RHO_SUGG and p < P_SUGG:
        return "SUGGESTIVE"
    return "NONE"


def summary_verdict(verdicts):
    """The overall rule."""
    if "ASSOCIATED" in verdicts:
        return "the stable loss tracks a number we can name (a correlation only)"
    if "SUGGESTIVE" in verdicts:
        return "weak hints only: a larger sample is needed"
    return "nothing in these 12 numbers ranks the persistent links (small n)"


def replication_verdict(rho, p):
    """Does the Lab 56 T2 hint repeat on the pooled loss?"""
    if math.isfinite(rho) and math.isfinite(p) and rho <= RHO_REPL and p < P_REPL:
        return "REPLICATES"
    return "does NOT replicate"


def analyse(rows):
    """[(name, label, rho, p, verdict)] for every feature over the rows."""
    loss = [r["loss"] for r in rows]
    out = []
    for name, label in FEATURES:
        x = [r[name] for r in rows]
        if not all(math.isfinite(v) for v in x) or len(set(x)) < 3:
            out.append((name, label, float("nan"), float("nan"), "NONE"))
            continue
        rho = spearman(x, loss)
        p = perm_p(x, loss, n_perm=N_PERM)
        out.append((name, label, rho, p, assoc_verdict(rho, p)))
    return out


def write_csv(rows, path):
    """Write the rows to a CSV file (no secrets are in them)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def report(rows):
    """Print the pooled table, the feature table and the verdicts."""
    log.info("pooled loss of each link (run 1 and repeat, inverse-variance):")
    log.info("  device        link       gap  implied   pooled loss +/- err")
    for r in sorted(rows, key=lambda x: -x["loss"]):
        log.info(
            "  %-13s %-8s %4d  %7.4f   %+.3f +/- %.3f",
            r["device"],
            r["link"],
            int(r["gap"]),
            r["implied"],
            r["loss"],
            r["err"],
        )
    results = analyse(rows)
    log.info(
        "Spearman rho of each feature against the pooled loss (n = %d):", len(rows)
    )
    for name, label, rho, p, v in results:
        if math.isfinite(rho):
            log.info("  %-12s %-42s rho %+.2f  p %.4f  %s", name, label, rho, p, v)
        else:
            log.info("  %-12s %-42s not testable (missing or constant)", name, label)
    t2 = [x for x in results if x[0] == "t2_min"][0]
    log.info("Lab 56 T2 hint: %s", replication_verdict(t2[2], t2[3]))
    log.info("verdict (summary): %s", summary_verdict([x[4] for x in results]))
    for dev in sorted({r["device"] for r in rows}):
        sub = [r["loss"] - r["implied"] for r in rows if r["device"] == dev]
        log.info(
            "  %-13s %d links, mean pooled excess %+.3f", dev, len(sub), np.mean(sub)
        )
    for label, sel in (("gap = 1", lambda g: g == 1), ("gap > 1", lambda g: g > 1)):
        sub = [r["loss"] for r in rows if sel(r["gap"])]
        if sub:
            log.info(
                "  %s: %d links, mean pooled loss %+.3f", label, len(sub), np.mean(sub)
            )
    return results


# ----------------------------------------------------------------------------
# Dry run on synthetic data
# ----------------------------------------------------------------------------
def synthetic(n, name, effect, seed):
    """Synthetic rows: random features, loss follows one feature plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {k: float(rng.random()) for k, _ in FEATURES}
        row.update({"device": "sim", "link": f"{i}-{i + 1}", "err": 0.015})
        rows.append(row)
    rk = np.argsort(np.argsort([r[name] for r in rows])) / float(n)
    for r, v in zip(rows, rk):
        r["loss"] = effect * float(v) + float(rng.normal(0, 0.01))
    return rows


def check_pure(r):
    """The statistics, the rules and the join on cases with known answers."""
    log.info("=== Check 1: the statistics on cases with known answers ===")
    planted = analyse(synthetic(14, "gap", 0.2, 1))
    hit = [x for x in planted if x[0] == "gap"][0]
    r.append(check(hit[4] == "ASSOCIATED", f"planted gap found: rho {hit[2]:+.2f}"))
    others = [x for x in planted if x[0] != "gap" and x[4] == "ASSOCIATED"]
    r.append(check(len(others) == 0, "no other feature falsely ASSOCIATED"))
    noise = analyse(synthetic(14, "gap", 0.0, 2))
    r.append(
        check(all(x[4] != "ASSOCIATED" for x in noise), "pure noise: none ASSOCIATED")
    )
    r.append(
        check(
            assoc_verdict(0.75, 0.001) == "ASSOCIATED"
            and assoc_verdict(0.75, 0.02) == "SUGGESTIVE"
            and assoc_verdict(0.3, 0.001) == "NONE"
            and assoc_verdict(float("nan"), 0.5) == "NONE",
            "rule: per feature",
        )
    )
    r.append(
        check(
            summary_verdict(["NONE", "ASSOCIATED"]).startswith("the stable")
            and summary_verdict(["SUGGESTIVE"]).startswith("weak")
            and summary_verdict(["NONE"]).startswith("nothing"),
            "rule: summary",
        )
    )
    r.append(
        check(
            replication_verdict(-0.6, 0.03) == "REPLICATES"
            and replication_verdict(-0.4, 0.03).startswith("does NOT")
            and replication_verdict(-0.6, 0.2).startswith("does NOT"),
            "rule: replication of the T2 hint",
        )
    )
    m, e = pooled(0.10, 0.02, 0.06, 0.02)
    r.append(
        check(
            abs(m - 0.08) < 1e-12 and abs(e - 0.02 / math.sqrt(2)) < 1e-12,
            "pooled mean and error",
        )
    )
    rows56 = [
        {"device": "d", "link": "10-23", "cz_err": "0.002", "t2_min": "50"},
        {"device": "d", "link": "4-5", "cz_err": "0.003", "t2_min": "60"},
    ]
    rows57 = [
        {
            "device": "d",
            "link": "10-23",
            "implied": "0.01",
            "loss1": "0.1",
            "err1": "0.02",
            "alpha1": "0.97",
            "loss2": "0.06",
            "err2": "0.02",
            "alpha2": "0.98",
        },
        {
            "device": "d",
            "link": "4-5",
            "implied": "0.01",
            "loss1": "0.1",
            "err1": "0.02",
            "alpha1": "0.97",
            "loss2": "0.06",
            "err2": "0.02",
            "alpha2": "0.80",
        },
    ]
    j = join_rows(rows56, rows57)
    r.append(
        check(
            len(j) == 1 and j[0]["gap"] == 13 and j[0]["qmin"] == 10,
            "join: poor-readout link dropped, gap and qmin computed",
        )
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 58: what persistent links share")
    ap.add_argument("--data", action="store_true", help="analyse the Lab 56/57 CSVs")
    ap.add_argument("--lab56-csv", default=None, help="Lab 56 links CSV")
    ap.add_argument("--lab57-csv", default=None, help="Lab 57 pairs CSV")
    args = ap.parse_args(argv)
    r = []
    check_pure(r)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.data:
        log.info("=== Check 2: report on synthetic links ===")
        rows = synthetic(14, "gap", 0.2, 3)
        report(rows)
        return summarise(r)
    p56 = args.lab56_csv or newest("logs/lab56/links_*.csv")
    p57 = args.lab57_csv or newest("logs/lab57/pairs_*.csv")
    log.info("=== Check 2: read %s and %s ===", p56, p57)
    r.append(check(bool(p56 and p57), "both CSV files found"))
    if not (p56 and p57):
        return 2
    rows = join_rows(read_csv(p56), read_csv(p57))
    r.append(
        check(
            len(rows) >= MIN_LINKS, f"{len(rows)} links with good readout in both runs"
        )
    )
    if not rows:
        return 1
    write_csv(rows, Path("logs/lab58") / f"pooled_{stamp}.csv")
    report(rows)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
