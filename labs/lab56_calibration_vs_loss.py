#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab56_calibration_vs_loss.py

Purpose
-------
Lab 55 found singlet-preparation loss above what the calibration implies on
well-read links of three backends (ibm_fez, ibm_kingston, ibm_marrakesh), but not
on every link: of the 15 links with good readout, 6 lose 3 sigma or more above
their implied loss, 8 are within 3 sigma, and 1 loses less. The reported CZ error
does not tell which. This lab asks whether ANYTHING ELSE in the backend's own
calibration does. It costs NO QPU time: it re-fetches the three finished Lab 55
jobs (their counts), redoes the Lab 55 loss analysis, reads the same backends'
calibration, and correlates the MEASURED LOSS of each link with ten numbers: nine
calibration features and, as the benchmark, the loss the calibration itself
implies (below). The excess (measured minus implied) is printed for reference.

DATA: the job files logs/lab55/job_<backend>_<id>.json written by Lab 55 (or
Lab 55 --retrieve ... --meta-file); each lists the job id, backend and the eight
links with the CZ error seen when the links were chosen. 24 links in all.

FEATURES (per link a-b, from the backend's calibration; "min" and "mean" are over
the two qubits):
  cz_err      the reported CZ error stored when the link was chosen
  cz_ns       CZ gate duration (ns)
  t1_min      shorter T1 (us)
  t2_min      shorter T2 (us)
  f_diff      |frequency a - frequency b| (GHz), when the backend reports it
  ro_mean     mean reported readout error
  sx_mean     mean reported sx error
  spectators  number of other qubits coupled to a or b
  worst_nb_cz the largest reported CZ error among the other links that touch a or b
  implied     the loss the calibration itself implies (the Lab 55 formula, built
              from cz_err and sx_mean): the benchmark, to see how well the standard
              calibration ranks the links against every other number

THE SET: links whose two readout attenuations are both at least 0.95, taken from
each link's own calibration circuits (the Lab 55 post-hoc cut; the cut uses no
gate-loss data). Links with worse readout have a correction that amplifies noise
(Lab 55 saw losses of -0.37 there) and would only add scatter.

THE RULES, fixed here before the first run (Spearman rank correlation rho of each
feature with the MEASURED loss over the set, n about 15; two-sided permutation
p-value from 20000 shuffles; 10 features, so the Bonferroni level is 0.05 / 10 =
0.005; the measured loss, not the excess, is used because the excess subtracts
the implied loss and would correlate with cz_err and sx_mean by construction):
  ASSOCIATED  |rho| >= 0.70 and p < 0.005 : that feature goes with the measured
              loss.
  SUGGESTIVE  |rho| >= 0.50 and p < 0.05 : worth a larger sample, not a finding.
  NONE        otherwise.
  SUMMARY     at least one ASSOCIATED feature : the calibration holds a predictor
              of the bad links (a correlation, not a cause). If ONLY `implied` is
              ASSOCIATED, the standard calibration does rank the links and the gap
              is an offset or a scale, not a missing predictor.
              No ASSOCIATED but some SUGGESTIVE : weak hints only.
              All NONE : nothing in these calibration numbers predicts the bad
              links, with the limit that n = 15 can only see |rho| of about 0.7
              or more, so "none" does not mean "no association".
  DEVICE      the mean excess per backend is printed beside the table, because a
              feature that differs between backends can mimic a device effect.

FIRST LIVE RUN (2026-10-18 02:49; the 24 links of Lab 55 joined to calibrations read
at 02:10 (ibm_fez), 02:17 (ibm_kingston) and 02:20 (ibm_marrakesh), i.e. after the
jobs ran at 00:34 to 01:19; no CZ error had moved by more than 5%). The set is
15 links (fez 5, kingston 6, marrakesh 4; mean excess +0.053, +0.007, +0.088).
By the rules above: no feature ASSOCIATED, one SUGGESTIVE.
  t2_min       rho -0.52  p 0.0485  SUGGESTIVE (shorter T2, more loss)
  sx_mean      rho +0.52  p 0.0503  NONE (misses the SUGGESTIVE line by 0.0003)
  t1_min       rho -0.16  p 0.58
  cz_err       rho -0.02  p 0.95
  ro_mean      rho -0.03  p 0.93
  spectators   rho -0.03  p 0.91
  worst_nb_cz  rho +0.07  p 0.79
  implied      rho +0.08  p 0.78   (the benchmark)
  cz_ns, f_diff   not testable: the backends report one CZ duration for these links
                  and no qubit frequency in their target.
SUMMARY: weak hints only. READING: the calibration's own implied loss does not rank
these 15 links at all (rho +0.08), and neither does the reported CZ error
(rho -0.02), so the extra loss is not a rescaling of the numbers the backend
reports. The one hint is that links with the shorter T2 (and, just short of the
line, the larger sx error) lose more, which would point at qubit quality and not
gate quality. It is weak: with ten features tested, one p of 0.0485 is about what
chance gives, T2 and sx error are probably correlated with each other, and the
T2 was read after the runs. With n = 15 only a strong association could be seen.
Not tested here: whether the HIGH links are the same links when measured again.

LIMITATIONS: n is about 15 and the features are correlated with each other;
the calibration is read when this lab runs, not when the jobs ran (the lab prints
the calibration time and flags a CZ error that has changed since the links were
chosen); the losses carry the Lab 55 errors (about 0.02 per link), which blur
the ranks; a correlation is not a cause.

Modes
-----
  (default)  DRY RUN on a simulated ibm_fez with a planted dependence (no account,
             no network); it tests the statistics and the report.
  --live     Connect to IBM (read-only: finished jobs and calibration; no QPU
             time) and analyse the job files.

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is no --token
flag because command-line arguments leak into shell history. The key is never
logged, never written to a file, and is scrubbed from error text. The optional
instance CRN comes from IBM_QUANTUM_INSTANCE. Nothing secret touches disk.

Usage
-----
    python3 labs/lab56_calibration_vs_loss.py                       # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN            # hidden
    python3 labs/lab56_calibration_vs_loss.py --live --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab56/lab56_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab56/links_<YYYYmmdd_HHMMSS>.csv (one row per link: features, loss, class)

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (missing key, package, bad flag).
130 User interrupted execution.
===============================================================================
"""

import argparse
import csv
import glob
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from applog import get_logger

log = get_logger("lab56")

# Everything shared is imported so each piece of logic exists once.
try:
    from lab17_mirror_benchmark import connect, scrub, wait_for
    from lab50_calibration_snapshot import qubit_row
    from lab51_herald_and_isolate import check
    from lab52_gate_count_orientation_idle import summarise
    from lab55_other_pairs_other_device import (
        build_jobs,
        classify,
        eligible_links,
        implied_link_loss,
        link_detail,
        select_links,
    )
    from qiskit_ibm_runtime.fake_provider import FakeFez
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

META_GLOB = "logs/lab55/job_*.json"
GOOD_ALPHA = 0.95  # both readout attenuations at least this (Lab 55 post-hoc cut)
N_PERM = 20000
RHO_ASSOC, RHO_SUGG = 0.70, 0.50
P_SUGG = 0.05
# (name, label) of the ten features, in the order they are tested.
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
)
P_ASSOC = 0.05 / len(FEATURES)  # Bonferroni over the features


# ----------------------------------------------------------------------------
# Statistics (rank correlation with a permutation p-value)
# ----------------------------------------------------------------------------
def ranks(x):
    """Ranks 1..n of x, ties given their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(x, y):
    """Spearman rank correlation of two equal-length samples."""
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry) / den if den > 0 else float("nan")


def perm_p(x, y, n_perm=N_PERM, seed=0):
    """Two-sided permutation p-value of the Spearman correlation of x and y."""
    rho = spearman(x, y)
    if not math.isfinite(rho):
        return float("nan")
    rng = np.random.default_rng(seed)
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    # Shuffle the ranks of y n_perm times at once (rows are permutations).
    idx = np.argsort(rng.random((n_perm, len(ry))), axis=1)
    rho_perm = (ry[idx] @ rx) / den
    hits = int(np.sum(np.abs(rho_perm) >= abs(rho) - 1e-12))
    return (hits + 1) / (n_perm + 1)


def assoc_verdict(rho, p):
    """The pre-registered per-feature rule."""
    if not (math.isfinite(rho) and math.isfinite(p)):
        return "NONE"
    if abs(rho) >= RHO_ASSOC and p < P_ASSOC:
        return "ASSOCIATED"
    if abs(rho) >= RHO_SUGG and p < P_SUGG:
        return "SUGGESTIVE"
    return "NONE"


def summary_verdict(verdicts):
    """The pre-registered overall rule."""
    if "ASSOCIATED" in verdicts:
        return (
            "the calibration holds a predictor of the bad links "
            "(a correlation, not a cause)"
        )
    if "SUGGESTIVE" in verdicts:
        return "weak hints only: a larger sample is needed"
    return (
        "nothing in these calibration numbers predicts the bad links "
        "(n is small: only |rho| of about 0.7 or more could be seen)"
    )


# ----------------------------------------------------------------------------
# Calibration features of one link
# ----------------------------------------------------------------------------
def coupling_map(target):
    """{(a, b): CZ error} with a < b, and neighbour sets, from a Qiskit target."""
    errs, adj = {}, {}
    for (a, b), props in target["cz"].items():
        if props is None or props.error is None:
            continue
        key = (min(a, b), max(a, b))
        errs[key] = max(errs.get(key, 0.0), float(props.error))
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return errs, adj


def frequency(target, q):
    """Qubit frequency in GHz, or NaN when the backend does not report it."""
    qp = target.qubit_properties[q] if target.qubit_properties else None
    f = getattr(qp, "frequency", None)
    return float(f) / 1e9 if f else float("nan")


def link_features(target, a, b, cz_err, errs, adj):
    """The nine calibration features of link a-b (see the docstring)."""
    durs = []
    for pair in ((a, b), (b, a)):
        props = target["cz"].get(pair)
        if props is not None and props.duration is not None:
            durs.append(float(props.duration) * 1e9)
    t1a, t2a, sxa, roa = qubit_row(target, a)
    t1b, t2b, sxb, rob = qubit_row(target, b)
    key = (min(a, b), max(a, b))
    others = [e for k, e in errs.items() if k != key and (a in k or b in k)]
    return {
        "cz_err": float(cz_err),
        "cz_ns": max(durs) if durs else float("nan"),
        "t1_min": min(t1a, t1b),
        "t2_min": min(t2a, t2b),
        "f_diff": abs(frequency(target, a) - frequency(target, b)),
        "ro_mean": (roa + rob) / 2.0,
        "sx_mean": (sxa + sxb) / 2.0,
        "spectators": float(len((adj[a] | adj[b]) - {a, b})),
        "worst_nb_cz": max(others) if others else float("nan"),
    }


def implied_for(target, a, b, cz_err):
    """The calibration's implied singlet loss for link a-b (Lab 55 formula)."""
    sx = target["sx"]
    return implied_link_loss(cz_err, (sx[(a,)].error, sx[(b,)].error))


# ----------------------------------------------------------------------------
# Rows: one per link
# ----------------------------------------------------------------------------
def make_row(device, a, b, cz_err, target, errs, adj, detail):
    """One row: identity, measured loss, implied loss, class, readout, features."""
    imp = implied_for(target, a, b, cz_err)
    loss, err = detail["loss"], detail["err"]
    row = {
        "device": device,
        "link": f"{a}-{b}",
        "loss": loss,
        "err": err,
        "implied": imp,
        "excess": loss - imp,
        "class": classify(loss, err, imp),
        "alpha_min": float(np.min(detail["ro"][0])),
    }
    row.update(link_features(target, a, b, cz_err, errs, adj))
    return row


def good_rows(rows, min_alpha=GOOD_ALPHA):
    """The links whose two readout attenuations are both at least min_alpha."""
    return [r for r in rows if r["alpha_min"] >= min_alpha]


def analyse_rows(rows):
    """[(name, label, rho, p, verdict)] for every feature, over the given rows."""
    loss = [r["loss"] for r in rows]
    out = []
    for name, label in FEATURES:
        x = [r[name] for r in rows]
        if not all(math.isfinite(v) for v in x) or len(set(x)) < 3:
            out.append((name, label, float("nan"), float("nan"), "NONE"))
            continue
        rho = spearman(x, loss)
        p = perm_p(x, loss)
        out.append((name, label, rho, p, assoc_verdict(rho, p)))
    return out


def write_csv(rows, path):
    """Write the rows to a CSV file (no secrets are in them)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_metas(pattern):
    """Read every Lab 55 job file that matches the glob; skip incomplete ones."""
    metas = []
    for path in sorted(glob.glob(pattern)):
        try:
            meta = json.loads(Path(path).read_text())
            if all(k in meta for k in ("job_id", "backend", "links")):
                metas.append(meta)
        except (OSError, ValueError):
            log.warn_("skipping unreadable job file %s", path)
    return metas


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def report(rows, dry):
    """Print the link table, the feature table and the verdicts."""
    if dry:
        log.info(
            "(dry run: the numbers below describe a simulator with a PLANTED effect)"
        )
    log.info("all links (excess = measured loss - implied loss):")
    log.info(
        "  device        link       alpha_min  loss +/- err      implied  excess   class"
    )
    for r in rows:
        log.info(
            "  %-12s %-9s  %.3f   %+.3f +/- %.3f   %.4f  %+.3f   %s",
            r["device"],
            r["link"],
            r["alpha_min"],
            r["loss"],
            r["err"],
            r["implied"],
            r["excess"],
            r["class"],
        )
    sel = good_rows(rows)
    log.info(
        "set: %d links with both readout attenuations >= %.2f", len(sel), GOOD_ALPHA
    )
    for dev in sorted({r["device"] for r in sel}):
        ex = [r["excess"] for r in sel if r["device"] == dev]
        log.info(
            "  %-12s %2d links, mean excess %+.3f", dev, len(ex), float(np.mean(ex))
        )
    if len(sel) < 8:
        log.warn_("fewer than 8 links in the set: no correlation is meaningful")
        return []
    results = analyse_rows(sel)
    log.info("feature      rho     p        verdict      (n = %d)", len(sel))
    for name, label, rho, p, verdict in results:
        if math.isfinite(rho):
            log.info("  %-11s %+.2f   %.4f   %-11s  %s", name, rho, p, verdict, label)
        else:
            log.info(
                "  %-11s   n/a     n/a    %-11s  %s (no spread)", name, verdict, label
            )
    log.info("verdict (summary): %s", summary_verdict([x[4] for x in results]))
    return results


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
def planted_rows(n, effect, rng, name="cz_ns"):
    """Rows with random features and loss = effect * (rank of one feature) + noise."""
    rows = []
    for i in range(n):
        row = {k: float(rng.random()) for k, _ in FEATURES}
        row.update({"device": "sim", "link": str(i), "alpha_min": 0.98})
        rows.append(row)
    rk = ranks([r[name] for r in rows]) / n
    for r, v in zip(rows, rk):
        r["loss"] = effect * float(v) + float(rng.normal(0, 0.01))
    return rows


def check_pure(r):
    """Ranks, correlation, permutation p, the rules and the file helpers."""
    log.info("=== Check 1: the statistics on cases with known answers ===")
    r.append(
        check(list(ranks([1, 2, 2, 3])) == [1.0, 2.5, 2.5, 4.0], "ranks, ties averaged")
    )
    x = np.arange(15.0)
    r.append(
        check(
            abs(spearman(x, x**3) - 1) < 1e-12 and abs(spearman(x, -x) + 1) < 1e-12,
            "Spearman is +1 for a monotone rise and -1 for a fall",
        )
    )
    p_hi, p_lo = perm_p(x, x**3), perm_p(x, np.random.default_rng(3).random(15))
    r.append(
        check(
            p_hi < 1e-3 and p_lo > 0.05,
            f"permutation p: planted {p_hi:.5f}, noise {p_lo:.2f}",
        )
    )
    rng = np.random.default_rng(7)
    planted = analyse_rows(planted_rows(15, 0.12, rng))
    hit = {n: v for n, _, _, _, v in planted}
    r.append(
        check(
            hit["cz_ns"] == "ASSOCIATED",
            f"a planted effect on cz_ns is found (rho {planted[1][2]:+.2f})",
        )
    )
    r.append(
        check(
            all(v != "ASSOCIATED" for n, v in hit.items() if n != "cz_ns"),
            "no other feature is called associated",
        )
    )
    # Pure noise, many repeats: ASSOCIATED must be rare (false-alarm rate).
    alarms = 0
    for k in range(60):
        rows = planted_rows(15, 0.0, np.random.default_rng(100 + k))
        alarms += any(v == "ASSOCIATED" for *_, v in analyse_rows(rows))
    r.append(check(alarms <= 3, f"false alarms on pure noise: {alarms} of 60 runs"))
    table = [
        ((0.80, 0.001), "ASSOCIATED"),
        ((0.80, 0.02), "SUGGESTIVE"),
        ((0.60, 0.03), "SUGGESTIVE"),
        ((0.60, 0.20), "NONE"),
        ((0.40, 0.001), "NONE"),
    ]
    r.append(
        check(
            all(assoc_verdict(*a) == w for a, w in table),
            "rule: ASSOCIATED, SUGGESTIVE, NONE",
        )
    )
    r.append(
        check(
            summary_verdict(["NONE", "ASSOCIATED"]).startswith("the calibration holds")
            and summary_verdict(["SUGGESTIVE", "NONE"]).startswith("weak")
            and summary_verdict(["NONE"]).startswith("nothing"),
            "rule: summary",
        )
    )
    # File helpers.
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp, "job_a.json")
        good.write_text(
            json.dumps({"job_id": "x", "backend": "b", "links": [[0, 1, 0.1]]})
        )
        Path(tmp, "job_b.json").write_text(json.dumps({"job_id": "y"}))
        metas = load_metas(str(Path(tmp, "job_*.json")))
        rows = planted_rows(4, 0.1, np.random.default_rng(1))
        out = Path(tmp, "x", "r.csv")
        write_csv(rows, out)
        back = list(csv.DictReader(open(out)))
        r.append(
            check(
                len(metas) == 1 and metas[0]["job_id"] == "x",
                "job files: complete ones loaded, incomplete skipped",
            )
        )
        r.append(check(len(back) == 4 and "loss" in back[0], "the CSV round-trips"))


def fake_rows(n_links=15):
    """Rows on a simulated ibm_fez with a PLANTED dependence (dry run and checks)."""
    backend = FakeFez()
    target = backend.target
    links = eligible_links(target)
    chosen = select_links(
        links, count=n_links, percentiles=tuple(np.linspace(3, 97, n_links))
    )
    errs, adj = coupling_map(target)
    rng = np.random.default_rng(11)
    keys = build_jobs(chosen)[0]
    rows = []
    for i, ((a, b), e) in enumerate(chosen):
        feats = link_features(target, a, b, e, errs, adj)
        imp = implied_for(target, a, b, e)
        rows.append((a, b, e, feats, imp))
    # Plant: excess grows with the rank of the shorter T2.
    t2 = [f["t2_min"] for _, _, _, f, _ in rows]
    rk = ranks(t2) / len(t2)
    out = []
    for (a, b, e, feats, imp), v in zip(rows, rk):
        loss = imp + 0.12 * float(v) + float(rng.normal(0, 0.01))
        row = {
            "device": "fake_fez",
            "link": f"{a}-{b}",
            "loss": loss,
            "err": 0.015,
            "implied": imp,
            "excess": loss - imp,
            "class": classify(loss, 0.015, imp),
            "alpha_min": 0.98,
        }
        row.update(feats)
        out.append(row)
    return out, len(keys)


def check_features(r):
    """The feature extraction on the simulated device's own calibration."""
    log.info("=== Check 2: calibration features on a simulated ibm_fez ===")
    rows, n_keys = fake_rows()
    need = (
        "cz_err",
        "cz_ns",
        "t1_min",
        "t2_min",
        "ro_mean",
        "sx_mean",
        "spectators",
        "worst_nb_cz",
    )
    r.append(check(len(rows) == 15, f"{len(rows)} links chosen"))
    r.append(
        check(
            all(math.isfinite(row[k]) for row in rows for k in need),
            "the eight core features are finite on every link",
        )
    )
    has_f = all(math.isfinite(row["f_diff"]) for row in rows)
    log.info("frequency difference available on the simulated device: %s", has_f)
    r.append(
        check(
            all(row["spectators"] >= 1 for row in rows),
            "every link has at least one spectator",
        )
    )
    r.append(
        check(
            n_keys == 5 * 15,
            "the Lab 55 job layout is unchanged (5 circuits per link)",
        )
    )
    return rows


# ----------------------------------------------------------------------------
# Live
# ----------------------------------------------------------------------------
def live_rows(service, secret, metas, wait):
    """Re-fetch each Lab 55 job and join its links to the calibration."""
    rows = []
    for meta in metas:
        name = meta["backend"]
        chosen = [((a, b), e) for a, b, e in meta["links"]]
        backend = service.backend(name)
        target = backend.target
        try:
            stamp = backend.properties().last_update_date
        except Exception:  # noqa: BLE001
            stamp = "unknown"
        job = service.job(meta["job_id"])
        state = wait_for(job, wait, poll=10)
        if state != "DONE":
            log.warn_("job %s (%s) is %s; skipped", meta["job_id"], name, state)
            continue
        result = job.result()
        keys = build_jobs(chosen)[0]
        counts = [result[i].data.meas.get_counts() for i in range(len(keys))]
        by_key = dict(zip(keys, counts))
        errs, adj = coupling_map(target)
        log.info("%s: calibration time now %s; %d links", name, stamp, len(chosen))
        for i, ((a, b), e) in enumerate(chosen):
            now = errs.get((min(a, b), max(a, b)), float("nan"))
            if not (0.95 < now / e < 1.05):
                log.warn_(
                    "%s %d-%d: CZ error now %.5f, was %.5f when chosen: the "
                    "features are from a newer calibration",
                    name,
                    a,
                    b,
                    now,
                    e,
                )
            rows.append(
                make_row(name, a, b, e, target, errs, adj, link_detail(by_key, i))
            )
    return rows


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 56: calibration versus loss")
    ap.add_argument(
        "--live",
        action="store_true",
        help="read jobs and calibration from IBM (no QPU time)",
    )
    ap.add_argument("--meta-glob", default=META_GLOB, help="Lab 55 job files")
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait per job")
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    ap.set_defaults(retrieve=None)
    args = ap.parse_args(argv)
    r = []
    check_pure(r)
    fake = check_features(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.live:
        log.info("=== Check 3: report on a simulated device ===")
        rows = fake
    else:
        log.info("=== Check 3: connect, re-fetch the Lab 55 jobs, read calibration ===")
        metas = load_metas(args.meta_glob)
        r.append(check(len(metas) >= 1, f"{len(metas)} Lab 55 job files found"))
        if not metas:
            return 1 if not all(r) else 2
        service, secret = connect(args)
        if isinstance(service, str):
            return 2
        try:
            rows = live_rows(service, secret, metas, args.wait)
        except Exception as exc:  # noqa: BLE001
            log.fail(
                "could not read the jobs: %s: %s",
                type(exc).__name__,
                scrub(exc, secret),
            )
            return 1
        r.append(check(len(rows) >= 8, f"{len(rows)} links joined to the calibration"))
        if not rows:
            return 1
    out = Path("logs/lab56") / f"links_{stamp}.csv"
    write_csv(rows, out)
    log.info("per-link table written to %s", out)
    report(rows, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
