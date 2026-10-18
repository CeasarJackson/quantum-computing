#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab59_vertical_links.py

Purpose
-------
Labs 55 to 58 found that some links lose 3 to 10 times the singlet-preparation
loss that the calibration implies, that the excess repeats from run to run (Lab
57, rho +0.95), and that no calibration number ranks the links (Labs 56 and 58).
Lab 58 left one structural lead: the only link in the sample that joins two ROWS
of the heavy-hex chip (marrakesh 43-56, qubit numbers 13 apart) had the highest
pooled loss, +0.154 +/- 0.014. One link cannot test that. This lab does: on each
backend it measures 4 "vertical" links (qubit numbers differ by more than 1,
between two chip rows) and 4 "row" links (numbers differ by 1, inside a row)
matched to them by reported CZ error, with the Lab 55 measurement (40 circuits x
2000 shots, one job per backend), and compares the loss in excess of the
calibration between the two groups.

QPU TIME: one job per backend, about 31 to 42 s each in Lab 55 and Lab 57 (the
recorded rates give 32 to 83 s); each job has a hard cap of --max-seconds
(default 60). Three jobs: about 100 to 130 s expected, 180 s at the cap. Use
--only to run a subset of the backends.

THE SET: links with both readout attenuations at least 0.95 in the run (the Lab
55 / 56 / 57 cut). Excess = loss - implied loss (the Lab 55 formula).

THE RULES, fixed here before any job is run. V = mean excess of the vertical links
and R = mean excess of the row links, pooled over the backends; sigma_V and sigma_R
are the errors of those means (square root of the sum of squared link errors,
divided by the number of links); D = V - R, sigma_D = hypot(sigma_V, sigma_R).
Each group needs at least 3 links, otherwise the verdict is INSUFFICIENT.
  VERTICAL HIGHER  D >= +0.03 and D > 3 sigma_D : the links between chip rows lose
                   more; the stable excess has a structural cause (the coupler
                   type), and the calibration does not carry it.
  VERTICAL LOWER   D <= -0.03 and D < -3 sigma_D.
  NO DIFFERENCE    |D| < 2 sigma_D : the group sizes here can rule out a
                   difference of about 3 sigma_D or more (printed).
  UNCLEAR          otherwise.
Printed with no rule of their own: V and R for each backend, the HIGH count in
each group (Lab 55 class), and the matched pairs (vertical link, its row partner).

LIMITATIONS: 4 + 4 links per backend, so at most 12 + 12 in total, fewer after the
readout cut; one run, so a link that happens to lose its excess (Lab 57 found two)
adds noise; the row partner is matched on the reported CZ error only, which Lab
56 showed does not rank the loss.

FIRST LIVE RUN (2026-10-18 05:50; ibm_fez job dba9ci1gv2cc73e6hkgg, ibm_kingston
dba9ci8c77vs7382g4l0, ibm_marrakesh dba9cihgv2cc73e6hkh0; the first job ran 41 s,
all three were done within 45 s, no queue wait). 8 of the 24 links failed the
readout cut (fez 2, kingston 1, marrakesh 4 of its 8 incl. 3 of 4 vertical), so
the groups are 8 vertical and 8 row links, and marrakesh adds only 1 vertical.
The rule printed:
  pooled vertical +0.024 (n = 8), row +0.022 (n = 8); D = +0.002 +/- 0.010 (0.2
  sigma)  ->  NO DIFFERENCE: the two kinds of link lose alike (to this precision).
  Detectable at 3 sigma: about 0.029 or more. Per backend D: fez +0.030 +/- 0.016,
  kingston +0.005 +/- 0.016, marrakesh -0.033 +/- 0.023 (n = 1 vs 2).
  HIGH: 2 of 8 vertical and 1 of 8 row links.
Per-link reading (NOT pre-registered): the HIGH links are fez 118-129 (vertical,
+0.079 +/- 0.018, implied 0.012) and fez 29-38 (vertical, +0.090 +/- 0.023, implied
0.018), and marrakesh 131-132 (row, +0.075 +/- 0.018, implied 0.011); the rest are
within 3 sigma of the implied loss. Lab 58's lead, marrakesh 43-56 being the
highest, does not generalise: it was one link. fez 118-129 is the first link of the
original ibm_fez chain (qubits 129, 118, ...), which Lab 51 found to lose less
than the others; here it is HIGH,
so that "link 0 loses less" result does not reproduce either (one run).
READING: links between chip rows do not lose more than row links at a 0.03
difference (3 sigma); the stable per-link excess has no structural cause that this
split can see. Not tested: other structure (distance to the chip edge, the number
of neighbours of each qubit, which qubit of the pair has the larger T2), and the
excess over days.

Modes
-----
  (default)  DRY RUN (no account, no network): the selection and the submit path on a
             simulated ibm_fez, and the statistics on planted cases.
  --live     Submit one job per backend (REAL QPU TIME), wait, analyse.
  --collect  Fetch jobs submitted earlier (no QPU time) and analyse.

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is no --token
flag because command-line arguments leak into shell history. The key is never
logged, never written to a file, and is scrubbed from error text. The optional
instance CRN comes from IBM_QUANTUM_INSTANCE. Nothing secret touches disk.

Usage
-----
    python3 labs/lab59_vertical_links.py                                  # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN                  # hidden
    python3 labs/lab59_vertical_links.py --live --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab59_vertical_links.py --live --only ibm_fez --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab59_vertical_links.py --collect --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab59/lab59_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab59/job_<backend>_<id>.json (job id, backend, shots, links, implied loss)
logs/lab59/links_<YYYYmmdd_HHMMSS>.csv (one row per link: group, loss, excess)

Exit Codes
----------
0   Success (all checks passed; a job still queued is reported, not failed, and
    can be fetched later with --collect).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (missing key, package, bad flag).
130 User interrupted execution.
===============================================================================
"""

import argparse
import csv
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from applog import get_logger

log = get_logger("lab59")

# Everything shared is imported so each piece of logic exists once.
try:
    from lab17_mirror_benchmark import connect, make_sampler, scrub, wait_for
    from lab46_hardware_noise_budget import MAX_SHOTS
    from lab51_herald_and_isolate import check
    from lab52_gate_count_orientation_idle import plan_time, summarise
    from lab55_other_pairs_other_device import (
        build_jobs,
        classify,
        eligible_links,
        implied_link_loss,
        link_detail,
        select_links,
        transpile_for,
    )
    from lab56_calibration_vs_loss import GOOD_ALPHA, load_metas
    from qiskit_ibm_runtime.fake_provider import FakeFez
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_GLOB = "logs/lab59/job_*.json"
OUT_DIR = Path("logs/lab59")
BACKENDS = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")
N_PAIRS = 4  # vertical links per backend (and as many row partners)
PERCENTILES = (15, 40, 65, 90)  # of the CZ-error ranking of the vertical links
D_MIN, SIGMA_HIGH, SIGMA_NONE = 0.03, 3.0, 2.0
MIN_GROUP = 3


# ----------------------------------------------------------------------------
# Choosing the links
# ----------------------------------------------------------------------------
def is_vertical(a, b):
    """True when the qubit numbers differ by more than 1 (a link between rows)."""
    return abs(a - b) > 1


def select_matched(links, n_pairs=N_PAIRS, percentiles=PERCENTILES):
    """Vertical links at the given percentiles, each with a row partner.

    Returns [((a, b), error)]: the vertical links first, then their row partners
    in the same order. All 2 * n_pairs links use different qubits. The partner of a
    vertical link is the unused row link with the nearest reported CZ error.
    """
    vert = {k: v for k, v in links.items() if is_vertical(*k)}
    row = {k: v for k, v in links.items() if not is_vertical(*k)}
    chosen_v = select_links(vert, count=n_pairs, percentiles=percentiles)
    used = {q for (a, b), _ in chosen_v for q in (a, b)}
    partners = []
    for _, err in chosen_v:
        best = None
        for (a, b), e in row.items():
            if a in used or b in used:
                continue
            if best is None or abs(e - err) < abs(best[1] - err):
                best = ((a, b), e)
        if best is not None:
            used.update(best[0])
            partners.append(best)
    return chosen_v + partners


def group_of(link):
    """'vertical' or 'row' for a link given as 'a-b'."""
    a, b = (int(v) for v in link.split("-"))
    return "vertical" if is_vertical(a, b) else "row"


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def group_mean(rows):
    """(mean excess, error of the mean, n) of the rows."""
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(r["excess"] for r in rows) / n
    return mean, math.sqrt(sum(r["err"] ** 2 for r in rows)) / n, n


def difference(rows, min_alpha=GOOD_ALPHA):
    """(D, sigma_D, V, R, nV, nR) over the links with good readout."""
    good = [r for r in rows if r["alpha_min"] >= min_alpha]
    v, sv, nv = group_mean([r for r in good if r["group"] == "vertical"])
    rr, sr, nr = group_mean([r for r in good if r["group"] == "row"])
    return v - rr, math.hypot(sv, sr), v, rr, nv, nr


def verdict(d, sigma, nv, nr):
    """The pre-registered rule."""
    if nv < MIN_GROUP or nr < MIN_GROUP or not math.isfinite(d):
        return "INSUFFICIENT: fewer than 3 links with good readout in a group"
    if d >= D_MIN and d > SIGMA_HIGH * sigma:
        return "VERTICAL HIGHER: links between chip rows lose more"
    if d <= -D_MIN and d < -SIGMA_HIGH * sigma:
        return "VERTICAL LOWER: links between chip rows lose less"
    if abs(d) < SIGMA_NONE * sigma:
        return "NO DIFFERENCE: the two kinds of link lose alike (to this precision)"
    return "UNCLEAR: a difference between 2 and 3 sigma, or too small to matter"


# ----------------------------------------------------------------------------
# Rows, report, files
# ----------------------------------------------------------------------------
def link_rows(device, chosen, implied, by_key):
    """One dict per link: group, loss, error, excess, readout, class."""
    rows = []
    for i, ((a, b), cz) in enumerate(chosen):
        d = link_detail(by_key, i)
        imp = implied[i]
        rows.append(
            {
                "device": device,
                "link": f"{a}-{b}",
                "group": group_of(f"{a}-{b}"),
                "cz_err": cz,
                "implied": imp,
                "loss": d["loss"],
                "err": d["err"],
                "excess": d["loss"] - imp,
                "alpha_min": float(np.min(d["ro"][0])),
                "class": classify(d["loss"], d["err"], imp),
            }
        )
    return rows


def write_csv(rows, path):
    """Write the rows to a CSV file (no secrets are in them)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report(rows):
    """Print every link, the group means per backend and pooled, and the verdict."""
    log.info(
        "  device        link       group     CZ err  implied  loss +/- err  class"
    )
    for r in rows:
        log.info(
            "  %-13s %-8s  %-8s  %.4f  %.4f   %+.3f +/- %.3f  %s%s",
            r["device"],
            r["link"],
            r["group"],
            r["cz_err"],
            r["implied"],
            r["loss"],
            r["err"],
            r["class"],
            "" if r["alpha_min"] >= GOOD_ALPHA else "  (readout too poor)",
        )
    for dev in sorted({r["device"] for r in rows}):
        d, s, v, rr, nv, nr = difference([r for r in rows if r["device"] == dev])
        log.info(
            "  %-13s vertical %+.3f (n=%d)  row %+.3f (n=%d)  D %+.3f +/- %.3f",
            dev,
            v,
            nv,
            rr,
            nr,
            d,
            s,
        )
    d, s, v, rr, nv, nr = difference(rows)
    log.info("pooled: vertical %+.3f (n=%d), row %+.3f (n=%d)", v, nv, rr, nr)
    log.info(
        "D = vertical - row = %+.3f +/- %.3f (%.1f sigma)", d, s, d / s if s else 0
    )
    log.info("detectable at 3 sigma: a difference of about %.3f or more", 3 * s)
    for g in ("vertical", "row"):
        sel = [x for x in rows if x["group"] == g and x["alpha_min"] >= GOOD_ALPHA]
        log.info(
            "  %s: %d of %d links HIGH",
            g,
            sum(x["class"] == "HIGH" for x in sel),
            len(sel),
        )
    log.info("verdict: %s", verdict(d, s, nv, nr))


def planted_rows(n, effect, seed):
    """Synthetic rows: n vertical and n row links, vertical shifted by effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in ("vertical", "row"):
        for i in range(n):
            shift = effect if g == "vertical" else 0.0
            rows.append(
                {
                    "device": "sim",
                    "link": f"{i}-{i + (13 if g == 'vertical' else 1)}",
                    "group": g,
                    "cz_err": 0.003,
                    "implied": 0.015,
                    "loss": 0.015 + 0.03 + shift + float(rng.normal(0, 0.015)),
                    "err": 0.015,
                    "excess": 0.03 + shift + float(rng.normal(0, 0.015)),
                    "alpha_min": 0.98,
                    "class": "CONSISTENT",
                }
            )
    return rows


def check_pure(r):
    """The selection helpers, the statistics and the rule on known cases."""
    log.info("=== Check 1: the statistics on cases with known answers ===")
    big = difference(planted_rows(10, 0.10, 1))
    r.append(
        check(
            verdict(big[0], big[1], big[4], big[5]).startswith("VERTICAL HIGHER"),
            f"planted +0.10 difference read VERTICAL HIGHER (D {big[0]:+.3f})",
        )
    )
    low = difference(planted_rows(10, -0.10, 2))
    r.append(
        check(
            verdict(low[0], low[1], low[4], low[5]).startswith("VERTICAL LOWER"),
            "planted -0.10 difference read VERTICAL LOWER",
        )
    )
    reads = []
    for seed in range(20):
        d = difference(planted_rows(10, 0.0, 100 + seed))
        reads.append(verdict(d[0], d[1], d[4], d[5]).split(":")[0])
    r.append(
        check(
            not any(x.startswith("VERTICAL") for x in reads),
            f"no difference: never read VERTICAL in 20 runs ({reads.count('NO DIFFERENCE')} NO DIFFERENCE)",
        )
    )
    few = difference(planted_rows(2, 0.1, 3))
    r.append(
        check(
            verdict(few[0], few[1], few[4], few[5]).startswith("INSUFFICIENT"),
            "two links per group: INSUFFICIENT",
        )
    )
    poor = planted_rows(10, 0.10, 4)
    for x in poor[:10]:
        x["alpha_min"] = 0.5
    r.append(
        check(
            difference(poor)[4] == 0,
            "links with poor readout are left out of the groups",
        )
    )
    r.append(
        check(
            is_vertical(43, 56)
            and not is_vertical(13, 14)
            and group_of("5-18") == "vertical",
            "vertical / row rule",
        )
    )


# ----------------------------------------------------------------------------
# Submit path
# ----------------------------------------------------------------------------
class _Args:
    """Stand-in for the argparse namespace (the submit path needs these fields)."""

    def __init__(self, shots=100, live=False):
        """Fields the submit path reads: shots, live, sampler and the time cap."""
        self.shots, self.live, self.sampler, self.max_seconds = shots, live, "v2", 60


def submit_one(service, name, args, out_dir=OUT_DIR):
    """Choose the links on this backend, transpile and submit one job."""
    backend = service.backend(name)
    chosen = select_matched(eligible_links(backend.target))
    if len(chosen) < 2 * N_PAIRS:
        log.warn_("%s: only %d matched links found", name, len(chosen))
    keys, circuits = build_jobs(chosen)
    isa = transpile_for(backend, chosen, keys, circuits)
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    ok = twoq == [0 if isinstance(k[0], str) else 1 for k in keys]
    sx = backend.target["sx"]
    implied = [
        implied_link_loss(e, (sx[(a,)].error, sx[(b,)].error)) for (a, b), e in chosen
    ]
    plan_time(args.shots, len(isa))
    job = make_sampler(args, backend).run(isa)
    rec = {
        "job_id": job.job_id(),
        "backend": name,
        "shots": args.shots,
        "links": [[a, b, e] for (a, b), e in chosen],
        "implied": implied,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"job_{name}_{rec['job_id']}.json").write_text(json.dumps(rec))
    log.info("%s: job %s submitted, %d links", name, rec["job_id"], len(chosen))
    return job, ok, rec


def check_submit(r):
    """Selection and the submit path on a simulated ibm_fez (no network)."""
    log.info("=== Check 2: selection and the submit path on a simulated ibm_fez ===")
    fez = FakeFez()
    chosen = select_matched(eligible_links(fez.target))
    used = [q for (a, b), _ in chosen for q in (a, b)]
    nv = sum(is_vertical(*k) for k, _ in chosen)
    r.append(check(len(chosen) == 2 * N_PAIRS, f"{len(chosen)} links chosen"))
    r.append(check(len(used) == len(set(used)), "all links use different qubits"))
    r.append(
        check(
            nv == N_PAIRS and all(is_vertical(*k) for k, _ in chosen[:N_PAIRS]),
            f"{nv} vertical links first, the rest row links",
        )
    )
    gaps = [abs(chosen[i][1] - chosen[i + N_PAIRS][1]) for i in range(N_PAIRS)]
    r.append(
        check(
            max(gaps) < 0.01, f"partners matched on CZ error (max gap {max(gaps):.4f})"
        )
    )

    class Svc:
        """Minimal service: hands out the simulated backend."""

        def backend(self, name):
            """Return the simulated backend whatever the name."""
            return fez

    with tempfile.TemporaryDirectory() as tmp:
        job, ok, rec = submit_one(Svc(), "fake_fez", _Args(100), Path(tmp))
        counts = job.result()
        saved = list(Path(tmp).glob("job_*.json"))
        r.append(check(ok, "one two-qubit gate per pair circuit and no routing"))
        r.append(
            check(
                len(saved) == 1
                and len(json.loads(saved[0].read_text())["implied"]) == 2 * N_PAIRS,
                "the job file records links and implied losses",
            )
        )
        r.append(check(len(counts) == 5 * 2 * N_PAIRS, "all circuits came back"))
        by_key = {
            k: counts[i].data.meas.get_counts()
            for i, k in enumerate(
                build_jobs([((a, b), e) for a, b, e in rec["links"]])[0]
            )
        }
        rows = link_rows(
            "fake_fez",
            [((a, b), e) for a, b, e in rec["links"]],
            rec["implied"],
            by_key,
        )
        r.append(
            check(len(rows) == 2 * N_PAIRS, "rows built from the simulated counts")
        )
    return rows


# ----------------------------------------------------------------------------
# Live
# ----------------------------------------------------------------------------
def fetch_rows(service, rec, wait):
    """Rows of a finished job, or [] when it is not DONE in time."""
    job = service.job(rec["job_id"])
    if wait_for(job, wait, poll=10) != "DONE":
        log.warn_("%s: job not DONE yet; fetch later with --collect", rec["backend"])
        return []
    chosen = [((a, b), e) for a, b, e in rec["links"]]
    keys = build_jobs(chosen)[0]
    result = job.result()
    by_key = {k: result[i].data.meas.get_counts() for i, k in enumerate(keys)}
    return link_rows(rec["backend"], chosen, rec["implied"], by_key)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 59: links between chip rows")
    ap.add_argument("--live", action="store_true", help="submit jobs (QPU time)")
    ap.add_argument("--collect", action="store_true", help="fetch earlier jobs")
    ap.add_argument(
        "--only",
        help="comma-separated backend names (default: fez, kingston, marrakesh)",
    )
    ap.add_argument(
        "--shots", type=int, default=2000, help=f"shots per circuit, 100..{MAX_SHOTS}"
    )
    ap.add_argument("--max-seconds", type=int, default=60, help="QPU time cap per job")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait per job")
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    ap.add_argument("--sampler", choices=("v2", "client"), default="v2")
    args = ap.parse_args(argv)
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    if args.live and args.collect:
        ap.error("use --live OR --collect, not both.")
    args.retrieve = True if args.collect else None  # makes connect() log in
    r = []
    check_pure(r)
    fake = check_submit(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not (args.live or args.collect):
        log.info("=== Check 3: report on planted links ===")
        report(planted_rows(8, 0.06, 5))
        return summarise(r)

    names = args.only.split(",") if args.only else list(BACKENDS)
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    rows = []
    try:
        if args.live:
            log.info("=== Check 3: submit one job per backend ===")
            log.warn_(
                "LIVE: %d job(s), each capped at %d s", len(names), args.max_seconds
            )
            recs = [submit_one(service, n, args)[2] for n in names]
        else:
            log.info("=== Check 3: fetch the jobs ===")
            recs = [m for m in load_metas(JOB_GLOB) if m["backend"] in names]
            r.append(check(len(recs) >= 1, f"{len(recs)} job files found"))
            if not recs:
                return 1
        for rec in recs:
            rows += fetch_rows(service, rec, args.wait)
    except Exception as exc:  # noqa: BLE001
        log.fail("failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    if not rows:
        log.warn_("nothing to analyse yet: run again with --collect when DONE")
        return 0 if all(r) else 1
    write_csv(rows, OUT_DIR / f"links_{stamp}.csv")
    report(rows)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
