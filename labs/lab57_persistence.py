#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab57_persistence.py

Purpose
-------
Lab 55 found singlet-preparation loss well above the calibration on some links of
three backends but not on others, and Lab 56 found that nothing in the backend's
reported calibration says which. That leaves one question the earlier data cannot
answer: IS A BAD LINK STILL BAD WHEN YOU MEASURE IT AGAIN? If the high-loss links are
high again, the loss is a stable property of the link that the calibration does not
report (a hidden coupler or neighbour problem). If they are not, the loss moves
from run to run, and a once-a-day calibration could not see it whatever it
measured. This lab repeats the Lab 55 measurement on EXACTLY the same links, with
the same circuits and shots, on the same backends, and compares the two runs link
by link. The first runs are re-fetched from their finished jobs (no QPU time).

QPU TIME: one job per backend, 40 circuits x 2000 shots = 80000 shots, about 32 to
83 s each by the recorded rates (Lab 55 ran in 31 to 42 s). Each job has a hard
cap of --max-seconds (default 60). Use --only to run a subset of the backends.

DATA: logs/lab55/job_<backend>_<id>.json, written by Lab 55 (job id, backend and
the eight links chosen then). This lab writes logs/lab57/job_<backend>_<id>.json
for each repeat (second job id, first job id, backend, shots, links).

THE SET: links whose two readout attenuations are both at least 0.95 IN BOTH RUNS
(the Lab 55 / Lab 56 cut; links with worse readout are noise-amplified).

THE RULES, fixed here before any repeat is run (loss = the Lab 55 amended loss,
summed over X, Y, Z; loss1 first run, loss2 repeat; Spearman rho of loss2 against
loss1 over the set, two-sided permutation p from 20000 shuffles):
  PERSISTENT  rho >= 0.70 and p < 0.01 : the order of the links repeats; the loss
              is a stable property of the link, and Lab 56's null means the
              calibration is missing a stable property.
  TRANSIENT   rho < 0.30 : the order does not repeat; the loss of a link changes
              from run to run, so a calibration cannot predict it.
  PARTLY      otherwise : some of the order repeats; a larger sample is needed.
  Supporting statistics, printed with no rule of their own:
  CHI2        mean over links of ((loss2 - loss1) / sqrt(err1^2 + err2^2))^2. About 1
              means the difference is shot noise (the same true loss); 2.5 or more
              means the true loss of a link changes between runs.
  SHIFT       mean (loss2 - loss1) with its error: a change of the whole device.
  HIGH        of the links classed HIGH in run 1 (Lab 55 class), how many are HIGH
              again, and how many of the others become HIGH.
With n about 15 only a strong persistence (rho near 0.7 or more) can be seen, so
PARTLY and TRANSIENT are weaker statements than PERSISTENT.

LIMITATIONS: a repeat 1 to 2 hours after the first run on a calibration that may
have been refreshed in between (the lab prints each backend's calibration time);
the same shot noise as Lab 55 (about 0.02 per link); one repeat only, so it cannot
separate a slow drift from random jumps.

Modes
-----
  (default)  DRY RUN on simulated links (no account, no network): tests the
             statistics, the report and the submit path on a simulated ibm_fez.
  --live     Submit one repeat job per backend (REAL QPU TIME), wait, analyse.
  --collect  Fetch repeat jobs submitted earlier (no QPU time) and analyse.

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is no --token
flag because command-line arguments leak into shell history. The key is never
logged, never written to a file, and is scrubbed from error text. The optional
instance CRN comes from IBM_QUANTUM_INSTANCE. Nothing secret touches disk.

Usage
-----
    python3 labs/lab57_persistence.py                                  # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN               # hidden
    python3 labs/lab57_persistence.py --live --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab57_persistence.py --live --only ibm_fez --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab57_persistence.py --collect --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab57/lab57_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab57/job_<backend>_<id>.json (repeat job, first job, links; no secrets)
logs/lab57/pairs_<YYYYmmdd_HHMMSS>.csv (one row per link: both losses and classes)

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

log = get_logger("lab57")

# Everything shared is imported so each piece of logic exists once.
try:
    from lab17_mirror_benchmark import connect, make_sampler, scrub, wait_for
    from lab46_hardware_noise_budget import MAX_SHOTS
    from lab51_herald_and_isolate import check
    from lab52_gate_count_orientation_idle import plan_time, summarise
    from lab55_other_pairs_other_device import (
        build_jobs,
        classify,
        implied_link_loss,
        link_detail,
        model_by_key,
        transpile_for,
    )
    from lab56_calibration_vs_loss import (
        GOOD_ALPHA,
        N_PERM,
        load_metas,
        perm_p,
        spearman,
    )
    from qiskit_ibm_runtime.fake_provider import FakeFez
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

FIRST_GLOB = "logs/lab55/job_*.json"
REPEAT_GLOB = "logs/lab57/job_*.json"
REPEAT_DIR = Path("logs/lab57")
RHO_PERSIST, P_PERSIST = 0.70, 0.01
RHO_TRANSIENT = 0.30
CHI_SHOT, CHI_CHANGE = 1.8, 2.5  # reading of the chi2 statistic
MIN_LINKS = 8  # below this no rank correlation is meaningful


# ----------------------------------------------------------------------------
# Pairing the two runs of a link
# ----------------------------------------------------------------------------
def pair_rows(device, chosen, implied, by1, by2):
    """One dict per link holding both runs: losses, errors, classes, readout."""
    rows = []
    for i, ((a, b), _) in enumerate(chosen):
        d1, d2 = link_detail(by1, i), link_detail(by2, i)
        imp = implied[i]
        rows.append(
            {
                "device": device,
                "link": f"{a}-{b}",
                "implied": imp,
                "loss1": d1["loss"],
                "err1": d1["err"],
                "class1": classify(d1["loss"], d1["err"], imp),
                "alpha1": float(np.min(d1["ro"][0])),
                "loss2": d2["loss"],
                "err2": d2["err"],
                "class2": classify(d2["loss"], d2["err"], imp),
                "alpha2": float(np.min(d2["ro"][0])),
            }
        )
    return rows


def good_pairs(rows, min_alpha=GOOD_ALPHA):
    """Links whose two readout attenuations are at least min_alpha in BOTH runs."""
    return [r for r in rows if min(r["alpha1"], r["alpha2"]) >= min_alpha]


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def persistence_stats(rows):
    """Rank correlation of the two runs, chi2, shift and the HIGH bookkeeping."""
    x = [r["loss1"] for r in rows]
    y = [r["loss2"] for r in rows]
    diff = np.array(y) - np.array(x)
    sig = np.array([math.hypot(r["err1"], r["err2"]) for r in rows])
    high1 = [r for r in rows if r["class1"] == "HIGH"]
    rest = [r for r in rows if r["class1"] != "HIGH"]
    return {
        "n": len(rows),
        "rho": spearman(x, y),
        "p": perm_p(x, y, N_PERM),
        "chi2": float(np.mean((diff / sig) ** 2)),
        "shift": float(np.mean(diff)),
        "shift_err": float(np.sqrt(np.sum(sig**2)) / len(rows)),
        "n_high1": len(high1),
        "high_again": sum(r["class2"] == "HIGH" for r in high1),
        "n_rest": len(rest),
        "new_high": sum(r["class2"] == "HIGH" for r in rest),
    }


def persistence_verdict(rho, p):
    """The pre-registered PERSISTENT / TRANSIENT / PARTLY rule."""
    if not (math.isfinite(rho) and math.isfinite(p)):
        return "PARTLY: the correlation could not be computed"
    if rho >= RHO_PERSIST and p < P_PERSIST:
        return (
            "PERSISTENT: the order of the links repeats; the loss is a stable "
            "property of the link that the calibration does not report"
        )
    if rho < RHO_TRANSIENT:
        return (
            "TRANSIENT: the order does not repeat; the loss of a link changes "
            "from run to run, so a calibration cannot predict it"
        )
    return "PARTLY persistent: some of the order repeats; a larger sample is needed"


def chi2_reading(chi2):
    """Plain-words reading of the chi2 statistic (no rule attached)."""
    if chi2 <= CHI_SHOT:
        return "the differences are about the size of the shot noise (same true loss)"
    if chi2 >= CHI_CHANGE:
        return "the true loss of a link changes between the runs"
    return "between the two readings: inconclusive"


# ----------------------------------------------------------------------------
# Report and files
# ----------------------------------------------------------------------------
def write_csv(rows, path):
    """Write the rows to a CSV file (no secrets are in them)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report(rows, dry):
    """Print every link's two runs, then the persistence statistics and verdict."""
    if dry:
        log.info("(dry run: simulated links with a PERSISTENT true loss)")
    log.info("every link, run 1 then the repeat (loss summed over X, Y, Z):")
    log.info("  device        link       implied   run 1               repeat")
    for r in rows:
        log.info(
            "  %-12s %-9s  %.4f   %+.3f +/- %.3f %-10s  %+.3f +/- %.3f %-10s",
            r["device"],
            r["link"],
            r["implied"],
            r["loss1"],
            r["err1"],
            r["class1"],
            r["loss2"],
            r["err2"],
            r["class2"],
        )
    sel = good_pairs(rows)
    log.info(
        "set: %d links with both attenuations >= %.2f in both runs",
        len(sel),
        GOOD_ALPHA,
    )
    if len(sel) < MIN_LINKS:
        log.warn_("fewer than %d links in the set: no verdict is meaningful", MIN_LINKS)
        return None
    st = persistence_stats(sel)
    log.info(
        "Spearman rho of repeat against run 1: %+.2f  (p = %.4f, n = %d)",
        st["rho"],
        st["p"],
        st["n"],
    )
    log.info("CHI2 %.2f: %s", st["chi2"], chi2_reading(st["chi2"]))
    log.info(
        "SHIFT of the mean loss, repeat minus run 1: %+.3f +/- %.3f",
        st["shift"],
        st["shift_err"],
    )
    log.info(
        "HIGH: %d of %d links HIGH in run 1 are HIGH again; %d of %d others became HIGH",
        st["high_again"],
        st["n_high1"],
        st["new_high"],
        st["n_rest"],
    )
    log.info("verdict (persistence rule): %s", persistence_verdict(st["rho"], st["p"]))
    return st


# ----------------------------------------------------------------------------
# Synthetic cases (dry run and checks)
# ----------------------------------------------------------------------------
def synthetic_rows(n, mode, seed):
    """n links with true depolarizing strengths; the repeat is persistent or not."""
    rng = np.random.default_rng(seed)
    lam1 = rng.uniform(0.005, 0.05, n)
    lam2 = lam1 if mode == "persistent" else rng.permutation(lam1)
    if mode == "partly":
        lam2 = 0.5 * lam1 + 0.5 * rng.permutation(lam1)
    chosen = [((2 * i, 2 * i + 1), 0.002) for i in range(n)]
    by1 = model_by_key(list(lam1), 2000, rng)
    by2 = model_by_key(list(lam2), 2000, rng)
    imp = [implied_link_loss(0.002, (0.0002, 0.0002))] * n
    rows = pair_rows("sim", chosen, imp, by1, by2)
    return rows


def check_pure(r):
    """Statistics, the rules and the file helpers, on cases with known answers."""
    log.info("=== Check 1: the statistics on cases with known answers ===")
    per = persistence_stats(synthetic_rows(15, "persistent", 1))
    r.append(
        check(
            persistence_verdict(per["rho"], per["p"]).startswith("PERSISTENT"),
            f"persistent case: rho {per['rho']:+.2f}, p {per['p']:.4f}",
        )
    )
    ind = persistence_stats(synthetic_rows(15, "transient", 2))
    r.append(
        check(
            persistence_verdict(ind["rho"], ind["p"]).startswith("TRANSIENT"),
            f"independent case: rho {ind['rho']:+.2f}",
        )
    )
    r.append(
        check(
            per["chi2"] < 2.0 and ind["chi2"] > 3.0,
            f"chi2: persistent {per['chi2']:.2f}, independent {ind['chi2']:.2f}",
        )
    )
    # False alarms: independent runs must almost never read PERSISTENT.
    alarms = 0
    for k in range(40):
        s = persistence_stats(synthetic_rows(15, "transient", 200 + k))
        alarms += persistence_verdict(s["rho"], s["p"]).startswith("PERSISTENT")
    r.append(
        check(alarms <= 1, f"independent runs read PERSISTENT {alarms} of 40 times")
    )
    table = [
        ((0.85, 0.001), "PERSISTENT"),
        ((0.85, 0.05), "PARTLY"),
        ((0.5, 0.02), "PARTLY"),
        ((0.1, 0.4), "TRANSIENT"),
        ((-0.2, 0.5), "TRANSIENT"),
    ]
    r.append(
        check(
            all(persistence_verdict(*a).startswith(w) for a, w in table),
            "rule: PERSISTENT, PARTLY, TRANSIENT",
        )
    )
    r.append(
        check(
            chi2_reading(1.0).startswith("the differences")
            and chi2_reading(4.0).startswith("the true loss")
            and chi2_reading(2.1).startswith("between"),
            "chi2 reading",
        )
    )
    rows = synthetic_rows(10, "persistent", 3)
    rows[0]["alpha2"] = 0.80
    r.append(
        check(
            len(good_pairs(rows)) == 9,
            "a link with poor readout in either run is left out",
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp, "x", "p.csv")
        write_csv(rows, out)
        r.append(
            check(len(list(csv.DictReader(open(out)))) == 10, "the CSV round-trips")
        )


class _Args:
    """Stand-in for the argparse namespace (the submit path needs these fields)."""

    def __init__(self, shots=100, live=False):
        """Fields the submit path reads: shots, live, sampler and the time cap."""
        self.shots, self.live, self.sampler, self.max_seconds = shots, live, "v2", 60


def submit_one(service, meta, args, out_dir=REPEAT_DIR):
    """Transpile the Lab 55 circuits onto the same links and submit one job."""
    backend = service.backend(meta["backend"])
    chosen = [((a, b), e) for a, b, e in meta["links"]]
    keys, circuits = build_jobs(chosen)
    isa = transpile_for(backend, chosen, keys, circuits)
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    ok = twoq == [0 if isinstance(k[0], str) else 1 for k in keys]
    plan_time(args.shots, len(isa))
    job = make_sampler(args, backend).run(isa)
    job_id = job.job_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "job_id": job_id,
        "first_job_id": meta["job_id"],
        "backend": meta["backend"],
        "shots": args.shots,
        "links": meta["links"],
    }
    (out_dir / f"job_{meta['backend']}_{job_id}.json").write_text(json.dumps(rec))
    log.info("%s: repeat job %s submitted", meta["backend"], job_id)
    return job, ok, rec


def check_submit(r):
    """The submit path on a simulated ibm_fez with two links (no network)."""
    log.info("=== Check 2: the submit path on a simulated ibm_fez ===")
    from lab55_other_pairs_other_device import eligible_links, select_links

    fez = FakeFez()
    chosen = select_links(eligible_links(fez.target), count=2, percentiles=(30, 70))
    meta = {
        "job_id": "first",
        "backend": "fake_fez",
        "links": [[a, b, e] for (a, b), e in chosen],
    }

    class Svc:
        """Minimal service: hands out the simulated backend."""

        def backend(self, name):
            """Return the simulated backend whatever the name."""
            return fez

    with tempfile.TemporaryDirectory() as tmp:
        job, ok, rec = submit_one(Svc(), meta, _Args(100), Path(tmp))
        saved = list(Path(tmp).glob("job_*.json"))
        counts = job.result()
        r.append(check(ok, "one two-qubit gate per pair circuit and no routing"))
        r.append(
            check(
                len(saved) == 1
                and json.loads(saved[0].read_text())["first_job_id"] == "first",
                "the repeat job file records both job ids",
            )
        )
        r.append(check(len(counts) == 10, "ten circuits came back for two links"))


# ----------------------------------------------------------------------------
# Live
# ----------------------------------------------------------------------------
def fetch_counts(service, job_id, n_keys, wait):
    """Counts list of a finished job, or None if it is not DONE in time."""
    job = service.job(job_id)
    if wait_for(job, wait, poll=10) != "DONE":
        return None
    result = job.result()
    return [result[i].data.meas.get_counts() for i in range(n_keys)]


def collect_rows(service, repeats, wait):
    """Join each repeat job to its first job and the backend's calibration."""
    rows = []
    for rec in repeats:
        name = rec["backend"]
        chosen = [((a, b), e) for a, b, e in rec["links"]]
        keys = build_jobs(chosen)[0]
        c2 = fetch_counts(service, rec["job_id"], len(keys), wait)
        c1 = fetch_counts(service, rec["first_job_id"], len(keys), wait)
        if c1 is None or c2 is None:
            log.warn_("%s: a job is not DONE yet; fetch later with --collect", name)
            continue
        target = service.backend(name).target
        sx = target["sx"]
        imp = [
            implied_link_loss(e, (sx[(a,)].error, sx[(b,)].error))
            for (a, b), e in chosen
        ]
        rows += pair_rows(name, chosen, imp, dict(zip(keys, c1)), dict(zip(keys, c2)))
    return rows


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 57: do bad links stay bad")
    ap.add_argument(
        "--live",
        action="store_true",
        help="submit one repeat job per backend (QPU time)",
    )
    ap.add_argument(
        "--collect",
        action="store_true",
        help="fetch repeat jobs submitted earlier (no QPU time)",
    )
    ap.add_argument(
        "--only",
        help="comma-separated backend names to run (default: all in the Lab 55 files)",
    )
    ap.add_argument(
        "--shots",
        type=int,
        default=2000,
        help=f"shots per circuit, 100..{MAX_SHOTS} (keep 2000 to match run 1)",
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
    check_submit(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not (args.live or args.collect):
        log.info("=== Check 3: report on simulated links ===")
        rows = synthetic_rows(16, "persistent", 5)
        report(rows, True)
        return summarise(r)

    only = set(args.only.split(",")) if args.only else None
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    try:
        if args.live:
            log.info("=== Check 3: submit one repeat job per backend ===")
            firsts = [
                m
                for m in load_metas(FIRST_GLOB)
                if only is None or m["backend"] in only
            ]
            r.append(
                check(len(firsts) >= 1, f"{len(firsts)} Lab 55 job files selected")
            )
            if not firsts:
                return 1
            log.warn_(
                "LIVE: %d job(s), each capped at %d s of QPU time",
                len(firsts),
                args.max_seconds,
            )
            repeats = [submit_one(service, m, args)[2] for m in firsts]
        else:
            log.info("=== Check 3: fetch the repeat jobs ===")
            repeats = [
                m
                for m in load_metas(REPEAT_GLOB)
                if only is None or m["backend"] in only
            ]
            r.append(check(len(repeats) >= 1, f"{len(repeats)} repeat job files found"))
            if not repeats:
                return 1
        rows = collect_rows(service, repeats, args.wait)
    except Exception as exc:  # noqa: BLE001
        log.fail("failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    if not rows:
        log.warn_(
            "nothing to analyse yet: run again with --collect when the jobs are DONE"
        )
        return 0 if all(r) else 1
    out = REPEAT_DIR / f"pairs_{stamp}.csv"
    write_csv(rows, out)
    log.info("per-link table written to %s", out)
    report(rows, False)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
