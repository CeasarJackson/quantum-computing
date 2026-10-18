#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab50_calibration_snapshot.py

Purpose
-------
Lab 49 showed that readout does not explain the ibm_fez depth-0 loss: about
0.25 of the energy (3% of each singlet's correlation) is lost in the three
singlet-preparation gates, against the 0.4% per two-qubit gate that the Lab 17
mirror benchmark measured on other qubits. One simple explanation is that the
three links used (129-118, 109-110, 111-112) are individually worse than
average. IBM publishes the error of every CZ link in the device calibration
data, and reading it costs NO QPU time (no job is submitted). This lab reads
it and answers three questions:

  1. What does the calibration say about the six qubits and their links:
     CZ error and duration of each link, single-qubit (sx) error, readout error,
     T1 and T2, and where each link ranks among all links of the device?
  2. What depth-0 loss does the calibration IMPLY? A CZ with average gate
     infidelity r behaves like the Lab 21 depolarizing step with
     lambda = r / 0.75 (checked numerically below), which takes lambda from
     each of the three singlet correlations in each of the three bases, so the
     three singlet gates lose 3 * sum(lambda) in energy, plus the sx gates (an
     estimated 3 per qubit, one-qubit lambda = 2 r). Compare with the gate-side
     loss that is left of the measured depth-0 total: 0.3125 +/- 0.031 (Lab 48)
     minus the 0.029 that Lab 49 attributes to readout.
  3. Are there better chains? A depth-first search over all 6-qubit paths of
     the coupling graph ranks them by the CZ error they would pay for the
     Lab 48 circuit (every bond gate is 3 CZ, the singlet links pay once
     more), and shows where the layout used so far stands.

THE RULE, fixed here before the first live read: if the calibration-implied
depth-0 loss is at least half of the measured gate-side loss, the reported link errors
account for (most of) the loss and the layout should change; if it is below a
quarter, the reported CZ errors do not explain it, and the cause is something
the calibration data do not contain (coherent error, crosstalk between the
three parallel gates, state-dependent error), which a per-link singlet test
could separate; in between, it explains part.

FIRST LIVE READ (ibm_fez, calibration time 2026-10-17 20:56 -04:00, about 80
minutes before the Lab 49 job): the six qubits sit on GOOD links. CZ error
0.0018, 0.0021, 0.0020, 0.0021, 0.0017 (better than 88%, 76%, 80%, 74%, 93% of
the 176 links; device median 0.0026), gate time 68 ns, sx error 0.00017 to
0.00034, T1 78 to 180 us, T2 47 to 125 us, readout error 0.006 to 0.020. The
calibration implies a depth-0 loss of 0.036 (singlet CZ 0.022 + sx 0.014),
13% of the 0.283 +/- 0.033 left for the gates, so by the rule fixed beforehand
the reported link errors do NOT explain it, and the layout is not the problem:
the cheapest chain on the device would save 9% of the CZ error (0.0319 against
0.0350). The calibration CZ error (0.2%) is also below the 0.36% per gate of the
mirror benchmark. So the depth-0 loss is 10 to 15 times what the reported gate
errors give, and something in the actual circuit is not in the calibration
data (initial-state excitation, crosstalk between the three parallel gates, or
a state-dependent or coherent error).

LIMITATIONS: the calibration is a snapshot (its timestamp is printed) taken
before or after the Lab 46, 48 and 49 jobs, not during them; calibration CZ
errors are average gate infidelities from randomized benchmarking, which sees
incoherent error mostly; crosstalk between parallel gates is not in them; the
sx count per qubit is an estimate. The default (dry run) reads the stored
snapshot of the simulated ibm_fez, which is old (see the printed date) and
says nothing about today's device; use --live for the real calibration.

Modes
-----
  (default)  DRY RUN on the stored snapshot of a simulated ibm_fez (offline).
  --live     Read the real device calibration (needs an IBM account; submits
             NO job and uses no QPU time).

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is no --token
flag because command-line arguments leak into shell history. The key is never
logged, never written to a file, and is scrubbed from error text. The optional
instance CRN comes from IBM_QUANTUM_INSTANCE. Nothing secret touches disk. The
connection code is shared with Lab 17 (labs/lab17_mirror_benchmark.py).

Usage
-----
    python3 labs/lab50_calibration_snapshot.py                    # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY      # hidden prompt
    python3 labs/lab50_calibration_snapshot.py --live --backend ibm_fez

Log files
---------
logs/lab50/lab50_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (missing key, package, bad flag).
130 User interrupted execution.
===============================================================================
"""

import argparse
import sys

import numpy as np

from applog import get_logger

log = get_logger("lab50")

# Cloud plumbing and the recorded numbers are reused, not copied.
try:
    from fezdata import LAYOUT
    from lab17_mirror_benchmark import connect, scrub
    from lab21_noise_aware_training import N, depolarize
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

# Lab 48: depth-0 energy -8.678 and -8.697 (exact -9), so a total loss of
# 0.3125 +/- 0.031. Lab 49: readout (predicted from the audit) accounts for
# 0.029 of it. What is left must come from the gates.
MEASURED_DEPTH0 = (0.3125, 0.031)
READOUT_PART = (0.029, 0.012)
MEASURED_GATES = (
    MEASURED_DEPTH0[0] - READOUT_PART[0],
    (MEASURED_DEPTH0[1] ** 2 + READOUT_PART[1] ** 2) ** 0.5,
)
SX_PER_QUBIT = 3  # estimate from the transpiled depth-0 circuits (Lab 48)
SINGLET_LINKS = ((0, 1), (2, 3), (4, 5))  # chain positions of the three singlets
RULE_HIGH, RULE_LOW = 0.5, 0.25  # fractions of the measured excess


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Pure functions (tested without any device)
# ----------------------------------------------------------------------------
def lambda_from_infidelity(r, n_qubits):
    """Depolarizing strength lambda that has average gate infidelity r.

    The channel is rho -> (1 - lambda) rho + lambda I/d. Its average fidelity
    is 1 - lambda (d - 1)/d with d = 2**n_qubits, so r = lambda (d - 1)/d.
    """
    d = 2**n_qubits
    return r * d / (d - 1)


def link_table(target):
    """{(a, b): error} for every CZ link (a < b), from a Qiskit target."""
    out = {}
    for (a, b), props in target["cz"].items():
        if props is None or props.error is None:
            continue
        key = (min(a, b), max(a, b))
        # Both directions are listed; keep the larger error to be cautious.
        out[key] = max(out.get(key, 0.0), float(props.error))
    return out


def rank_of(value, values):
    """Fraction of `values` that are strictly better (smaller) than `value`."""
    arr = np.asarray(list(values))
    return float(np.mean(arr < value))


def adjacency(links):
    """Neighbour lists from the link dictionary."""
    adj = {}
    for a, b in links:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def chain_cost(path, links):
    """CZ error paid by the Lab 48 circuit on this path (virtual order = path).

    Every one of the 5 bonds is used by the bond gates (3 CZ each, five bond
    gates in one layer); the three singlet links (positions 0-1, 2-3, 4-5) pay
    one more CZ for the singlet preparation.
    """
    cost = 0.0
    for i in range(len(path) - 1):
        err = links[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))]
        cost += 3 * err + (err if i % 2 == 0 else 0.0)
    return cost


def best_chains(links, length=N, top=3):
    """The `top` cheapest simple paths of `length` qubits (depth-first search)."""
    adj = adjacency(links)
    found = {}

    def extend(path):
        """Grow a simple path one neighbour at a time; record complete ones."""
        if len(path) == length:
            # A path and its reverse are the same chain; orient by first qubit.
            key = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
            found[key] = chain_cost(list(key), links)
            return
        for nxt in adj[path[-1]]:
            if nxt not in path:
                extend(path + [nxt])

    for start in adj:
        extend([start])
    return sorted(found.items(), key=lambda kv: kv[1])[:top]


def implied_depth0_loss(layout, links, sx_error):
    """(loss, singlet part, sx part) the calibration errors imply at depth 0."""
    singlet = 0.0
    for a, b in SINGLET_LINKS:
        err = links[(min(layout[a], layout[b]), max(layout[a], layout[b]))]
        # 3 bases, each singlet correlation scaled by (1 - lambda).
        singlet += 3 * lambda_from_infidelity(err, 2)
    sx = 0.0
    for q in layout:
        # One-qubit depolarizing on a singlet qubit also scales every basis.
        sx += 3 * SX_PER_QUBIT * lambda_from_infidelity(sx_error[q], 1) / 2
    return singlet + sx, singlet, sx


def verdict(implied, measured):
    """Apply the pre-registered rule."""
    share = implied / measured
    if share >= RULE_HIGH:
        return "account for (most of) the depth-0 loss: change the layout"
    if share < RULE_LOW:
        return "do NOT explain the depth-0 loss: look beyond the reported CZ errors"
    return "explain part of the depth-0 loss"


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
def check_pure(r):
    """Conversions, ranking, the chain search and the rule on known inputs."""
    log.info("=== Check 1: the conversions and the search on known inputs ===")
    # lambda <-> infidelity against the Lab 21 channel on a random state.
    rng = np.random.default_rng(50)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    worst = 0.0
    for lam in (0.002, 0.01, 0.05):
        out = depolarize(rho, 2, 0, 1 - lam)
        # Fidelity of the output with the input state.
        fid = float(np.real(psi.conj() @ out @ psi))
        # For any pure state the channel gives exactly 1 - lambda + lambda/4.
        worst = max(worst, abs(fid - (1 - 0.75 * lam)))
        r.append(
            check(
                abs(lambda_from_infidelity(1 - fid, 2) - lam) < 1e-12,
                f"infidelity {1 - fid:.5f} -> lambda {lam}",
            )
        )
    r.append(check(worst < 1e-12, "fidelity of the Lab 21 channel is 1 - 3 lambda / 4"))
    r.append(
        check(
            abs(lambda_from_infidelity(0.001, 1) - 0.002) < 1e-12,
            "one qubit: lambda = 2 r",
        )
    )
    # A line 0-1-2-3-4-5-6 with one very bad link: the best path avoids it.
    line = {(i, i + 1): 0.004 for i in range(6)}
    line[(2, 3)] = 0.05
    best = best_chains(line, length=6, top=5)
    r.append(check(len(best) == 2, f"a line of 7 has 2 paths of 6 ({len(best)})"))
    r.append(
        check(
            abs(best[0][1] - chain_cost(list(best[0][0]), line)) < 1e-15, "costs add up"
        )
    )
    # A square with a tail: the cheap route is chosen.
    graph = {(0, 1): 0.001, (1, 2): 0.001, (2, 3): 0.001, (3, 4): 0.001, (4, 5): 0.001}
    graph.update({(0, 5): 0.05, (1, 6): 0.001, (5, 6): 0.05})
    top = best_chains(graph, length=6, top=1)[0][0]
    r.append(
        check(set(top) == {0, 1, 2, 3, 4, 5}, "the cheapest path avoids bad links")
    )
    # Ranking and the rule.
    r.append(check(rank_of(0.0, [0.01, 0.02]) == 0.0, "rank: best link is 0"))
    r.append(
        check(
            abs(rank_of(0.015, [0.01, 0.02, 0.03]) - 1 / 3) < 1e-12,
            "rank: 1 of 3 better",
        )
    )
    r.append(check("most" in verdict(0.2, 0.28), "rule: high share"))
    r.append(check("part" in verdict(0.1, 0.28), "rule: middle share"))
    r.append(check("NOT" in verdict(0.02, 0.28), "rule: low share"))
    # The implied loss on a device where every error equals Lab 17's lambda.
    lam = 0.0037114
    links = {(i, i + 1): 0.75 * lam for i in range(5)}
    sx = {q: 0.0 for q in range(6)}
    loss, singlet, _ = implied_depth0_loss(list(range(6)), links, sx)
    r.append(
        check(
            abs(loss - 9 * lam) < 1e-12 and abs(singlet - loss) < 1e-12,
            f"with Lab 17's error everywhere the implied loss is 9 lambda ({loss:.4f})",
        )
    )


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def qubit_row(target, q):
    """(T1 us, T2 us, sx error, readout error) of one qubit, None when missing."""
    qp = target.qubit_properties[q] if target.qubit_properties else None
    t1 = getattr(qp, "t1", None)
    t2 = getattr(qp, "t2", None)
    sx = target["sx"][(q,)].error
    ro = target["measure"][(q,)].error
    return (
        t1 * 1e6 if t1 else float("nan"),
        t2 * 1e6 if t2 else float("nan"),
        sx,
        ro,
    )


def report(r, backend, layout, dry):
    """Print the snapshot of the layout, rank the links, and apply the rule."""
    target = backend.target
    where = "stored snapshot of a simulated device" if dry else "REAL calibration"
    try:
        stamp = backend.properties().last_update_date
    except Exception:  # noqa: BLE001
        stamp = "unknown"
    log.info("%s %s, calibration time: %s", backend.name, where, stamp)
    links = link_table(target)
    r.append(check(len(links) > 100, f"{len(links)} CZ links read"))
    err_all = np.array(list(links.values()))
    log.info(
        "all links: median CZ error %.4f, mean %.4f, 90th percentile %.4f, worst %.4f",
        np.median(err_all),
        err_all.mean(),
        np.percentile(err_all, 90),
        err_all.max(),
    )

    log.info("qubits of the layout %s:", layout)
    rows = {q: qubit_row(target, q) for q in layout}
    sx_err = {q: rows[q][2] for q in layout}
    for i, q in enumerate(layout):
        t1, t2, sx, ro = rows[q]
        log.info(
            "  virtual %d = physical %3d: T1 %6.1f us  T2 %6.1f us  sx error %.5f  readout error %.4f",
            i,
            q,
            t1,
            t2,
            sx,
            ro,
        )
    log.info("links of the layout (bond i joins virtual i and i+1):")
    for i in range(N - 1):
        a, b = layout[i], layout[i + 1]
        key = (min(a, b), max(a, b))
        if key not in links:
            log.fail("physical %d and %d are not coupled", a, b)
            r.append(False)
            continue
        dur = target["cz"][(a, b)].duration if (a, b) in target["cz"] else float("nan")
        role = "singlet + bond" if i % 2 == 0 else "bond only"
        log.info(
            "  bond %d (%3d-%3d, %s): CZ error %.4f  duration %.0f ns  better than %2.0f%% of links",
            i,
            a,
            b,
            role,
            links[key],
            dur * 1e9,
            100 * (1 - rank_of(links[key], err_all)),
        )

    implied, singlet, sx = implied_depth0_loss(layout, links, sx_err)
    meas, merr = MEASURED_GATES
    log.info(
        "implied depth-0 loss from the calibration: %.3f (singlet CZ %.3f + sx %.3f); "
        "measured total 0.3125 minus the Lab 49 readout part 0.029 = %.3f +/- %.3f "
        "from the gates (the Lab 21 model at lambda = 0.0037 gives 0.033)",
        implied,
        singlet,
        sx,
        meas,
        merr,
    )
    log.info(
        "implied loss as a share of the measured gate-side loss: %.0f%% (Lab 17 equivalent "
        "of 9 lambda: %.3f)",
        100 * implied / meas,
        9 * 0.0037114,
    )

    chains = best_chains(links)
    here = (
        chain_cost(list(layout), links)
        if all(
            (min(layout[i], layout[i + 1]), max(layout[i], layout[i + 1])) in links
            for i in range(N - 1)
        )
        else float("nan")
    )
    log.info("CZ error paid by the Lab 48 circuit (3 per bond + 1 per singlet link):")
    log.info("  this layout %s: %.4f", list(layout), here)
    for path, cost in chains:
        log.info("  cheaper chain %s: %.4f", list(path), cost)
    log.info(
        "verdict (pre-registered rule): the reported link errors %s",
        verdict(implied, meas),
    )
    if dry:
        log.info(
            "(dry run: that verdict is about the old stored snapshot, not ibm_fez today)"
        )


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 50: calibration snapshot")
    ap.add_argument("--live", action="store_true", help="read the real calibration")
    ap.add_argument("--backend", default="ibm_fez", help="backend name")
    ap.add_argument(
        "--layout",
        default=",".join(str(q) for q in LAYOUT),
        help="six physical qubits (default: the Lab 44 chain)",
    )
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    args = ap.parse_args(argv)
    layout = [int(x) for x in args.layout.split(",")]
    if len(layout) != N or len(set(layout)) != N:
        ap.error(f"--layout needs {N} different qubits")
    r = []
    check_pure(r)

    log.info("=== Check 2: read the calibration (no job is submitted) ===")
    # connect() reads these two attributes; there is no retrieve in this lab.
    args.retrieve = None
    secret = ""
    if args.live:
        service, secret = connect(args)
        if isinstance(service, str):
            return 2
        try:
            backend = service.backend(args.backend)
        except Exception as exc:  # noqa: BLE001
            log.fail(
                "could not read %s: %s: %s",
                args.backend,
                type(exc).__name__,
                scrub(exc, secret),
            )
            return 1
    else:
        from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: PLC0415

        backend = FakeFez()
    report(r, backend, layout, not args.live)
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
