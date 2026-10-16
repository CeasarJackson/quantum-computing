#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab43_decoder_cost.py

Purpose
-------
Lab 22 wrote a union-find (UF) decoder and showed it is a little less accurate
than minimum-weight matching. Its real selling point is COST: it is meant to
run in hardware (an FPGA or an ASIC next to the cryostat) fast enough to keep
up with a superconducting machine, which measures its stabilizers about once
per microsecond. If the decoder falls behind, the backlog grows and the
computation stalls. This lab builds a hardware-style cost model.

  * INSTRUMENTATION. A copy of the Lab 22 decoder that counts elementary
    operations: half-edge growth steps, union-find steps, merges, and the edge
    visits of peeling. It must give exactly the same corrections as Lab 22.
  * LOAD. Below threshold the work is proportional to the number of defects
    (about 13-17 operations each), and the number of defects is proportional
    to the space-time volume, so the cost PER STABILIZER ROUND is independent
    of the code distance. Above threshold clusters percolate and the cost per
    defect climbs with d.
  * PARALLEL DEPTH. Growth rounds (the part hardware does in parallel) grow
    only slowly with d, while the serial work grows with the volume.
  * CAPACITY. A linear-in-p model for the cost per stabilizer round, fitted on
    small codes, is tested on a held-out large code, then used to ask how fast
    a decoder core must be to keep up, how many cores a machine needs, and how
    much memory the decoder state takes.
  * MATCHING. For contrast, the textbook dense-blossom bound m^3 in the number
    of defects m (modern sparse blossom does far better in practice, so this
    is a model, not a measurement).

Assumptions are marked where they enter: 8 clock cycles per elementary
operation on a serial core, 2 cycles per parallel growth round, 1 cycle per
node when peeling, a 1 microsecond stabilizer cycle.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab43_decoder_cost.py
    python3 labs/lab43_decoder_cost.py --shots 400 --seed 7
    python3 labs/lab43_decoder_cost.py --quick

Log files
---------
logs/lab43/lab43_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (pymatching).
130 User interrupted execution.
===============================================================================
"""

import argparse
import math
import sys

import numpy as np

from applog import get_logger

log = get_logger("lab43")

try:
    import pymatching

    from lab22_union_find_decoder import UnionFind, build_code
except ImportError:  # pragma: no cover
    log.fail("pymatching is not installed: pip install pymatching")
    sys.exit(2)

# Hardware assumptions (see the module docstring).
CYCLES_PER_OP = 8
CYCLES_PER_GROWTH_ROUND = 2
CYCLES_PER_PEEL_NODE = 1
STABILIZER_CYCLE_S = 1e-6


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


class CountingUF(UnionFind):
    """The Lab 22 decoder with operation counters (same algorithm, same output)."""

    def decode_counted(self, defects):
        """Return (flipped edges, correction, counters) for a defect list."""
        n = self.n_nodes
        parent = list(range(n))
        parity = [0] * n
        touches = [False] * n
        members = [[i] for i in range(n)]
        touches[self.boundary] = True
        for d in defects:
            parity[d] = 1
        # Counters: growth rounds, edge growth steps, find steps, merges,
        # and the size of the largest cluster at any time.
        cnt = {"rounds": 0, "grow": 0, "finds": 0, "unions": 0, "maxcl": 1}
        growth, full = {}, set()

        def find(x):
            """Root of x's cluster, with path compression; counts every step."""
            cnt["finds"] += 1
            while parent[x] != x:
                cnt["finds"] += 1
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            """Merge two clusters, keeping the larger member list."""
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            cnt["unions"] += 1
            if len(members[ra]) < len(members[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            parity[ra] ^= parity[rb]
            touches[ra] = touches[ra] or touches[rb]
            members[ra].extend(members[rb])
            members[rb] = []
            cnt["maxcl"] = max(cnt["maxcl"], len(members[ra]))

        while True:
            active = {find(d) for d in defects}
            active = [c for c in active if parity[c] and not touches[c]]
            if not active:
                break
            cnt["rounds"] += 1
            newly = []
            for c in active:
                for node in members[c]:
                    for e in self.adj[node]:
                        if e in full:
                            continue
                        cnt["grow"] += 1
                        growth[e] = growth.get(e, 0) + 1
                        if growth[e] >= 2:
                            newly.append(e)
            for e in newly:
                if e not in full:
                    full.add(e)
                    union(self.eu[e], self.ev[e])
        cnt["full"] = len(full)
        # Peeling visits both ends of every fully grown edge once.
        cnt["ops"] = cnt["grow"] + cnt["finds"] + cnt["unions"] + 2 * len(full)
        flips, corr = self._peel(full, defects)
        return flips, corr, cnt


def events_3d(d, p, shots, rng):
    """Detection events of d rounds of noisy syndrome extraction (p = q).

    Data errors flip qubits and each syndrome bit is misread with the same
    probability, as in Labs 18 and 22; the events are the changes between
    consecutive rounds, with a final perfect round.
    """
    hz = build_code(d)[1]
    n, checks = d * d, hz.shape[0]
    data = (rng.random((shots, d, n)) < p).astype(np.uint8)
    meas = (rng.random((shots, d, checks)) < p).astype(np.uint8)
    # Errors accumulate over rounds; each round's syndrome may be misread.
    total = np.cumsum(data, axis=1) % 2
    syn = ((total @ hz.T) % 2) ^ meas
    syn = np.concatenate([syn, ((total[:, -1] @ hz.T) % 2)[:, None, :]], axis=1)
    events = syn.copy()
    events[:, 1:] ^= syn[:, :-1]
    return hz, events.reshape(shots, -1).astype(np.uint8)


def decoder_for(d):
    """Counting decoder on the space-time matching graph of distance d."""
    hz = build_code(d)[1]
    graph = pymatching.Matching.from_check_matrix(
        hz, weights=1.0, repetitions=d + 1, timelike_weights=1.0
    )
    return CountingUF(graph), hz.shape[0]


def measure(d, p, shots, rng):
    """Decode `shots` random shots; return per-shot counter arrays and sizes."""
    # One decoder per distance; events are drawn fresh for every call.
    uf, checks = decoder_for(d)
    _, events = events_3d(d, p, shots, rng)
    # One row of counters per shot.
    rows = []
    for ev in events:
        defects = np.flatnonzero(ev)
        _, _, cnt = uf.decode_counted(defects)
        rows.append(
            (len(defects), cnt["rounds"], cnt["ops"], cnt["maxcl"], cnt["full"])
        )
    # Columns: defects, growth rounds, operations, largest cluster, grown edges.
    arr = np.array(rows, dtype=float)
    return {
        "d": d,
        "p": p,
        "nodes": uf.n_nodes,
        "edges": len(uf.eu),
        "checks": checks,
        "defects": arr[:, 0],
        "rounds": arr[:, 1],
        "ops": arr[:, 2],
        "maxcl": arr[:, 3],
        "full": arr[:, 4],
    }


def latency_cycles(m):
    """Per-shot latency of a parallel-growth, serial-peel hardware decoder.

    Growth runs one round per step in all clusters at once; peeling then walks
    the largest cluster (clusters peel in parallel with each other).
    """
    return CYCLES_PER_GROWTH_ROUND * m["rounds"] + CYCLES_PER_PEEL_NODE * m["maxcl"]


def ops_per_check_round(m):
    """Mean operations per stabilizer per round (the d-independent load)."""
    return float(m["ops"].mean() / ((m["d"] + 1) * m["checks"]))


def slope(x, y):
    """Log-log slope of y against x (least squares)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def check_instrument(r, ctx):
    """Check 1: counting changes nothing about the decoder's answers."""
    log.info("=== Check 1: instrumented decoder equals Lab 22 ===")
    rng = np.random.default_rng(ctx["seed"])
    same = consistent = total = 0
    for d in (3, 5, 7):
        uf, _ = decoder_for(d)
        for p in (0.01, 0.03):
            _, events = events_3d(d, p, 100, rng)
            for ev in events:
                defects = np.flatnonzero(ev)
                flips, corr, _ = uf.decode_counted(defects)
                _, corr_ref = uf.decode(defects)
                same += int(np.array_equal(corr, corr_ref))
                consistent += int(uf.consistent(flips, defects))
                total += 1
    log.info("%d shots: identical %d, consistent %d", total, same, consistent)
    r.append(check(same == total, "same corrections as the Lab 22 decoder"))
    r.append(check(consistent == total, "every correction reproduces its syndrome"))
    # No defects means no work at all.
    uf, _ = decoder_for(5)
    _, _, cnt = uf.decode_counted(np.array([], dtype=int))
    r.append(
        check(cnt["ops"] == 0 and cnt["rounds"] == 0, "no defects: zero operations")
    )


def check_load(r, ctx):
    """Check 2: below threshold the load is linear in defects and extensive."""
    log.info("=== Check 2: cost per defect and per stabilizer round ===")
    # A different stream from Check 1, so the two checks are independent.
    rng = np.random.default_rng(ctx["seed"] + 1)
    table = {}
    for p in ctx["ps"]:
        for d in ctx["ds"]:
            table[(p, d)] = measure(d, p, ctx["shots"], rng)
    ctx["table"] = table
    for p in ctx["ps"]:
        for d in ctx["ds"]:
            m = table[(p, d)]
            log.info(
                "p=%.3f d=%2d: defects %6.2f  ops %7.1f  ops/defect %5.1f"
                "  rounds %.2f  ops/check-round %.3f",
                p,
                d,
                m["defects"].mean(),
                m["ops"].mean(),
                m["ops"].mean() / m["defects"].mean(),
                m["rounds"].mean(),
                ops_per_check_round(m),
            )
    # Operations per defect: a small constant below threshold. (Values 8-24
    # leave room for statistical noise; typical values are 13-18.)
    below = [p for p in ctx["ps"] if p <= 0.01]
    # The smallest distance is skipped below: boundary effects dominate at d=5.
    for p in below:
        per_def = [
            table[(p, d)]["ops"].mean() / table[(p, d)]["defects"].mean()
            for d in ctx["ds"]
        ]
        spread = (max(per_def) - min(per_def)) / np.mean(per_def)
        r.append(
            check(
                8 < min(per_def) and max(per_def) < 24 and spread < 0.3,
                f"p={p}: 8-24 operations per defect, spread {spread:.0%}",
            )
        )
    # Extensive: the cost per stabilizer per round does not depend on d.
    for p in below:
        loads = [ops_per_check_round(table[(p, d)]) for d in ctx["ds"][1:]]
        spread = (max(loads) - min(loads)) / np.mean(loads)
        r.append(
            check(spread < 0.3, f"p={p}: cost per check-round flat in d ({spread:.0%})")
        )
    # Total cost is linear in the space-time volume (number of graph nodes).
    p = 0.005 if 0.005 in ctx["ps"] else below[0]
    nodes = [table[(p, d)]["nodes"] for d in ctx["ds"]]
    ops = [table[(p, d)]["ops"].mean() for d in ctx["ds"]]
    sl = slope(nodes, ops)
    log.info("log-log slope of ops vs volume at p=%s: %.2f", p, sl)
    r.append(
        check(0.9 < sl < 1.25, f"cost grows linearly with volume (slope {sl:.2f})")
    )
    # Defects per node per p: the "noise density" that sets the load.
    dens = [
        table[(p, d)]["defects"].mean() / (table[(p, d)]["nodes"] * p)
        for d in ctx["ds"][1:]
    ]
    r.append(
        check(
            max(dens) / min(dens) < 1.25, f"defects ~ nodes x p ({np.mean(dens):.1f})"
        )
    )


def check_threshold(r, ctx):
    """Check 3: above threshold the clusters percolate and the cost explodes."""
    log.info("=== Check 3: percolation above threshold ===")
    table, ds = ctx["table"], ctx["ds"]
    # Lowest and highest physical rate: well below and well above threshold.
    lo, hi = min(ctx["ps"]), max(ctx["ps"])
    frac = {
        (p, d): table[(p, d)]["maxcl"].mean() / table[(p, d)]["nodes"]
        for p in (lo, hi)
        for d in ds
    }
    for p in (lo, hi):
        log.info(
            "p=%.3f largest cluster / graph: %s",
            p,
            ["%.3f" % frac[(p, d)] for d in ds],
        )
    # Below threshold the largest cluster is a shrinking fraction of the graph;
    # above it, it is a finite fraction that does not shrink.
    r.append(
        check(frac[(lo, ds[-1])] < 0.6 * frac[(lo, ds[0])], "p low: clusters shrink")
    )
    r.append(
        check(
            frac[(hi, ds[-1])] > 0.2, "p high: a giant cluster spans >20% of the graph"
        )
    )
    r.append(
        check(
            frac[(hi, ds[-1])] >= 0.9 * frac[(hi, ds[0])],
            "p high: fraction does not shrink with d",
        )
    )
    per_def = {
        p: [table[(p, d)]["ops"].mean() / table[(p, d)]["defects"].mean() for d in ds]
        for p in (lo, hi)
    }
    r.append(
        check(
            per_def[hi][-1] > 1.6 * per_def[lo][-1],
            "operations per defect at high p are >1.6x",
        )
    )
    r.append(
        check(per_def[hi][-1] > per_def[hi][0], "...and rise with d (superlinear cost)")
    )


def check_depth(r, ctx):
    """Check 4: the parallel depth grows far more slowly than the serial work."""
    log.info("=== Check 4: parallel depth and hardware latency ===")
    table, ds = ctx["table"], ctx["ds"]
    p = 0.005 if 0.005 in ctx["ps"] else min(ctx["ps"])
    small, large = table[(p, ds[0])], table[(p, ds[-1])]
    # Ratio of work (serial operations) to depth (growth rounds) over d.
    work = large["ops"].mean() / small["ops"].mean()
    depth = large["rounds"].mean() / small["rounds"].mean()
    log.info(
        "d %d -> %d at p=%s: serial work x%.1f, growth rounds x%.1f",
        ds[0],
        ds[-1],
        p,
        work,
        depth,
    )
    r.append(check(work > 3 * depth, "serial work grows >3x faster than growth rounds"))
    worst = max(
        float(table[(q, d)]["rounds"].max() - 2 * (d + 1))
        for q in ctx["ps"]
        for d in ds
    )
    r.append(check(worst <= 0, "growth rounds never exceed 2(d+1) half-edge steps"))
    # Latency of a parallel-growth decoder, 99th percentile over shots.
    lat = {d: np.percentile(latency_cycles(table[(p, d)]), 99) for d in ds}
    log.info(
        "p99 latency (cycles) at p=%s: %s", p, {d: round(v) for d, v in lat.items()}
    )
    gain = lat[ds[-1]] / lat[ds[0]]
    r.append(
        check(
            gain < work / 2, f"latency grows x{gain:.1f}, half the work growth or less"
        )
    )
    # At 1 GHz these are nanoseconds, against a microsecond cycle.
    ns = lat[ds[-1]]
    r.append(
        check(ns * 1e-9 < 0.2 * STABILIZER_CYCLE_S, f"p99 latency {ns:.0f} ns << 1 us")
    )


def check_capacity(r, ctx):
    """Check 5: a fitted cost model, tested on a held-out large code."""
    log.info("=== Check 5: capacity model, validated on a held-out code ===")
    table, ds = ctx["table"], ctx["ds"]
    below = [p for p in ctx["ps"] if p <= 0.01]
    # The model is deliberately simple: cost per check-round scales with p.
    # Fit: operations per check-round = slope * p (line through the origin),
    # least squares over the training codes (all but the largest d).
    xs = np.array([p for p in below for _ in ds[:-1]])
    ys = np.array([ops_per_check_round(table[(p, d)]) for p in below for d in ds[:-1]])
    slope_p = float(xs @ ys / (xs @ xs))
    log.info("model: operations per check-round = %.1f x p", slope_p)
    # Held-out: a bigger code and a physical rate not used in the fit.
    rng = np.random.default_rng(ctx["seed"] + 2)
    d_test, p_test = ctx["test"]
    held = measure(d_test, p_test, ctx["shots"], rng)
    got, want = ops_per_check_round(held), slope_p * p_test
    log.info("held-out d=%d p=%s: measured %.4f, model %.4f", d_test, p_test, got, want)
    r.append(
        check(abs(got - want) < 0.2 * want, f"model predicts d={d_test} within 20%")
    )
    # Capacity of one serial core: highest p it can keep up with.
    log.info("highest p one serial core keeps up with (1 us cycle):")
    pmax = {}
    # A serial core handles clock * cycle / CYCLES_PER_OP operations per round.
    for clock in (0.5e9, 1e9, 2e9):
        for d in (11, 17, 25):
            checks = d * d - 1
            budget = clock * STABILIZER_CYCLE_S / CYCLES_PER_OP
            pmax[(clock, d)] = budget / (slope_p * checks)
        log.info(
            "  %.1f GHz: %s",
            clock / 1e9,
            {d: "%.3f" % pmax[(clock, d)] for d in (11, 17, 25)},
        )
    # p_max falls like 1/d^2: a bigger code needs a faster core or more cores.
    r.append(
        check(
            abs(slope([11, 17, 25], [pmax[(1e9, d)] for d in (11, 17, 25)]) + 2) < 0.1,
            "p_max falls as 1/d^2 for a single serial core",
        )
    )
    # Cores needed for a machine of 1000 logical qubits at p = 1e-3, d = 17.
    p_phys, d_log, n_log = 1e-3, 17, 1000
    per_patch = slope_p * p_phys * (d_log * d_log - 1)
    cores = per_patch * n_log * CYCLES_PER_OP / (1e9 * STABILIZER_CYCLE_S)
    per_core = 1e9 * STABILIZER_CYCLE_S / (per_patch * CYCLES_PER_OP)
    log.info(
        "d=17, p=1e-3: %.1f ops/us per patch; 1000 logical qubits need %.0f serial "
        "1 GHz cores (one core serves %.1f patches)",
        per_patch,
        cores,
        per_core,
    )
    # Serial cores are the pessimistic design: hundreds of them, about five
    # patches each. Dedicated parallel hardware (Check 4) needs far fewer cycles.
    r.append(check(3 < per_core < 10, "a serial 1 GHz core serves 3-10 patches"))
    r.append(
        check(100 < cores < 400, "so a 1000-patch machine needs hundreds of cores")
    )
    ctx["slope_p"] = slope_p


def check_memory(r, ctx):
    """Check 6: the state of the decoder fits on chip."""
    log.info("=== Check 6: decoder memory ===")
    # Two distances a factor of 2.3 apart show how the state grows.
    sizes = {}
    for d in (11, 25):
        uf, _ = decoder_for(d)
        n, e = uf.n_nodes, len(uf.eu)
        # Per node: parent pointer, parity bit, boundary bit. Per edge: 2 bits
        # of growth state. (Adjacency is static wiring, not stored state.)
        bits = n * (math.ceil(math.log2(n)) + 2) + 2 * e
        sizes[d] = bits / 8
        log.info("d=%d: %d nodes, %d edges, %.1f KB of state", d, n, e, sizes[d] / 1024)
    ratio = sizes[25] / sizes[11]
    # Volume grows as d^3, with a logarithmic factor from the pointer width.
    r.append(check(10 < ratio < 16, f"state grows ~d^3 log d (x{ratio:.1f})"))
    r.append(check(sizes[25] < 256 * 1024, "d=25 state fits in 256 KB of on-chip SRAM"))


def check_matching_model(r, ctx):
    """Check 7: the textbook matching bound against the measured UF work."""
    log.info("=== Check 7: dense matching bound versus union-find ===")
    table, ds = ctx["table"], ctx["ds"]
    # Ratio of the m^3 bound to the measured UF operations, per (p, d).
    ratios = {}
    for p in (min(ctx["ps"]), 0.01):
        for d in ds:
            m = table[(p, d)]
            # Dense blossom: about m^3 steps for m defects. A model, not data.
            ratios[(p, d)] = float((m["defects"] ** 3).mean() / m["ops"].mean())
        log.info("p=%.3f m^3 / UF ops: %s", p, ["%.1f" % ratios[(p, d)] for d in ds])
    r.append(
        check(
            ratios[(0.01, ds[-1])] > 20, "at p=0.01 the m^3 bound is >20x the UF work"
        )
    )
    r.append(
        check(ratios[(0.01, ds[-1])] > ratios[(0.01, ds[0])], "the gap widens with d")
    )


def main(argv=None):
    """Run every Lab 43 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 43: union-find decoder cost model")
    ap.add_argument("--shots", type=int, default=300, help="shots per (p, d) point")
    # The seed fixes every random error draw.
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fewer shots")
    args = ap.parse_args(argv)
    if args.shots < 50:
        ap.error(f"--shots must be at least 50. Got {args.shots}.")
    ctx = {
        "seed": args.seed,
        "shots": 100 if args.quick else args.shots,
        "ds": (5, 7, 9, 11),
        "ps": (0.002, 0.005, 0.01, 0.04),
        "test": (13, 0.003),
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_instrument(r, ctx)
    check_load(r, ctx)
    check_threshold(r, ctx)
    check_depth(r, ctx)
    check_capacity(r, ctx)
    check_memory(r, ctx)
    check_matching_model(r, ctx)
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
