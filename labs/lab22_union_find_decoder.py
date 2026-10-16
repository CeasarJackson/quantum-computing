#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab22_union_find_decoder.py

Purpose
-------
Labs 12 and 18 decoded the surface code with minimum-weight perfect matching
(MWPM) from the pymatching package. MWPM is accurate but its worst-case cost
grows quickly, which is why real-time decoders for large machines look
elsewhere. Lab 22 writes the best-known alternative from scratch: the
UNION-FIND decoder (Delfosse and Nickerson), which runs in almost-linear time.

How it works, in four steps:
  1. Every syndrome defect starts as a tiny cluster.
  2. Each odd cluster (one that has not yet found a partner or the boundary)
     grows outward by half an edge; clusters that touch merge.
  3. Growth stops when every cluster is even or touches the boundary.
  4. PEELING: inside each cluster build a spanning tree and walk it from the
     leaves, flipping the edge above any node that still holds a defect.

The decoder works on any matching graph, so the same code serves the 2D
data-error problem of Lab 12 and the 3D space-time problem of Lab 18 (the
graphs are taken from pymatching, but the decoding is entirely our own).

The lab verifies:
  1. Self-consistency: on thousands of random syndromes, the correction ALWAYS
     reproduces the syndrome it was given (so it is a valid correction).
  2. Guarantee: every error of weight <= (d-1)/2 is corrected (all of them, by
     brute force on d=3 and d=5).
  3. Accuracy: UF fails somewhat more often than MWPM, never less (beyond
     noise), and its threshold is slightly lower.
  4. Same story in space-time with faulty measurements (Lab 18's model).
  5. Cost: our pure-Python UF versus the compiled MWPM, for information.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab22_union_find_decoder.py
    python3 labs/lab22_union_find_decoder.py --shots 4000 --seed 3
    python3 labs/lab22_union_find_decoder.py --quick

Log files
---------
logs/lab22/lab22_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (pymatching).
130 User interrupted execution.
===============================================================================
"""

import argparse
import itertools
import math
import sys
import time

import numpy as np

from applog import get_logger

log = get_logger("lab22")

try:
    import pymatching
except ImportError:  # pragma: no cover
    log.fail("pymatching is not installed: pip install pymatching")
    sys.exit(2)

DISTANCES = (3, 5, 7)
P_2D = (0.06, 0.08, 0.09, 0.10, 0.11, 0.12)
P_3D = (0.015, 0.02, 0.025, 0.03, 0.035)


# ----------------------------------------------------------------------------
# Code construction (same rotated surface code as Labs 12 and 18)
# ----------------------------------------------------------------------------
def build_code(d):
    """Rotated surface code: (HX, HZ) check matrices over d*d data qubits."""
    # Plaquettes sit on the corners of the data grid, (i, j) in 0..d. Their
    # type alternates like a checkerboard. Bulk plaquettes touch four data
    # qubits; the boundary keeps only every other one (weight 2), with Z-type
    # on the top/bottom edges and X-type on the left/right edges.
    xs, zs = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            qubits = [
                a * d + b
                for a, b in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j))
                if 0 <= a < d and 0 <= b < d
            ]
            kind = "X" if (i + j) % 2 == 0 else "Z"
            bulk = 1 <= i <= d - 1 and 1 <= j <= d - 1
            if bulk:
                keep = True
            elif i in (0, d) and 1 <= j <= d - 1:
                keep = kind == "Z"
            elif j in (0, d) and 1 <= i <= d - 1:
                keep = kind == "X"
            else:
                keep = False  # corners host no check
            if keep:
                (xs if kind == "X" else zs).append(qubits)
    return to_matrix(xs, d * d), to_matrix(zs, d * d)


def to_matrix(checks, n):
    """List of qubit-index lists -> binary check matrix (one row per check)."""
    # Dense check matrix is fine at these sizes and keeps the code simple.
    h = np.zeros((len(checks), n), dtype=np.uint8)
    for row, qubits in enumerate(checks):
        h[row, qubits] = 1
    return h


def logicals(d):
    """Logical X (a row of X's) and logical Z (a column of Z's), as bit vectors."""
    # Logical operators run across the lattice along one edge.
    lx = np.zeros(d * d, dtype=np.uint8)
    lz = np.zeros(d * d, dtype=np.uint8)
    # A logical X runs along a full row; any row is equivalent.
    lx[0:d] = 1  # top row
    lz[0::d] = 1  # left column
    return lx, lz


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def stderr(x, shots):
    """Binomial standard error of a measured rate."""
    return math.sqrt(max(x, 1e-12) * (1 - x) / shots)


# ----------------------------------------------------------------------------
# The union-find decoder
# ----------------------------------------------------------------------------
class UnionFind:
    """Union-find decoder for a matching graph with one boundary node."""

    def __init__(self, matching):
        """Copy the graph out of a pymatching object (nodes, edges, faults)."""
        self.n_det = matching.num_detectors
        self.boundary = self.n_det  # index of the boundary node
        self.n_nodes = self.n_det + 1
        self.eu, self.ev, self.faults = [], [], []
        self.adj = [[] for _ in range(self.n_nodes)]
        for u, v, attr in matching.edges():
            # pymatching marks the boundary with None or with an index >= n_det.
            u = self.boundary if u is None or u >= self.n_det else u
            v = self.boundary if v is None or v >= self.n_det else v
            idx = len(self.eu)
            self.eu.append(u)
            self.ev.append(v)
            self.faults.append(sorted(attr.get("fault_ids", set())))
            self.adj[u].append(idx)
            self.adj[v].append(idx)
        self.n_faults = 1 + max((f for fs in self.faults for f in fs), default=-1)

    def decode(self, defects):
        """Return (flipped edge list, correction vector) for a defect list."""
        n = self.n_nodes
        parent = list(range(n))
        parity = [0] * n
        touches = [False] * n  # cluster contains the boundary
        members = [[i] for i in range(n)]
        touches[self.boundary] = True
        for d in defects:
            parity[d] = 1
        growth = {}  # edge index -> half-edges grown
        full = set()  # fully grown edges

        def find(x):
            """Root of x's cluster, with path compression."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            """Merge two clusters, keeping the larger member list."""
            # Merge two clusters; find() compresses paths, so lookups stay cheap.
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if len(members[ra]) < len(members[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            parity[ra] ^= parity[rb]
            touches[ra] = touches[ra] or touches[rb]
            members[ra].extend(members[rb])
            members[rb] = []

        # Growth phase: odd clusters that do not touch the boundary grow.
        while True:
            active = {find(d) for d in defects}
            active = [c for c in active if parity[c] and not touches[c]]
            if not active:
                break
            newly = []
            for c in active:
                for node in members[c]:
                    for e in self.adj[node]:
                        if e in full:
                            continue
                        growth[e] = growth.get(e, 0) + 1
                        if growth[e] >= 2:
                            newly.append(e)
            # Merge only after all growth of this round, so order cannot matter.
            for e in newly:
                if e not in full:
                    full.add(e)
                    union(self.eu[e], self.ev[e])
        return self._peel(full, defects)

    def _peel(self, full, defects):
        """Peel spanning forests of the grown edges to get a correction."""
        # syn[node] = 1 while that node still holds an unmatched defect.
        syn = {}
        for d in defects:
            syn[d] = 1
        # Adjacency of the fully grown edges only: this is the cluster graph.
        nbrs = {}
        for e in full:
            nbrs.setdefault(self.eu[e], []).append((self.ev[e], e))
            nbrs.setdefault(self.ev[e], []).append((self.eu[e], e))
        seen, flips = set(), []
        # Start from the boundary so that component absorbs odd parity.
        roots = ([self.boundary] if self.boundary in nbrs else []) + list(nbrs)
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            order, tree_edge = [root], {}
            for x in order:  # breadth-first spanning tree
                for y, e in nbrs.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        tree_edge[y] = (x, e)
                        order.append(y)
            for x in reversed(order[1:]):  # leaves first
                if syn.get(x, 0):
                    p, e = tree_edge[x]
                    flips.append(e)
                    syn[p] = syn.get(p, 0) ^ 1
                    syn[x] = 0
        # Translate flipped edges into flipped data qubits (fault ids).
        corr = np.zeros(self.n_faults, dtype=np.uint8)
        for e in flips:
            for f in self.faults[e]:
                corr[f] ^= 1
        return flips, corr

    def consistent(self, flips, defects):
        """True if the flipped edges have exactly the given defects as ends."""
        # Parity of how many flipped edges end at each node.
        count = {}
        for e in flips:
            for node in (self.eu[e], self.ev[e]):
                if node != self.boundary:
                    count[node] = count.get(node, 0) ^ 1
        return {k for k, v in count.items() if v} == set(defects)


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def fail_rate_2d(d, p, shots, rng, decoder):
    """Logical failure rate for data errors only ('uf' or 'mwpm')."""
    # X errors are caught by the Z checks; failure = residual flips logical Z.
    hz, lz = build_code(d)[1], logicals(d)[1]
    errors = (rng.random((shots, d * d)) < p).astype(np.uint8)
    syn = (errors @ hz.T) % 2
    if decoder == "mwpm":
        corr = pymatching.Matching(hz).decode_batch(syn)
    else:
        uf = UnionFind(pymatching.Matching(hz))
        corr = np.array([uf.decode(np.flatnonzero(s))[1] for s in syn])
    return float((((errors ^ corr) @ lz) % 2).mean())


def fail_rate_3d(d, p, shots, rng, decoder):
    """Logical failure rate with data errors and measurement errors (p = q)."""
    # Same decoding loop as 2D but the detectors include time.
    hz, lz = build_code(d)[1], logicals(d)[1]
    n, c = d * d, hz.shape[0]
    data = (rng.random((shots, d, n)) < p).astype(np.uint8)
    meas = (rng.random((shots, d, c)) < p).astype(np.uint8)
    # Errors accumulate over rounds; each round's syndrome may be misread.
    total = np.cumsum(data, axis=1) % 2
    syn = ((total @ hz.T) % 2) ^ meas
    syn = np.concatenate([syn, ((total[:, -1] @ hz.T) % 2)[:, None, :]], axis=1)
    # Detection events = syndrome changes between consecutive rounds.
    events = syn.copy()
    events[:, 1:] ^= syn[:, :-1]
    events = events.reshape(shots, -1).astype(np.uint8)
    graph = pymatching.Matching.from_check_matrix(
        hz, weights=1.0, repetitions=d + 1, timelike_weights=1.0
    )
    if decoder == "mwpm":
        corr = graph.decode_batch(events)
    else:
        uf = UnionFind(graph)
        corr = np.array([uf.decode(np.flatnonzero(e))[1] for e in events])
    return float((((total[:, -1] ^ corr) @ lz) % 2).mean())


def crossing(ps, small, large):
    """Linear-interpolated p where the small-d and large-d curves cross."""
    # Sign change of (small - large) marks the crossing; interpolate linearly.
    diff = [a - b for a, b in zip(small, large)]
    for k in range(len(ps) - 1):
        if diff[k] > 0 >= diff[k + 1]:
            return ps[k] + diff[k] / (diff[k] - diff[k + 1]) * (ps[k + 1] - ps[k])
    return None


def check_validity(r, rng):
    """Checks 1 and 2: valid corrections and the distance guarantee."""
    log.info("=== Check 1: the correction always reproduces the syndrome ===")
    for d in (3, 5, 7):
        hz = build_code(d)[1]
        uf = UnionFind(pymatching.Matching(hz))
        bad = 0
        for _ in range(3000):
            # Dense random errors, so clusters merge and boundaries are hit.
            e = (rng.random(d * d) < 0.12).astype(np.uint8)
            defects = np.flatnonzero((hz @ e) % 2)
            flips, corr = uf.decode(defects)
            if not uf.consistent(flips, defects) or not np.array_equal(
                (hz @ corr) % 2, (hz @ e) % 2
            ):
                bad += 1
        r.append(
            check(
                bad == 0, f"d={d}: 3000 random syndromes, " f"{bad} invalid corrections"
            )
        )
    log.info("=== Check 2: every error of weight <= (d-1)/2 is corrected ===")
    for d in (3, 5):
        hz, lz = build_code(d)[1], logicals(d)[1]
        uf = UnionFind(pymatching.Matching(hz))
        t = (d - 1) // 2
        total = failed = 0
        for w in range(1, t + 1):
            for combo in itertools.combinations(range(d * d), w):
                e = np.zeros(d * d, dtype=np.uint8)
                e[list(combo)] = 1
                corr = uf.decode(np.flatnonzero((hz @ e) % 2))[1]
                total += 1
                failed += int(((e ^ corr) @ lz) % 2)
        r.append(
            check(
                failed == 0, f"d={d}: all {total} errors of weight " f"<= {t} corrected"
            )
        )
    # Weight (d+1)/2 is where a decoder is allowed to fail.
    d = 3
    hz, lz = build_code(d)[1], logicals(d)[1]
    uf = UnionFind(pymatching.Matching(hz))
    fails = 0
    for combo in itertools.combinations(range(9), 2):
        e = np.zeros(9, dtype=np.uint8)
        e[list(combo)] = 1
        fails += int(((e ^ uf.decode(np.flatnonzero((hz @ e) % 2))[1]) @ lz) % 2)
    log.info("d=3 weight-2 errors defeating UF: %d of 36 (MWPM: 18 in Lab 12)", fails)
    r.append(
        check(
            fails > 0, "weight-2 errors can defeat distance 3, as they " "do for MWPM"
        )
    )


def check_accuracy(r, shots, rng):
    """Checks 3 and 4: UF versus MWPM in 2D and in space-time."""
    # Two settings: Lab 12's data-error model and Lab 18's space-time model.
    results = {}
    for tag, grid, fn, lit, lo, hi in (
        ("2D data errors", P_2D, fail_rate_2d, "MWPM ~0.094 here", 0.07, 0.11),
        (
            "3D with faulty measurements",
            P_3D,
            fail_rate_3d,
            "MWPM ~0.025 here",
            0.015,
            0.035,
        ),
    ):
        log.info("=== Check: %s, %d shots per point ===", tag, shots)
        # rates[(d, p, decoder)] = measured logical failure rate.
        rates = {}
        for d in DISTANCES:
            for p in grid:
                # Same random seed for both decoders: they see the same noise.
                seed = int(rng.integers(1 << 30))
                rates[d, p, "uf"] = fn(d, p, shots, np.random.default_rng(seed), "uf")
                rates[d, p, "mwpm"] = fn(
                    d, p, shots, np.random.default_rng(seed), "mwpm"
                )
        for p in grid:
            log.info(
                "p=%.3f: d=3 UF %.4f / MWPM %.4f   d=7 UF %.4f / MWPM %.4f",
                p,
                rates[3, p, "uf"],
                rates[3, p, "mwpm"],
                rates[7, p, "uf"],
                rates[7, p, "mwpm"],
            )
        # Per-point comparison: UF must not beat MWPM beyond sampling noise.
        worse = [
            (rates[d, p, "uf"] - rates[d, p, "mwpm"]) for d in DISTANCES for p in grid
        ]
        sig = [
            3 * stderr(max(rates[d, p, "mwpm"], 1e-4), shots) * math.sqrt(2)
            for d in DISTANCES
            for p in grid
        ]
        r.append(
            check(
                all(w > -s for w, s in zip(worse, sig)),
                f"{tag}: UF never beats MWPM beyond noise",
            )
        )
        r.append(
            check(
                float(np.mean(worse)) > 0,
                f"{tag}: on average UF is a little worse "
                f"(+{float(np.mean(worse)):.4f} absolute)",
            )
        )
        cu = crossing(
            grid, [rates[3, p, "uf"] for p in grid], [rates[7, p, "uf"] for p in grid]
        )
        cm = crossing(
            grid,
            [rates[3, p, "mwpm"] for p in grid],
            [rates[7, p, "mwpm"] for p in grid],
        )
        log.info(
            "d=3/d=7 crossing: UF %s, MWPM %s (%s)",
            f"{cu:.4f}" if cu else "n/a",
            f"{cm:.4f}" if cm else "n/a",
            lit,
        )
        r.append(
            check(
                cu is not None and cm is not None and lo < cu < hi and cu <= cm + 0.005,
                f"{tag}: UF threshold {cu if cu else float('nan'):.3f} "
                f"lies in {lo}-{hi} and is not above MWPM's",
            )
        )
        results[tag] = rates
    return results


def check_speed(r, rng):
    """Check 5: timing, for information."""
    log.info("=== Check 5: cost per decoded shot (d=7, 2D, p=0.08) ===")
    d, p, shots = 7, 0.08, 2000
    hz = build_code(d)[1]
    errors = (rng.random((shots, d * d)) < p).astype(np.uint8)
    syn = (errors @ hz.T) % 2
    t0 = time.perf_counter()
    pymatching.Matching(hz).decode_batch(syn)
    t_m = (time.perf_counter() - t0) / shots
    uf = UnionFind(pymatching.Matching(hz))
    t0 = time.perf_counter()
    for s in syn:
        uf.decode(np.flatnonzero(s))
    t_u = (time.perf_counter() - t0) / shots
    log.info(
        "MWPM (compiled C++): %.1f us/shot; UF (pure Python): %.1f us/shot",
        1e6 * t_m,
        1e6 * t_u,
    )
    log.info(
        "UF's advantage is its scaling and simple hardware mapping, not "
        "Python speed: a compiled or FPGA version is what real systems use."
    )
    r.append(check(t_u > 0 and t_m > 0, "timings measured"))


def main(argv=None):
    """Run every Lab 22 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 22: union-find decoder")
    ap.add_argument(
        "--shots", type=int, default=6000, help="experiments per data point"
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="use 2000 shots")
    args = ap.parse_args(argv)
    if args.shots < 500:
        ap.error(f"--shots must be >= 500. Got {args.shots}.")
    # --quick trades statistical power for speed; checks use wide margins.
    shots = 2000 if args.quick else args.shots
    rng = np.random.default_rng(args.seed)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_validity(r, rng)
    check_accuracy(r, shots, rng)
    check_speed(r, rng)
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
