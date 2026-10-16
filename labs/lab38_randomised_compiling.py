#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab38_randomised_compiling.py

Purpose
-------
Lab 34 twirled coherent errors with the free symmetry frames P (x) P around the
Heisenberg bond gate. Those frames cannot touch a two-qubit error such as ZZ
crosstalk, because ZZ commutes with every P (x) P. RANDOMISED COMPILING does it
properly at the level of the native two-qubit gate.

Compile each bond gate into three CNOTs plus one-qubit rotations. Around every
CNOT put a random Pauli pair A before and the compensating Pauli B after, where
B CNOT A = CNOT (a CNOT maps Paulis to Paulis, so B is a Pauli too and the
ideal circuit never changes). An error E that follows the CNOT is then seen as
B E B^+ with B uniformly random over all 16 Paulis. Averaged, that is a PAULI
channel: every off-diagonal ("coherent") term of E is gone.

The lab verifies, with exact density matrices on 6 sites and sampled
trajectories on a longer chain:
  1. The compilation is exact (3 CNOTs per bond), the compensation is a Pauli,
     and the ideal circuit is unchanged by the randomisation.
  2. A full twirl turns any error into a Pauli channel; the symmetry frames of
     Lab 34 leave a pure ZZ error completely untouched.
  3. Frozen coherent errors grow like fold^2 under folding; after the full
     twirl they grow like fold^1, like random noise.
  4. So a straight line then works (zero-noise extrapolation) for ZZ
     crosstalk, a one-qubit rotation and both together; without the twirl the
     line makes it WORSE, and the symmetry frames fix only the one-qubit part.
  5. The sampled sampler reproduces the exact channel.
  6. On the longer chain the single-run bias is NOT smaller after twirling
     (the gain is entirely in the scaling), and a line through folds 1 and 3
     recovers the clean energy.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab38_randomised_compiling.py
    python3 labs/lab38_randomised_compiling.py --trajectories 120
    python3 labs/lab38_randomised_compiling.py --quick

Log files
---------
logs/lab38/lab38_<YYYYmmdd_HHMMSS>.log

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
from mpslib import bond_unitary, run_hva, sparse_hamiltonian, train_angles
from noiselib import (
    PAULI,
    cnot_error,
    compiled_bond_steps,
    compiled_mps,
    dm_compiled_state,
    dm_expectation,
    mean_se,
    richardson_weights,
)

log = get_logger("lab38")

FOLDS = (1, 3, 5)
# Three error kinds: ZZ crosstalk, a one-qubit X rotation, and both.
KINDS = {"ZZ crosstalk": (0.02, 0.0), "one-qubit X": (0.0, 0.02), "both": (0.02, 0.02)}


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def exponent(values):
    """Fitted power b in |bias| ~ fold^b over FOLDS."""
    return float(np.polyfit(np.log(FOLDS), np.log(np.abs(values)), 1)[0])


def biases(n, params, clean, zz, xi, mode):
    """Exact energy bias at every fold for one error kind and twirling mode."""
    # One exact density matrix per fold; the folded circuit repeats every gate,
    # so a frozen error adds up coherently with the fold factor.
    return np.array(
        [
            dm_expectation(dm_compiled_state(n, params, zz, xi, mode, f), n) - clean
            for f in FOLDS
        ]
    )


def pauli_pairs():
    """The sixteen two-qubit Paulis."""
    return [np.kron(PAULI[a], PAULI[b]) for a in "IXYZ" for b in "IXYZ"]


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_machinery(r, ctx):
    """Check 1: the compilation and the compensation."""
    log.info("=== Check 1: the CNOT-level compilation ===")
    # Negative and large angles matter: folding uses U(-a) as well as U(a).
    for angle in (0.1, 0.7, -0.4, 1.3):
        mat = np.eye(4, dtype=complex)
        steps = compiled_bond_steps(angle)
        for _, gate in steps:
            mat = gate @ mat
        # Fidelity |tr(U^+ V)| / 4 equals 1 exactly when they differ by a phase.
        fid = abs(np.trace(bond_unitary(angle).conj().T @ mat)) / 4
        r.append(
            check(abs(fid - 1) < 1e-12, f"angle {angle:+.1f}: compiled = bond gate")
        )
    # Three is the minimum for a general two-qubit gate; fewer cannot do it.
    r.append(
        check(
            sum(kind == "C" for kind, _ in compiled_bond_steps(0.3)) == 3,
            "each bond gate uses exactly three CNOTs",
        )
    )
    # Every CNOT maps a Pauli pair A to another Pauli pair B (up to a sign).
    cnot = [g for k, g in compiled_bond_steps(0.3) if k == "C"][0]
    # The Clifford property: a CNOT sends every Pauli product to a Pauli product.
    pairs = pauli_pairs()
    ok = True
    for a in pairs:
        b = cnot @ a.conj().T @ cnot.conj().T
        # Exactly one Pauli has overlap magnitude one with b.
        ok &= any(abs(np.trace(p.conj().T @ b)) / 4 > 1 - 1e-12 for p in pairs)
    r.append(check(ok, "the compensation B = CNOT A^+ CNOT^+ is always a Pauli"))
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    # With the errors off, the random Paulis must cancel exactly in both modes.
    for mode in ("full", "sym"):
        e = dm_expectation(dm_compiled_state(n, params, 0.0, 0.0, mode), n)
        r.append(
            check(
                abs(e - clean) < 1e-10, f"{mode} twirl leaves the ideal circuit alone"
            )
        )
    # The same statement for one sampled trajectory.
    rng = np.random.default_rng(ctx["seed"])
    e = compiled_mps(n, params, 64, 0.0, 0.0, rng, "full").energy()
    r.append(
        check(abs(e - clean) < 1e-9, "the sampled randomised circuit is ideal too")
    )


def check_channel(r, ctx):
    """Check 2: the full twirl gives a Pauli channel; frames miss pure ZZ."""
    log.info("=== Check 2: what the twirl does to one error ===")
    pairs = pauli_pairs()
    # A deliberately large error, so that rounding cannot hide a wrong identity.
    err = cnot_error(0.2, 0.3)
    # A random pure two-qubit state: the identity must hold for every input.
    rng = np.random.default_rng(ctx["seed"] + 1)
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    v /= np.linalg.norm(v)
    rho = np.outer(v, v.conj())
    twirled = sum(
        p @ err @ p.conj().T @ rho @ (p @ err @ p.conj().T).conj().T for p in pairs
    )
    twirled = twirled / 16
    # (The process matrix chi is diagonal after the twirl; its diagonal is p_P.)
    # Pauli-channel weight of each Pauli: the diagonal of the process matrix.
    weights = [abs(np.trace(p.conj().T @ err) / 4) ** 2 for p in pairs]
    pauli = sum(w * p @ rho @ p.conj().T for w, p in zip(weights, pairs))
    r.append(check(abs(sum(weights) - 1) < 1e-12, "the Pauli weights sum to one"))
    r.append(
        check(np.allclose(twirled, pauli, atol=1e-12), "full twirl = Pauli channel")
    )
    # The coherent channel keeps its off-diagonal terms: it differs from the Pauli one.
    plain = err @ rho @ err.conj().T
    r.append(
        check(
            not np.allclose(plain, pauli, atol=1e-3), "the untwirled error is not one"
        )
    )
    # The Lab 34 frames P (x) P commute with ZZ, so conjugating changes nothing.
    # Pure ZZ: the case the symmetry frames of Lab 34 cannot reach.
    zz = cnot_error(0.2, 0.0)
    frames = [np.kron(PAULI[k], PAULI[k]) for k in "IXYZ"]
    same = all(np.allclose(f @ zz @ f.conj().T, zz, atol=1e-12) for f in frames)
    r.append(check(same, "symmetry frames leave a pure ZZ error exactly as it was"))
    flips = [np.allclose(p @ zz @ p.conj().T, zz, atol=1e-12) for p in pairs]
    r.append(check(sum(flips) < 16, "the full set of Paulis does change it"))


def check_scaling(r, ctx):
    """Check 3 and 4: fold scaling and extrapolation, exact, n=6."""
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    # Lagrange weights for a line (folds 1, 3) and a parabola (folds 1, 3, 5).
    w2, w3 = richardson_weights(FOLDS[:2]), richardson_weights(FOLDS)
    log.info("=== Check 3: fold scaling and extrapolation (exact, n=6) ===")
    # Results keyed by (error kind, twirling mode).
    out = {}
    for kind, (zz, xi) in KINDS.items():
        for mode in ("none", "sym", "full"):
            v = biases(n, params, clean, zz, xi, mode)
            out[kind, mode] = (v, float(w2 @ v[:2]), float(w3 @ v))
            log.info(
                "%-13s %-4s bias %s  line %+.1e  parabola %+.1e",
                kind,
                mode,
                np.round(v, 5),
                out[kind, mode][1],
                out[kind, mode][2],
            )
    for kind in KINDS:
        v0, lin0, _ = out[kind, "none"]
        v2, lin2, quad2 = out[kind, "full"]
        r.append(
            check(
                exponent(v0) > 1.6,
                f"{kind}: frozen error grows like fold^{exponent(v0):.2f}",
            )
        )
        r.append(
            check(
                0.9 < exponent(v2) < 1.1,
                f"{kind}: twirled error grows like fold^{exponent(v2):.2f}",
            )
        )
        r.append(
            check(abs(lin0) > abs(v0[0]), f"{kind}: without twirling the line is WORSE")
        )
        r.append(
            check(abs(lin2) < 0.03 * abs(v2[0]), f"{kind}: twirled + line removes >97%")
        )
        r.append(
            check(
                abs(quad2) < 0.005 * abs(v2[0]),
                f"{kind}: twirled + parabola removes >99.5%",
            )
        )
    # The symmetry frames only twirl one-qubit errors, so they split the kinds.
    v_x, lin_x, _ = out["one-qubit X", "sym"]
    v_z, lin_z, _ = out["ZZ crosstalk", "sym"]
    r.append(
        check(
            abs(lin_x) < 0.05 * abs(v_x[0]),
            "symmetry frames + line fix the one-qubit error",
        )
    )
    r.append(check(abs(lin_z) > 0.5 * abs(v_z[0]), "but not the ZZ crosstalk"))
    ctx["exact_none"] = out["both", "none"][0]


def check_sampler(r, ctx):
    """Check 4: the sampled randomised circuit reproduces the exact channel."""
    n, params, clean = 6, ctx["params6"], ctx["clean6"]
    zz, xi = KINDS["both"]
    count = ctx["small"]
    log.info("=== Check 4: sampler against the exact channel, N=%d ===", count)
    # Ground truth: the exact average over all 16 Paulis at every CNOT.
    exact = dm_expectation(dm_compiled_state(n, params, zz, xi, "full"), n) - clean
    rng = np.random.default_rng(ctx["seed"] + 4)
    vals = [
        compiled_mps(n, params, 64, zz, xi, rng, "full").energy() for _ in range(count)
    ]
    mean, se = mean_se(np.array(vals) - clean)
    log.info("exact %+.5f | sampled %+.5f +- %.5f", exact, mean, se)
    r.append(
        check(
            abs(mean - exact) < 4 * se, "sampled mean equals the exact twirled channel"
        )
    )
    # Without twirling the MPS run is deterministic and must equal its density matrix.
    none = compiled_mps(n, params, 64, zz, xi, rng, "none").energy() - clean
    plain = dm_expectation(dm_compiled_state(n, params, zz, xi, "none"), n) - clean
    r.append(
        check(
            abs(none - plain) < 1e-9,
            "and the frozen-error MPS run equals its density matrix",
        )
    )


def check_long_chain(r, ctx):
    """Check 5: the longer chain, folds 1 and 3."""
    n, count = ctx["n"], ctx["count"]
    zz, xi = ctx["rate"], ctx["rate"]
    params = ctx["params8"]
    clean = run_hva(n, params, 32).energy()
    log.info("=== Check 5: n=%d, errors %.3f, N=%d, folds 1 and 3 ===", n, zz, count)
    rng = np.random.default_rng(ctx["seed"] + 5)
    # Frozen errors need no sampling: one deterministic run per fold.
    frozen = [
        compiled_mps(n, params, 32, zz, xi, rng, "none", f).energy() - clean
        for f in (1, 3)
    ]
    # A line through folds 1 and 3 has weights 1.5 and -0.5.
    line_frozen = 1.5 * frozen[0] - 0.5 * frozen[1]
    log.info("frozen: %+.4f %+.4f -> line %+.4f", frozen[0], frozen[1], line_frozen)
    # Twirled runs are random: average many and keep the standard error.
    twirled = []
    for fold in (1, 3):
        vals = np.array(
            [
                compiled_mps(n, params, 32, zz, xi, rng, "full", fold).energy()
                for _ in range(count)
            ]
        )
        twirled.append(mean_se(vals - clean))
        log.info("twirled fold %d: %+.4f +- %.4f", fold, *twirled[-1])
    (b1, s1), (b3, s3) = twirled
    line = 1.5 * b1 - 0.5 * b3
    # Independent errors add in quadrature after weighting.
    se_line = float(np.hypot(1.5 * s1, 0.5 * s3))
    log.info("twirled line: %+.4f +- %.4f", line, se_line)
    r.append(
        check(frozen[1] / frozen[0] > 4, "frozen errors grow faster than linearly")
    )
    r.append(
        check(abs(line_frozen) > abs(frozen[0]), "and the line on them overshoots")
    )
    r.append(
        check(
            2.2 < b3 / b1 < 3.8,
            f"twirled bias scales like the fold ({b3 / b1:.2f}x for 3x)",
        )
    )
    r.append(
        check(
            abs(line) < 4 * se_line + 0.1 * abs(b1),
            "twirled line recovers the clean energy",
        )
    )
    r.append(
        check(
            abs(line) < 0.5 * abs(line_frozen),
            "at least 2x closer than the frozen line",
        )
    )
    log.info(
        "single-run bias: twirled %+.4f vs frozen %+.4f (twirling alone does not help)",
        b1,
        frozen[0],
    )


def main(argv=None):
    """Run every Lab 38 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 38: randomised compiling")
    ap.add_argument("--trajectories", type=int, default=60, help="long chain")
    ap.add_argument("--rate", type=float, default=0.01, help="coherent error angle")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if args.trajectories < 30:
        ap.error(f"--trajectories must be at least 30. Got {args.trajectories}.")
    if not 0.002 <= args.rate <= 0.03:
        ap.error(f"--rate must be between 0.002 and 0.03. Got {args.rate}.")
    quick = args.quick
    # Quick mode shortens the chain; the angles still come from 8-site training.
    ctx = {
        "seed": args.seed,
        "n": 10 if quick else 12,
        "count": 40 if quick else args.trajectories,
        "rate": args.rate,
        "small": 120 if quick else 300,
    }
    # Angles are trained on small chains, exactly as in Labs 28-37.
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
    ctx["clean6"] = run_hva(6, ctx["params6"], 64).energy()
    # r collects every check's boolean so the exit code reflects all of them.
    # Order matters: the machinery check comes first so later failures are meaningful.
    r = []
    check_machinery(r, ctx)
    check_channel(r, ctx)
    check_scaling(r, ctx)
    check_sampler(r, ctx)
    check_long_chain(r, ctx)
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
