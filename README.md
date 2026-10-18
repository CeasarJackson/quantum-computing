# Quantum Computing Labs

**50 self-checking labs, from a single qubit to quantum error mitigation and
hardware-style decoder costs, plus a reproducible robustness study of a
variational quantum classifier.**

Every lab is a standalone Python script that *predicts, measures, and checks*:
it prints PASS/FAIL for each claim it makes (about 1,200 checks in total), so
the results below are tested, not just stated.

Author: Dr. Ceasar Jackson Jr.

---

## Highlights

- **Ran on real quantum hardware.** IBM `ibm_fez` (156 qubits): GHZ(3) state
  fidelity 0.940 ± 0.011; a mirror-circuit benchmark measured **0.28% error per
  CNOT**; a Trotterised Ising quench tracked the exact dynamics within 0.048 RMS.
- **Built the simulators from scratch.** A matrix-product-state simulator,
  DMRG (matches exact diagonalisation to 1e-8 and the Bethe-ansatz energy
  density to 4 decimal places), and an adjoint gradient exact to 3e-13.
- **Quantum error correction.** Surface-code thresholds measured at three
  noise levels (9.4% → 2.5% → 0.7%, matching the literature), plus a
  union-find decoder written from scratch and benchmarked against MWPM.
- **Error mitigation, compared side by side.** Zero-noise extrapolation,
  symmetry verification, probabilistic error cancellation and randomised
  compiling, including *where each one fails* (for example, frozen coherent
  errors make straight-line extrapolation worse than doing nothing).
- **A reproducible ML study.** A 4-qubit PennyLane classifier evaluated over
  a partition × initialization grid, with classical baselines, provenance
  hashes and stated limitations (see below).

## The PennyLane study

A four-qubit variational classifier learns 4-bit parity, trained on 12 inputs
and tested on the 4 held-out inputs.

| Phase | Design | Result |
|---|---|---|
| I | 5 partitions × 5 initializations, 200 steps | 23/25 configurations perfect on held-out data; mean 94.0% (SD 20.8). Both failures shared one initialization seed. |
| II | 10 new partitions × 5 new initializations, 200 steps | 44/50 configurations perfect on held-out data; mean 90.0%. **Every failure was a failure to train** (58–67% training accuracy), never to generalize, and all six came from two initialization seeds. 20 configurations re-ran bit for bit. |

**What this does and does not show.** There are only 16 possible inputs, so
each test set has 4 examples and accuracy moves in 25-point steps. The
circuit's CNOT chain is well matched to parity, so high accuracy shows a
matched structure, **not a quantum advantage**. Full write-ups:
[Phase I](docs/pennylane_factorial_study.md) ·
[Phase II](docs/pennylane_phase2_study.md) ([PDF](docs/pennylane_phase2_study.pdf))

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python labs/lab01_single_qubit.py        # any single lab
python labs/run_all.py --skip-slow       # everything, with a summary table
python labs/pennylane_phase2_robustness.py --dry-run   # validate the study design
```

Tested with Python 3.13, Qiskit 2.5, PennyLane 0.45. The real-hardware labs
(11, 15, 17, 44, 46, 48, 49, 50) run in dry-run mode on a simulated device unless given `--live`;
live runs read the IBM API key **only** from an environment variable.

GitHub Actions (`.github/workflows/ci.yml`) installs the pinned requirements and runs
`python labs/run_all.py --skip-slow` on every push and pull request to `main`. It uses no
secrets, so the hardware labs only ever run in dry-run mode there.

## Repository layout

| Path | Contents |
|---|---|
| `labs/lab01_*.py` … `lab50_*.py` | The labs, one topic each (full index: [course-index.md](course-index.md)) |
| `labs/ex01_*.py` … `ex10_*.py` | Optional exercises |
| `labs/mpslib.py`, `labs/dmrgx.py`, `labs/pbmps.py`, `labs/fezdata.py`, `labs/noiselib.py` | Shared MPS/DMRG (with symmetries and excited states), per-bond ansatz, recorded ibm_fez data, and noise/mitigation libraries |
| `labs/PennyLane.py`, `labs/pennylane_*.py` | The classifier and its evaluation and analysis scripts |
| `results/pennylane/` | Raw, unmodified experiment outputs (CSV/JSON) |
| `docs/` | Study write-ups |

## Engineering standards

Every script has a documented header, argparse CLI, exit codes
(0/1/2/130), colored console plus plain-text file logging, and explanatory
comments. `labs/check_standards.py` enforces this on every file, and all code
is `black`-formatted. Experiment outputs are written with exclusive file
creation, so a result can never be silently overwritten.
