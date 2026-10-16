# Optimization Failure, Not Overfitting

## Phase II Robustness Study of a Four-Qubit Variational Parity Classifier

**Research study:** PennyLane Phase II, partition × initialization robustness

**Experiment date:** October 16, 2026

**Author:** Dr. Ceasar Jackson Jr.

**Status:** Completed experimental study; descriptive analysis

---

## Abstract

This study extends the Phase I factorial evaluation of a four-qubit variational quantum classifier trained on 4-bit parity. Ten training partitions, each validated as unique before training began, were crossed with five fresh parameter-initialization seeds, giving 50 configurations.

The classifier achieved perfect held-out accuracy in 44 of 50 configurations. Mean held-out accuracy was 90.0% (configuration-level SD 27.7 points; partition-level SD 8.8 points).

All six lower-performing configurations also failed to fit their own training data (58–67% training accuracy), and every configuration that reached 100% training accuracy also reached 100% held-out accuracy. In this design, failures are therefore failures of optimization, not of generalization. The six failures came from two of the five initialization seeds (seed 200: 4 of 10 partitions; seed 203: 2 of 10), and every partition succeeded with at least four of its five seeds.

Twenty configurations were run twice, in two independent executions, and reproduced bit for bit.

These results characterize this specific model and design. They do not demonstrate quantum computational advantage.

---

## 1. Research Objectives

Phase I (25 configurations) found two lower-performing runs, both using one initialization seed, and could not say whether that pattern reflected the seed, the partition, or chance. Phase II asked:

1. Does near-perfect held-out accuracy persist across a larger set of new, validated partitions and new initialization seeds?
2. Do failures cluster by initialization, by partition, or neither?
3. When the classifier fails, does it fail to *train* or fail to *generalize*?
4. Are the results reproducible across independent executions?

## 2. Experimental Methodology

| Parameter | Configuration |
|---|---|
| Quantum model | Four-qubit variational classifier (`labs/PennyLane.py`, unchanged from Phase I) |
| Classification task | Binary parity of 4 bits |
| Input combinations | 16 |
| Training / held-out examples per partition | 12 / 4 (disjoint) |
| Partition seeds | 100–109 (new; none used in Phase I) |
| Initialization seeds | 200–204 (new; none used in Phase I) |
| Optimization steps | 200 |
| Learning rate | 0.1 |
| Ridge parameter (baselines) | 1.0 |
| Configurations | 50 |
| Bootstrap | 2,000 cluster replicates over partitions, seed 20261015 |
| Python / PennyLane / NumPy | 3.13.16 / 0.45.1 / 2.5.4 |
| Platform | macOS 27.0.1, Apple silicon |

**Pre-training validation.** Before any training, the runner checked every partition: 12 and 4 unique inputs, no overlap, together covering all 16 inputs, parity labels recomputed independently, and no two seeds producing the same held-out set. A failed check aborts the study before any result is written.

**Record keeping.** Each configuration's result was written to its own file with exclusive creation as soon as it finished, so no result can be overwritten or silently re-run. The run recorded the SHA-256 of both Phase I source files it depends on.

## 3. Results

### 3.1 Held-out accuracy matrix

| Partition | Init 200 | Init 201 | Init 202 | Init 203 | Init 204 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 100% | 100% | 100% | 100% | 100% | 100% |
| 101 | **0%** | 100% | 100% | 100% | 100% | 80% |
| 102 | **25%** | 100% | 100% | 100% | 100% | 85% |
| 103 | **25%** | 100% | 100% | 100% | 100% | 85% |
| 104 | 100% | 100% | 100% | 100% | 100% | 100% |
| 105 | 100% | 100% | 100% | **0%** | 100% | 80% |
| 106 | 100% | 100% | 100% | 100% | 100% | 100% |
| 107 | 100% | 100% | 100% | 100% | 100% | 100% |
| 108 | 100% | 100% | 100% | **25%** | 100% | 85% |
| 109 | **25%** | 100% | 100% | 100% | 100% | 85% |

### 3.2 Aggregate statistics

| Metric | Result |
|---|---:|
| Configurations | 50 |
| Perfect held-out configurations | 44/50 |
| Mean held-out accuracy | 90.0% |
| Held-out accuracy SD, across configurations | 27.7 points |
| Held-out accuracy SD, across partition means | 8.8 points |
| Partition-mean range | 80% – 100% |
| Cluster-bootstrap 95% percentile interval (partition means) | 85.0% – 95.0% |
| Mean training accuracy | 95.3% |
| Mean training / held-out MSE | 0.122 / 0.206 |

The bootstrap interval resamples the ten partition means. It describes variation across *these selected, overlapping* partitions; it is **not** a confidence interval for performance on new data.

### 3.3 Classical baselines (same partitions)

| Baseline | Mean training accuracy | Mean held-out accuracy |
|---|---:|---:|
| Variational quantum classifier | 95.3% | **90.0%** |
| Majority class | 55.0% | 35.0% |
| Ridge-linear | 60.8% | 20.0% |
| Full polynomial ridge | 100% | 50.0%* |
| 1-nearest neighbour | 100% | 0.0% |

\*Abstention-adjusted score: a near-zero prediction earns half credit. The polynomial model scores exactly 50% on every partition because it abstains on every held-out input; this is not conventional accuracy.

Parity is solvable exactly by a trivial classical rule (sum the bits mod 2). These baselines were chosen to show that generic learners do not *discover* parity from 12 examples, not to represent the best classical method. Nearest neighbour scores 0% by construction: every held-out input's nearest neighbour differs by one bit, which flips parity.

## 4. Failure Analysis

### 4.1 Every failure is an optimization failure

| Partition | Init | Training accuracy | Held-out accuracy | Training MSE | Held-out MSE |
|---|---|---:|---:|---:|---:|
| 101 | 200 | 66.7% | 0.0% | 0.888 | 1.775 |
| 102 | 200 | 58.3% | 25.0% | 0.887 | 1.480 |
| 103 | 200 | 58.3% | 25.0% | 0.942 | 1.381 |
| 105 | 203 | 66.7% | 0.0% | 0.842 | 1.998 |
| 108 | 203 | 58.3% | 25.0% | 0.918 | 1.621 |
| 109 | 200 | 58.3% | 25.0% | 0.952 | 1.192 |

The split is exact: the 44 configurations with 100% training accuracy are the 44 with 100% held-out accuracy, and the 6 that missed on held-out data all missed badly on training data too. No configuration fit its training set and then failed on held-out inputs.

**Observed finding:** within this design, held-out success is fully predicted by training success.

**Interpretation, with care:** the circuit's CNOT chain computes parity, so when optimization succeeds it lands on the parity function itself and generalizes trivially. This is consistent with a well-matched architecture, not with the model learning a hard function from little data.

### 4.2 Failures cluster by initialization

| Initialization seed | Failed partitions |
|---|---|
| 200 | 4 of 10 (101, 102, 103, 109) |
| 201 | 0 of 10 |
| 202 | 0 of 10 |
| 203 | 2 of 10 (105, 108) |
| 204 | 0 of 10 |

No partition failed with more than one seed. Seed 200 succeeded on six partitions, so a seed is not "bad" in isolation; outcomes depend on the seed–partition combination. This matches Phase I, where both failures shared initialization seed 46 (which succeeded on three other partitions).

**Unresolved mechanism:** the recorded metrics do not distinguish local minima, barren plateaus, or simply too few steps. In earlier course runs, initialization seed 46 sat on a plateau and escaped after about 110 steps, so a longer budget may rescue some of these.

## 5. Reproducibility

### 5.1 Independent re-execution

An earlier execution with identical settings and identical source hashes (`20261016T040116_789594Z_34400`) completed 20 configurations before it was stopped. All 20 overlap the canonical run and **match it exactly** (maximum absolute difference in every metric: 0), including three of the failures (101/200, 102/200, 103/200). The failures are deterministic properties of the seeds, not run-to-run noise.

### 5.2 Source provenance

The canonical run recorded:

- `labs/PennyLane.py` — SHA-256 `01aca233…cdf9589`
- `labs/pennylane_repeated_eval.py` — SHA-256 `a3c2781c…097749d`

Both hashes match the files at Git commits `a7730f5` and `c8329f5`.

Two disclosures:

- **Mid-run whitespace change.** For about two minutes during the run, `labs/PennyLane.py` carried a whitespace-only change (one blank line removed between imports) before it was reverted. Each configuration launches that file afresh, so the change could have reached configuration 102/201 and possibly 102/202; both scored 100%, and their results equal those of the independent re-execution in §5.1, which ran entirely before the change. It has no computational effect.
- **Published runner differs from the recorded hash.** In the public repository (commit `55dc4ce` onward), `labs/pennylane_repeated_eval.py` contains a type-checking fix: three missing-match checks were rewritten as explicit `is None` tests. Behavior is identical, but a re-run from the published code will record a different hash for that file (`19b60603…`). To reproduce the exact recorded hashes, check out `c8329f5`.

### 5.3 Data files

Canonical run: `results/pennylane/phase2/20261016T043159_636926Z_41231/`

- `configuration.json` — every setting, versions, source hashes
- `partition_manifest.json` — the validated train/held-out inputs for each partition
- `split_<s>_init_<i>.json` — one file per configuration (50)
- `detailed_results.csv`, `partition_statistics.csv`, `summary.json`
- `execution.log`

Also preserved: the 20-configuration re-execution above, and a 5-step smoke test (`20261016T040040_466610Z_34143`). Neither is counted in the statistics.

### 5.4 Reproduce

```bash
cd quantum-computing
source .venv/bin/activate
python labs/pennylane_phase2_robustness.py --dry-run     # validates all 10 partitions; writes nothing
python labs/pennylane_phase2_robustness.py               # full study, about 55 minutes on Apple silicon
```

## 6. Comparison with Phase I

| | Phase I | Phase II |
|---|---:|---:|
| Configurations | 25 (5 × 5) | 50 (10 × 5) |
| Perfect held-out | 23 (92%) | 44 (88%) |
| Mean held-out accuracy | 94.0% | 90.0% |
| Failures sharing one init seed | 2 of 2 | 4 of 6 (6 of 6 in two seeds) |
| Failures with low training accuracy | 2 of 2 | 6 of 6 |

The two phases use disjoint seeds and agree: a roughly 10% failure rate at 200 steps, driven by optimization and clustered by initialization.

## 7. Scientific Limitations

1. Only 16 possible inputs; each held-out set has four examples, so per-configuration accuracy moves in 25-point steps.
2. Partitions are distinct but share held-out inputs, so partition results are not independent samples.
3. Initializations within a partition share data and are correlated.
4. Five initialization seeds cannot establish a per-seed failure *rate*; the clustering is descriptive.
5. Loss and gradient trajectories were not recorded, so the failure mechanism is not identified.
6. The classical baselines are generic learners, not the best classical solution to parity.
7. Simulation only; no hardware noise.
8. No claim of quantum advantage or out-of-distribution generalization.

## 8. Recommended Follow-Up

- Re-run the six failing configurations at 400–1,000 steps, recording loss and gradient norms, to separate slow convergence from true local minima.
- Use 20 or more initialization seeds on a fixed partition to estimate a per-seed failure rate.
- Repeat with a mismatched ansatz (no CNOT chain) to test how much of the success is architectural.
- Add depolarizing noise (as in Labs 29–31) to measure how robust training is.

## 9. Conclusion

Across 50 validated configurations on new partitions and new initializations, the four-qubit variational parity classifier reached perfect held-out accuracy 88% of the time. Every failure was a failure to train, never a failure to generalize after training, and failures concentrated in two initialization seeds. Results reproduced exactly across independent executions.

The contribution is a validated, reproducible record of *optimization reliability* for a small variational model, together with clear evidence that its successes reflect an architecture matched to the task, not a quantum advantage.

---

**Research integrity statement:** All numbers in this report were computed from the canonical run's exported files listed in §5.3. Raw result files were not modified.
