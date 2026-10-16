# Initialization Sensitivity and Generalization
## A Four-Qubit Variational Quantum Parity Classifier

**Research study:** PennyLane factorial evaluation

**Experiment date:** October 15, 2026

**Author:** Ceasar Jackson

**Status:** Completed experimental study; descriptive analysis

---

## Abstract

This study evaluates initialization sensitivity and held-out performance in a four-qubit variational quantum classifier trained on a binary parity task. A 5 × 5 factorial design combined five training-partition seeds with five parameter-initialization seeds, producing 25 unique configurations.

The classifier achieved perfect held-out accuracy in 23 of 25 configurations. Mean held-out accuracy was 94.00%, with a sample standard deviation of 20.77 percentage points.

Two configurations produced 25% held-out accuracy. Both involved initialization seed 46, paired with training-partition seeds 44 and 45. Their reduced training accuracy indicates incomplete optimization under the tested training budget.

These results characterize the behavior of this specific model and experimental design. They do not demonstrate quantum computational advantage.

---

## 1. Research Objectives

The primary objective was to evaluate whether training partition and parameter initialization influence the optimization reliability of a small variational quantum classifier.

Secondary objectives included evaluating held-out classification performance, documenting descriptive variation across configurations, and preserving a reproducible experimental record.

## 2. Experimental Methodology

| Parameter | Configuration |
|---|---|
| Quantum model | Four-qubit variational classifier |
| Trainable parameters | 16 |
| Classification task | Binary parity |
| Input combinations | 16 |
| Training examples per partition | 12 |
| Held-out examples per partition | 4 |
| Partition seeds | 42, 43, 44, 45, 46 |
| Initialization seeds | 43, 44, 45, 46, 47 |
| Optimization steps | 200 |
| Learning rate | 0.1 |
| Ridge parameter | 1.0 |
| Experimental configurations | 25 |
| Python | 3.13.16 |
| PennyLane | 0.45.1 |

Each partition used 12 training combinations and four disjoint held-out combinations. For each partition, the classifier was trained independently using five initialization seeds.

The detailed CSV files identify partition and initialization seeds but do not independently encode every hyperparameter. The settings above are supported by the validated execution commands and logs.

## 3. Complete Factorial Results

### 3.1 Held-out accuracy matrix

| Partition | Init 43 | Init 44 | Init 45 | Init 46 | Init 47 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 100% | 100% | 100% | 100% | 100% | 100% |
| 43 | 100% | 100% | 100% | 100% | 100% | 100% |
| 44 | 100% | 100% | 100% | 25% | 100% | 85% |
| 45 | 100% | 100% | 100% | 25% | 100% | 85% |
| 46 | 100% | 100% | 100% | 100% | 100% | 100% |

### 3.2 Aggregate descriptive statistics

| Metric | Result |
|---|---:|
| Unique configurations | 25 |
| Mean training accuracy | 97.00% |
| Mean held-out accuracy | 94.00% |
| Held-out accuracy sample SD | 20.77 percentage points |
| Mean training MSE | 0.09670852 |
| Mean held-out MSE | 0.12700724 |
| Perfect held-out configurations | 23/25 |
| Lower-performing configurations | 2/25 |

These statistics describe the 25 selected configurations. They should not be interpreted as independent estimates of population-level generalization accuracy.

## 4. Initialization Sensitivity

The two configurations with less than perfect held-out accuracy were:

| Partition | Initialization | Training accuracy | Held-out accuracy | Training MSE | Held-out MSE |
|---|---|---:|---:|---:|---:|
| 44 | 46 | 66.7% | 25.0% | 0.925636 | 1.379920 |
| 45 | 46 | 58.3% | 25.0% | 0.972273 | 1.193993 |

Both lower-performing configurations used initialization seed 46. However, the same initialization seed achieved 100% training and held-out accuracy on partitions 42, 43, and 46.

**Observed finding:** Optimization outcomes varied with the combination of training partition and initialization.

**Unresolved mechanism:** The available summary metrics do not establish whether the failures arose from local minima, plateaus, gradient behavior, or insufficient optimization steps.

## 5. Classical Baseline Comparisons

The experiment recorded majority-class, ridge-linear, polynomial, and nearest-neighbor baseline results.

Baseline values must be interpreted according to their implementation-specific scoring conventions. In particular, the polynomial baseline's reported abstention-adjusted score should not be represented as conventional classification accuracy.

Parity is a simple, exactly solvable classical task. Therefore, differences between the variational classifier and the particular implemented baselines do not establish quantum advantage.

## 6. Reproducibility

### 6.1 Repository and environment

- Repository: `~/Projects/labs/quantum-computing`
- Local Git checkpoint: `a7730f5`
- Python: `3.13.16`
- PennyLane: `0.45.1`
- Experiment runner: `labs/pennylane_repeated_eval.py`
- Analysis utility: `labs/pennylane_factorial_analysis.py`

### 6.2 Canonical experimental datasets

- Partition 42: `results/pennylane/disjoint_runs_20261015_135700_901540.csv`
- Partition 43: `results/pennylane/disjoint_runs_20261015_143226_802847.csv`
- Partition 44: `results/pennylane/disjoint_runs_20261015_194041_859006.csv`
- Partition 45: `results/pennylane/disjoint_runs_20261015_203356_702327.csv`
- Partition 46: `results/pennylane/disjoint_runs_20261015_204308_944440.csv`

### 6.3 Validated analysis exports

- `results/pennylane/factorial_analysis/factorial_summary_20261015_205050_627551.json`
- `results/pennylane/factorial_analysis/factorial_records_20261015_205050_627551.csv`
- `results/pennylane/factorial_analysis/historical_occurrences_20261015_205050_627551.json`

### 6.4 Reproduce the statistical validation

```bash
cd ~/Projects/labs/quantum-computing
source .venv/bin/activate
python labs/pennylane_factorial_analysis.py --dry-run
```

The dry run validates the canonical datasets and reports descriptive statistics without generating new analysis exports.

### 6.5 Reproduce an experimental partition

```bash
python labs/pennylane_repeated_eval.py \
  --mode initialization \
  --fixed-split-seed 46 \
  --seeds 43,44,45,46,47 \
  --train-unique 12 \
  --steps 200 \
  --learning-rate 0.1 \
  --ridge 1.0
```

This command produces a new execution record. New results must not be counted as additional unique factorial configurations.

## 7. Historical Execution Inventory

The historical audit identified 17 seed combinations appearing in multiple recorded executions.

These occurrences are not automatically equivalent experiments. Historical records may differ in optimization steps, learning rate, training subset size, or other settings.

The canonical 25-configuration dataset therefore uses an explicit source-file manifest rather than counting all historical CSV rows.

## 8. Scientific Limitations

1. Each held-out partition contains only four examples, so individual test accuracy changes in 25-percentage-point increments.
2. The dataset contains only 16 possible binary input combinations, limiting conclusions about larger problems.
3. Partition results are not statistically independent because training and held-out examples can overlap across different partitions.
4. The experiment uses a limited selection of initialization and partition seeds.
5. Optimization trajectories and gradient statistics were not included in the consolidated analysis.
6. Historical CSVs do not independently encode all experimental hyperparameters.
7. The recorded classical baselines do not exhaust classical methods for parity classification.
8. The results do not demonstrate quantum computational advantage.

## 9. Recommended Follow-Up Experiments

Future experiments should examine the two lower-performing configurations under additional optimization steps and record loss and gradient trajectories.

Additional research should include more initialization seeds, expanded partition sampling, controlled noise experiments, circuit-depth comparisons, and tasks with more demanding generalization requirements.

## 10. Conclusion

The 25-configuration factorial study demonstrates that the four-qubit variational parity classifier can achieve perfect held-out accuracy across most of the tested initialization and partition combinations.

The two lower-performing configurations demonstrate observed sensitivity to the combination of parameter initialization and training partition under the selected optimization budget. Additional independent repetitions are needed to establish the stability and reproducibility of this finding.

The principal contribution of this study is a validated experimental record of optimization reliability and initialization sensitivity, not evidence of quantum computational advantage.

---

**Research integrity statement:** All reported numerical findings were generated from the validated canonical factorial analysis exports. Original experimental CSV files were preserved without modification.
