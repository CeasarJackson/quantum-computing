# PennyLane Classifier: Guide

Companion to the Quantum Computing course (`labs/`). Covers the two PennyLane scripts and the lab that tests them.

| File | Role |
|---|---|
| `labs/PennyLane.py` | Trains a 4-qubit variational binary classifier (the model). |
| `labs/pennylane_repeated_eval.py` | Repeats the training over many seeds and compares it with two classical baselines (the experiment). |
| `labs/lab32_pennylane_classifier.py` | Lab 32: verifies both scripts (the test suite, 22 checks). |

## 1. What the model does

- Task: label each 4-bit input by its parity (+1 for an even number of ones, -1 for odd). Parity is not linearly separable, so it is a fair stress test for a small circuit.
- Encoding: each bit becomes an RX rotation of 0 or pi on its own qubit.
- Trainable part: 2 layers of RY and RZ on every qubit, each followed by a CNOT chain closed into a ring. 2 layers x 4 qubits x 2 angles = 16 parameters.
- Readout: the expectation of Pauli-Z on qubit 0, which lies in [-1, 1]. The sign is the predicted class.
- Loss and optimiser: mean squared error against the +-1 labels, plain gradient descent, gradients by parameter-shift (exact for these gates, no finite differences).
- Baselines: majority class (learned from the training labels only) and, in the repeated evaluation, a ridge-regularised linear classifier on the raw bits, a ridge model on all 16 spin products (the model class contains parity) and 1-nearest-neighbour.

## 2. Install

PennyLane is pinned in `requirements.txt` (11 new lines, none of the existing pins changed):

```
appdirs==1.4.4          pennylane==0.45.1
autograd==1.8.0         pennylane_lightning==0.45.0
autoray==0.8.4          scipy-openblas32==0.3.34.237.1
cachetools==7.2.1       termcolor==3.3.0
diastatic-malt==2.15.3  tomlkit==0.15.1
gast==0.7.0
```

With the project venv active: `pip install -r requirements.txt`. No network access, credentials or data files are used at run time.

## 3. Running the classifier

```
python labs/PennyLane.py --dry-run
python labs/PennyLane.py --split disjoint --train-unique 12 --steps 50
python labs/PennyLane.py --split disjoint --split-seed 42 --init-seed 47
python labs/PennyLane.py --steps 50 --samples 20 --test-samples 20 --seed 42
```

| Option | Default | Meaning |
|---|---|---|
| `--split` | `sampled` | `sampled` draws train and test inputs independently (they can repeat or overlap). `disjoint` partitions all 16 inputs, so every test input is unseen. Prefer `disjoint`. |
| `--train-unique` | 12 | Distinct training inputs in a disjoint split (1..15); the other 16-k are the test set. |
| `--steps` | 50 | Gradient-descent steps. |
| `--learning-rate` | 0.1 | Step size. |
| `--seed` | 42 | Master seed. |
| `--split-seed` | `--seed` | Seed for the train/test partition only. |
| `--init-seed` | `--seed`+1 | Seed for the initial weights only. |
| `--diff-method` | `parameter-shift` | or `best` (backpropagation); both give the same gradient. |
| `--dry-run` | off | Check dependencies and arguments, train nothing. |
| `--verbose` | off | Log the loss at every step. |

Splitting the two seeds is what makes controlled experiments possible: change one source of randomness at a time.

Log: `logs/PennyLane/PennyLane_<timestamp>.log`. Exit codes: 0 success, 1 failure, 2 bad argument, 130 interrupted.

## 4. Repeated evaluation

```
python labs/pennylane_repeated_eval.py --dry-run
python labs/pennylane_repeated_eval.py --seeds 42,43,44,45,46 --steps 50
python labs/pennylane_repeated_eval.py --mode initialization --fixed-split-seed 42 --seeds 43,44,45,46,47
python labs/pennylane_repeated_eval.py --mode partition --fixed-init-seed 43 --seeds 42,43,44,45,46
```

Modes (what each seed varies):

| Mode | Split seed | Init seed | Answers |
|---|---|---|---|
| `paired` | seed | seed+1 | Overall spread when both vary. |
| `initialization` | fixed (`--fixed-split-seed`) | seed | How much does the starting point matter on one partition? |
| `partition` | seed | fixed (`--fixed-init-seed`) | How much does the choice of test inputs matter? |

For each seed the script runs `PennyLane.py` in a subprocess with the current interpreter, reads its metrics back from the log lines, and fits the baselines on the identical partition: majority class, ridge-linear on the bits, ridge on all 16 products of the four spin values (this basis contains parity exactly) and 1-nearest-neighbour with Hamming distance. Output goes to `results/pennylane/`:

- `disjoint_runs_<stamp>.csv`: one row per seed. Columns: `seed, mode, split_seed, init_seed, quantum_{train,test}_mse, quantum_{train,test}_accuracy_pct, majority_{train,test}_accuracy_pct, linear_{train,test}_accuracy_pct`.
- `disjoint_summary_<stamp>.csv`: per metric `count, mean, std_sample, min, max`.

CSV files are written to a temporary file and then renamed, so an interrupted run never leaves a half-written result. Log: `logs/pennylane_repeated_eval/`.

## 5. How to read the results (statistics)

A 12/4 split has only four test inputs, so test accuracy can only be 0, 25, 50, 75 or 100 %.

- When 1, 2 or 3 of the 4 test inputs are right, the 95 % Wilson interval contains 0.5: 3 of the 5 possible outcomes cannot be told from a coin flip.
- A model that guesses has mean accuracy 50 % and standard deviation 25 points on four inputs. In about 44 % of five-seed guessing runs the sample SD reaches 25 points or more, so a 17 to 41 point SD in the summary is not evidence of anything.
- Results from the earlier Mac runs (50 steps, 12/4 splits, five seeds each):
  - Initialization mode on split 42: quantum test accuracy mean 70 % (SD 41), while majority and ridge-linear both score 0 % on that particular partition. One unlucky partition makes the baselines look terrible, which shows how much the partition matters.
  - Partition mode: quantum 25 % (SD 18), majority 35 % (SD 22), linear 30 % (SD 21). No method is distinguishable from another.
  - 100 steps, initialization mode on split 42 (2026-10-14 12:24 run, seeds 43-47): quantum test accuracy 100, 100, 100, 50, 100 % (mean 90 %, SD 22); majority and ridge-linear 0 % (that partition's four test labels are all the minority class, and a linear model cannot represent parity). Four of five initializations reach training MSE of about 0.001 to 0.01 and test MSE of about 0.001 to 0.012, i.e. confident, correct predictions on inputs never trained on; init 46 had not finished at 100 steps (train MSE 0.93, 66.7 %); see the next bullet. Getting all four test inputs right by guessing has probability 1/16, so four of five perfect runs is real evidence that the circuit learns parity on this partition, and the 50-step runs (mean 70 %) were under-trained.
  - Init 46 is a plateau, not a bad minimum (2026-10-14 12:55 run, split 42, same seeds, longer training). The loss sits at 0.95 to 0.93 from step 40 to step 100, then escapes: MSE 0.311 at 150 steps, 0.0033 at 200, 0.00033 at 300, with held-out accuracy 100 % at 150, 200 and 300 steps (test MSE 0.46, 0.0043, 0.00041). The runs are deterministic, so the shorter runs are exact prefixes of the longer ones. Practical consequence: judge convergence by the loss curve, not by a fixed step count; 100 steps is too short for some initializations at learning rate 0.1, and 200 steps (about 65 s per run on the Mac) is a safer setting for repeated evaluations. A larger learning rate or a different initialization scale might shorten the plateau; that has not been tested.
  - Partition mode at 200 steps (sandbox run, 2026-10-14, splits 42-46, fixed init 43): quantum test accuracy 100 % on all five partitions (SD 0), against majority 0, 25, 50, 50, 50 % (mean 35 %) and ridge-linear 0, 25, 25, 50, 50 % (mean 30 %). Guessing would have to get all four test inputs right five times in a row, probability (1/16)^5, about 1e-6. So the earlier 25 % mean at 50 steps came from under-training, not from the partitions. Split 45 had only reached training MSE 0.50 at 200 steps (accuracy already 100 %, the loss still falling), the other four were at 0.0004 to 0.0035, which is the same plateau behaviour as init 46.
  - Nonlinear classical comparators (splits 42-46, 12 train / 4 test, computed with `pennylane_repeated_eval.py`, checked by Lab 32 check 8): the 16-term polynomial fits all 12 training inputs exactly but learns nothing about the unseen four: its ridge score there is exactly zero (about 1e-15 of rounding noise), so it is scored as a coin flip, 50 % on every split. An earlier version counted the sign of that noise as a prediction, which gave different numbers on different machines (40 % in the sandbox, 70 % on the Mac) and made Lab 32 fail on the Mac; that is fixed; 1-nearest-neighbour scores 0 % on all five splits (mean 0 %); ridge-linear 30 %. 1-NN fails by construction: every input one bit away has the opposite parity, so with all neighbours in the training set it is always wrong (Lab 32 confirms this on 10 leave-one-out cases). The quantum circuit's 100 % on all five splits therefore is not what any nonlinear model gives.
  - Why it succeeds, and what that does not mean: the circuit is not a generic model. RX(0 or pi) encodes each bit as a basis state and the CNOT chain and ring accumulate the XOR of the bits, which is parity, so the architecture carries a parity-friendly inductive bias; training mostly has to tune 16 angles into the right regime. That is a property of this ansatz on this task. A classical model given the same bias (a feature equal to the XOR of the bits) would solve it with one parameter, so the result says that the circuit is well matched to parity, not that quantum models are better.
- Conclusion to carry forward: the training works (loss falls, gradients are exact), the evaluation protocol is sound, and given enough steps (200) every initialization tried learns parity on split 42. What is still not shown is an advantage: generic classical learners fail here, but a classical model given the XOR feature solves parity exactly with one parameter; so the partition-mode result above (five splits, all 100 %) is the check that it generalises beyond split 42. A stronger comparison needs a task with many more test examples and no built-in match between model structure and answer.

## 6. How Lab 32 verifies it

`python labs/lab32_pennylane_classifier.py [--quick] [--steps N] [--seed S]` (about 50 s in the sandbox, less with `--quick`). 25 checks in eight groups:

1. Data: sampled angles are 0 or pi, labels equal the parity sign; disjoint splits use all 16 inputs exactly once for any training size, and sizes 0, 16 and -3 are rejected.
2. Baselines: the majority label comes from the training labels only; ties predict +1.
3. Circuit: 20 random circuits (including generic angles) match a from-scratch NumPy state-vector simulation to 3e-16.
4. Gradient: parameter-shift equals central finite differences (3e-10) and the `best` backpropagation gradient (2e-16).
5. Training: the loss falls (2.308 to 1.455 in 12 steps), the same seeds reproduce identical output, a different initialisation seed changes the result.
6. Helpers: seed parsing, CSV round trip with no temporary file left behind, and a real 2-step subprocess run whose parsed accuracies come in steps of 25 %.
7. Statistics: the Wilson intervals and guessing spread quoted in section 5.
8. Classical comparators: the polynomial fits the training set exactly yet no comparator learns parity from 12 of 16 inputs, and 1-NN is wrong on 10 of 10 leave-one-out inputs.

## 7. Known limitations

- File name versus package: `PennyLane.py` has the same name (ignoring case) as the `pennylane` package. It works on macOS and Linux today because Python imports by exact name, but a case-insensitive import on another platform or tool could pick the wrong module. Renaming to `pennylane_classifier.py` would remove the risk; it would mean updating the import and the script path in `pennylane_repeated_eval.py` and the imports in Lab 32.
- Coupling by text: the repeated evaluation reads the classifier's log messages with regular expressions. If the wording of "Training complete", "Held-out test" or "Disjoint input split verified" changes, it fails loudly (and Lab 32 check 6 fails first). That is deliberate: it never reports numbers it did not parse.
- Cost: training is a Python loop over samples, so a five-seed, 50-step evaluation takes a minute or two (the earlier Mac runs took about 1.5 minutes).
- Simulator only: `default.qubit`, exact expectation values, no shot noise and no hardware noise.

## 8. Housekeeping done when the files were integrated

- Moved out of `labs/` to `archive/pennylane_drafts/` (nothing deleted): three earlier drafts (`PennyLane_independent_seeds.py`, `PennyLane_with_test.py`, `pennylane_repeated_eval_stability.py`) and the `.bak` files. `check_standards.py` and `run_all.py` therefore audit only live scripts.
- `pennylane_repeated_eval.py`: usage lines wrapped, two long lines split, six why-comments added; behaviour unchanged. The previous version is saved as `archive/pennylane_drafts/pennylane_repeated_eval.pre_fix.py`.
- `PennyLane.py` is unchanged.
- `run_all.py` runs Lab 32 (which exercises both scripts) with a 90 s timeout.
