# Excess Singlet-Preparation Loss on IBM Heron: Calibration Versus Measurement (Labs 55–59)

| Field | Value |
|---|---|
| Project | Quantum Computing Course (claude.ai Project "Quantum Computing") |
| Author | Dr. Ceasar Jackson Jr. |
| Date | 2026-10-18 |
| Repository | `CeasarJackson/quantum-computing` (local commits through `5964243`; the last five are not yet pushed) |
| Branch | `main` |
| System under test | Two-qubit singlet preparation (H, CX, Z, X compiled to one CZ per link) on 156-qubit Heron backends `ibm_fez`, `ibm_kingston`, `ibm_marrakesh` (IBM Open plan), measured by X/Y/Z correlators with readout correction |
| Scripts | `labs/lab55_other_pairs_other_device.py`, `lab56_calibration_vs_loss.py`, `lab57_persistence.py`, `lab58_what_persistent_links_share.py`, `lab59_vertical_links.py` |

## Executive Summary

Earlier labs (46–54, recorded in `course-index.md`) found that singlet preparation on one six-qubit `ibm_fez` chain lost about ten times what the reported gate errors imply. Labs 55–59 asked whether that is specific to that chain, whether the reported calibration can tell good links from bad ones, and whether the extra loss is stable. On well-read links of all three backends the loss is about 3–4 times the calibration-implied value, it repeats when measured again about two hours later (Spearman ρ = +0.95, n = 13), and no calibration number, qubit position, or row-versus-vertical coupler type ranks the links. The cause of the stable per-link excess is not identified.

## Verified Results

Every row below was observed in a terminal output pasted into this session.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Lab 55 pre-registered rule, `ibm_fez` (8 fresh links) | GAP exists (R 3.1, excess +0.048 ± 0.007) | Lab 55 live run, 01:19 |
| 2 | Lab 55 pre-registered rule, `ibm_kingston`, `ibm_marrakesh` | AGREES (R 0.8 / −0.0) but unreliable: the mean-over-links rule was dominated by failed readout correction | Lab 55 live runs; amendment `eac46cd` |
| 3 | Lab 55 per-link reading (post-hoc), links with both readout attenuations ≥ 0.95 | 14 of 15 above implied; excluding `kingston 14-15`, 14 of 14, mean 0.081 vs 0.021 (R 3.8, +0.059 ± 0.005) | Lab 55 `--meta-file` re-analysis of the three jobs |
| 4 | Lab 56: any of 10 features ASSOCIATED with loss (n = 15) | None; one SUGGESTIVE (`t2_min` ρ −0.52, p 0.0485); the calibration's own implied loss ρ +0.08, p 0.78 | Lab 56 live run 02:49 |
| 5 | Lab 57: loss persists on repeat (n = 13) | PERSISTENT: ρ = +0.95, p < 0.0001; χ² 1.39; shift −0.017 ± 0.007; 4 of 5 HIGH links HIGH again; 0 of 8 others became HIGH | Lab 57 live run 03:39 |
| 6 | Lab 58: pooled loss vs 12 features (n = 13) | None ASSOCIATED or SUGGESTIVE; `t2_min` hint does not replicate (ρ −0.42, p 0.15) | Lab 58 run on the Lab 56/57 CSVs |
| 7 | Lab 59: links between chip rows vs row links (8 vs 8) | NO DIFFERENCE: D = +0.002 ± 0.010; a difference ≥ 0.029 would be seen at 3σ | Lab 59 live run 05:50 |
| 8 | Dry-run checks (no QPU) | Lab 57: 12 pass; Lab 58: 10 pass; Lab 59: 14 pass; `check_standards` pass on all | terminal output |

## Failure Signature Before Resolution

Two signatures prompted the amendment of Lab 55. First, the pre-registered rule averaged over links and gave "AGREES" on two backends while individual links were physically impossible. Second, a link with poor readout produced a negative singlet loss. From the Lab 57 first-run column (the same link, Lab 55 measurement):

```
ibm_marrakesh 66-67      0.0153   -0.368 +/- 0.081 UNPHYSICAL  +0.127 +/- 0.023 HIGH
ibm_kingston  14-15      0.1484   -0.020 +/- 0.021 BELOW       -0.005 +/- 0.018 BELOW
```

`marrakesh 66-67` had readout attenuation α = 0.63 at run time (Lab 55 per-link reading). Dividing by a small α amplifies noise, so the corrected correlator can go below −1.

## Controlled Resolution Test

What was done, in order, so it can be reproduced from the repository:

1. Lab 55: choose eight disjoint links per backend from the backend's own calibration (readout error < 0.06, reported CZ error < 0.05) at CZ-error percentiles 5, 20, 35, 50, 65, 80, 90, 97. Measure the singlet correlators in X, Y, Z (40 circuits × 2000 shots, one job per backend). The loss of a link is Σ over X, Y, Z of (correlator + 1).
2. Lab 55 amendment: correct readout with a model m = αz + β (α = 1 − f0 − f1, β = f1 − f0), add the calibration's α uncertainty to the shot noise, and classify each link UNPHYSICAL / HIGH / BELOW / CONSISTENT. This amendment is labelled "not pre-registered" in the docstring.
3. Lab 56 (no QPU): rank-correlate the measured loss of links with both α ≥ 0.95 against ten calibration features (permutation p from 20 000 shuffles, Bonferroni).
4. Lab 57: repeat the Lab 55 measurement on the same 24 links with the same circuits and shots. Persistence rule fixed beforehand: PERSISTENT at ρ ≥ 0.70 and p < 0.01; TRANSIENT at ρ < 0.30.
5. Lab 58 (no QPU): repeat the feature test on the inverse-variance mean of the two runs, adding two structural features (qubit-number gap, lower qubit number).
6. Lab 59: four links between chip rows and four row links per backend, matched on reported CZ error. Rule fixed beforehand: VERTICAL HIGHER at D ≥ +0.03 and > 3σ, NO DIFFERENCE at |D| < 2σ.

## Successful Evidence

Lab 57 (the persistence test), verbatim from the live run:

```
03:39:46 [INFO] set: 13 links with both attenuations >= 0.95 in both runs
03:39:46 [INFO] Spearman rho of repeat against run 1: +0.95  (p = 0.0000, n = 13)
03:39:46 [INFO] CHI2 1.39: the differences are about the size of the shot noise (same true loss)
03:39:46 [INFO] SHIFT of the mean loss, repeat minus run 1: -0.017 +/- 0.007
03:39:46 [INFO] HIGH: 4 of 5 links HIGH in run 1 are HIGH again; 0 of 8 others became HIGH
03:39:46 [INFO] verdict (persistence rule): PERSISTENT: the order of the links repeats; the loss is a stable property of the link that the calibration does not report
```

Lab 59 (the row-versus-vertical test), verbatim:

```
05:50:44 [INFO] pooled: vertical +0.024 (n=8), row +0.022 (n=8)
05:50:44 [INFO] D = vertical - row = +0.002 +/- 0.010 (0.2 sigma)
05:50:44 [INFO] detectable at 3 sigma: a difference of about 0.029 or more
05:50:44 [INFO] verdict: NO DIFFERENCE: the two kinds of link lose alike (to this precision)
```

Persistent HIGH links (loss in run 1 → repeat, against the implied loss):

| Link | Implied | Run 1 | Repeat |
|---|---|---|---|
| fez 145-146 | 0.0107 | +0.075 ± 0.019 | +0.080 ± 0.017 |
| fez 13-14 | 0.0157 | +0.154 ± 0.019 | +0.086 ± 0.017 |
| fez 66-67 | 0.0149 | +0.151 ± 0.029 | +0.121 ± 0.030 |
| kingston 105-106 | 0.0102 | +0.111 ± 0.018 | +0.099 ± 0.016 |
| marrakesh 43-56 | 0.0234 | +0.165 ± 0.020 | +0.143 ± 0.019 |

Two links lost their excess between runs: `marrakesh 60-61` (0.175 ± 0.022 → 0.032 ± 0.030, 3.9σ) and `kingston 148-149` (0.051 → 0.031).

## Root-Cause Assessment

What the evidence supports:

- The excess singlet-preparation loss is not confined to the six original `ibm_fez` chain qubits or to `ibm_fez`: on well-read links it appears on all three backends at roughly 3–4 times the calibration-implied loss (rows 1–3).
- It is largely a stable property of a link over about two hours (row 5), heterogeneous across links, and not reported by the backend's calibration: the calibration's own implied loss does not rank the links (rows 4 and 6).
- It is not a readout artefact for the links used, because the analysis keeps only links whose measured readout attenuation is ≥ 0.95, and the earlier labs found it random (about 96% incoherent, Lab 54), Z-weighted and local to each link (course index).
- Neither T1/T2, reported CZ/SX/readout error, neighbour error, spectator count, qubit position, nor whether the coupler joins two chip rows explains which links carry it (rows 4, 6, 7).

### Experimental limitation

- The Lab 55 pre-registered mean-over-links rule was badly built and its AGREES verdicts are not evidence. Everything about per-link behaviour, including the α ≥ 0.95 cut and the HIGH/BELOW classes, is post-hoc and was chosen after seeing the first runs.
- The groups are small (n = 13 for Labs 57–58, 8 vs 8 for Lab 59). Only a rank correlation of about 0.7 or more, or a group difference of about 0.03, could have been detected, so "no feature found" is weak.
- The implied loss assumes depolarizing noise with three SX gates per qubit; a different model would shift the excess, though not the ranking tests.
- Calibration features were read at 02:10–02:20, after the first jobs and before the repeat; the repeat was 1–2 hours later, so a calibration refresh and slow drift cannot be separated. The two links that lost their excess show it is not perfectly stable.
- Persistence over days, the cause of the excess, and other structure (distance to the chip edge, neighbour counts) were not tested. One run per link per condition.

## Recommended Procedure

1. Before trusting a link's singlet or Bell-pair quality, measure it: do not rely on the reported two-qubit error alone (Labs 55–57).
2. Measure readout attenuation α in the same job and drop links with α < 0.95; correct readout with both α and β and carry the α uncertainty into the error bar (Lab 55 amendment).
3. State the decision rule (what counts as agreeing, and over which set) in the script docstring before the live run; label anything added afterwards "not pre-registered".
4. Repeat the measurement on the same links within a few hours to separate stable from transient excess (Lab 57), and use `--collect` / `--retrieve` for slow queues instead of resubmitting.
5. Run no-QPU re-analyses on the saved CSVs before spending more allowance (Labs 56 and 58).

## Alignment with Official Documentation

IBM documents the reported quantities this report compares against. The QPU information page defines the median two-qubit error as:

> "Average gate fidelity of the two-qubit operation from randomized benchmarking."

and states that the per-edge two-qubit error comes from "the same batch of measurements used to calculate the 2Q median error", measured in isolation with edges separated by at least two qubits, with the final value including "the error of the layer of single-qubit Cliffords". Readout error is described as "the average probability of incorrectly measuring a qubit's state", "commonly calculated as the mean of `prob_meas0_prep1` and `prob_meas1_prep0`". T2 "is reported from a Hahn echo sequence". (IBM Quantum Platform, QPU information.)

The calibration-jobs page describes how often these values change: monitoring jobs "approximately once an hour", calibrations "triggered whenever monitoring jobs indicate that parameters" drift, `last_update_date` updated "whenever a calibration occurs", and "Two jobs submitted at the same time might run under different calibration sets, depending on timing." (IBM Quantum Platform, Calibration jobs.) The fetched text of that page does not state an explicit "reported errors drift between calibrations" sentence; it lists drift as depending on factors such as when calibrations were last run.

| Claim | Established by |
|---|---|
| The reported two-qubit error comes from isolated randomized benchmarking including a layer of single-qubit Cliffords | IBM documentation (QPU information) |
| Reported readout error is the mean of `prob_meas0_prep1` and `prob_meas1_prep0` | IBM documentation (QPU information) |
| Calibrations and monitoring run throughout the day, and jobs may run under different calibration sets | IBM documentation (Calibration jobs) |
| Singlet-preparation loss on well-read links is about 3–4 times the calibration-implied loss | This session's evidence only: `logs/lab55/job_*.json`, `logs/lab56/links_20261018_024924.csv` (SHA-256 below) |
| The excess repeats about two hours later (ρ +0.95) | This session's evidence only: `logs/lab57/pairs_20261018_033855.csv` |
| Links between chip rows lose no more than row links (D +0.002 ± 0.010) | This session's evidence only: `logs/lab59/links_20261018_054955.csv` |
| The calibration does not carry the quantity that distinguishes the bad links | This session's evidence only (Labs 56 and 58, small n) |

### Note on terminology

"Median 2Q error", "readout error", "T1", "T2", `last_update_date` and "calibration set" are IBM's terms. "Implied loss", "excess", "HIGH / BELOW / CONSISTENT / UNPHYSICAL", "persistent", "vertical link" and "row link" are this report's own operational language for the analysis; IBM does not use or attest them. IBM's own text, as fetched, does not say that RB-derived two-qubit error should predict singlet-preparation loss; the comparison is this report's, using a depolarizing mapping (loss = 3λ with λ = r/0.75) that is a modelling assumption.

## Evidence Artifact

Saved in the repository under `logs/` (not tracked by git), SHA-256 as computed on the Mac on 2026-10-18:

```
dc551525d59a1951060fb2c96ab9f71c8258394578e9288fc6e682ecca535fde  logs/lab55/job_ibm_fez_dba5dn424mrc7384u3f0.json
12b97149c7e5c33d7901800225d079e32e8c5eb4fec58e75a7bf886048691266  logs/lab55/job_ibm_kingston_dba4os1gv2cc73e6bv90.json
7c6dc96afdd8d63fcf35d63a9dfe1fe50d23a39bc9cd29b15b64105348353d0d  logs/lab55/job_ibm_marrakesh_dba5d51gv2cc73e6cupg.json
608b179eb023000ecd1d393f1f0e1871001f909ed226ac764fbf486c13429cda  logs/lab56/links_20261018_024924.csv
73cb21969dd623a68081d69cd36e037998253677fafd37b9080d1677f08f17ef  logs/lab57/job_ibm_fez_dba7f5424mrc73850k70.json
446982716e5885a81f0701b85f41e2b30834b8e2fdc53c115722de63858125ec  logs/lab57/job_ibm_kingston_dba7f5fi0auc7391kf5g.json
058efc75ed312d56c287894cc93da05c7f7656e5b191c0c6acbe41871391f260  logs/lab57/job_ibm_marrakesh_dba7f5gc77vs7382duog.json
9183838f563e43a23df3b68c084b36cfceb13ec4d82b0016102f88c6fbe76334  logs/lab57/pairs_20261018_033855.csv
68e4a464a0037766ca92059efc75ce27706c76c638959e097907887285658233  logs/lab59/job_ibm_fez_dba9ci1gv2cc73e6hkgg.json
5f64b6cee5aa3d5a3b8fd34bdd2be0b41a3357860b83e896fbe05ebdde020aa5  logs/lab59/job_ibm_kingston_dba9ci8c77vs7382g4l0.json
30a103c1a773ae1039b5bb5cd9484c79f14ec9236fc5032ab42b101057751f73  logs/lab59/job_ibm_marrakesh_dba9cihgv2cc73e6hkh0.json
9ac5bde394fcd47b825c8eca7fa533acb3703c3e444b87903ccc262d809f5eee  logs/lab59/links_20261018_054955.csv
```

Git history for these labs (newest first): `5964243` (record Lab 59), `8813f00` (Lab 59), `fdbbe2f` (Lab 58), `4e6c236` (record Lab 57), `60e268b` (Lab 57), `91ebfb4` (record Lab 56), `3903a76` (Lab 56), `0f27814` (record Lab 55), `eac46cd` (Lab 55 amendment), `f790089` (Lab 55). The Lab 58 output was produced in the sandbox from the Lab 56/57 CSVs above; its `pooled_*.csv` was written in the sandbox, not on the Mac, and is not listed.

## Portfolio Significance

The investigation shows a method more than a result: a surprising number on hardware was followed by repeated, cheaper tests, each with its decision rule written in the script before the run, and each failure of a rule (the badly built Lab 55 mean, the unphysical negative losses, the T2 hint that did not replicate) recorded rather than smoothed over. Each step used the saved data to decide whether more QPU time was worth spending, and the whole sequence ran within a few minutes of allowance. The data engineering is reusable: one measurement pipeline, one set of shared estimators, CSV hand-offs between labs, and hashes of the evidence. The honest ending is a stable, reproducible anomaly with its cause still open.

## References

1. IBM Quantum Platform. "QPU information." <https://quantum.cloud.ibm.com/docs/en/guides/qpu-information>
2. IBM Quantum Platform. "Calibration jobs." <https://quantum.cloud.ibm.com/docs/guides/calibration-jobs>
3. Repository `CeasarJackson/quantum-computing`: `course-index.md` (Labs 46–59 paragraphs) and the lab docstrings listed in the metadata table.
