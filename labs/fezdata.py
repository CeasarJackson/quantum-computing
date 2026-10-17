#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/fezdata.py

Purpose
-------
The numbers recorded from the live ibm_fez run of Lab 44 (2026-10-16), kept in
ONE place so that Labs 46 and 47 analyse exactly the same data:

  * LAYOUT        the six physical qubits that run used (one fixed chain);
  * LAMBDA_FEZ    the depolarizing strength Lab 44 trained against (0.0037);
  * ANGLES        the six angle sets (ideal and noise-trained, depth 1-3)
                  that Lab 44's seed-7 training produced. They are stored, not
                  retrained, because training takes about 40 s and the result
                  is deterministic; the labs check that the stored angles
                  still reproduce the model energies Lab 44 printed;
  * MODEL_ENERGY  the noisy-model energies Lab 44 printed for those angles;
  * MEASURED      the measured energies and standard errors (2000 shots per
                  basis, readout-corrected);
  * run-size facts (shots, circuits, seconds running on the device) for
                  both live runs so far (Lab 44 and Lab 46);

  * shortfalls / fit_lines / fit_design  the small fits Labs 46 and 47 share: energy
                  lost against the noiseless energy, and a weighted fit of the
                  measured loss to c + s * (modelled loss).

Nothing here is secret: no job identifier, no account name, no key.

Security
--------
Holds recorded numbers only. No secrets, no network access.

Usage
-----
    from fezdata import ANGLES, MEASURED, LAYOUT, LAMBDA_FEZ

Log files
---------
None. This is a library module. (Logging is the caller's job, through
`from applog import get_logger`; this library only holds data and fits.)

Exit Codes
----------
Library module; it never exits the process.
===============================================================================
"""

import numpy as np

# Physical qubits of the Lab 44 live run, in chain order (virtual qubit 0 first).
LAYOUT = (129, 118, 109, 110, 111, 112)

# Equivalent depolarizing strength per CNOT from Lab 17's recorded mirror decay.
LAMBDA_FEZ = 0.0037114

# Angles of Lab 44's seed-7 training under that noise; keys are (depth, kind).
ANGLES = {
    (1, "ideal"): [-0.141671, -0.216088],
    (1, "aware"): [0.140333, 0.216259],
    (2, "ideal"): [-0.148602, -0.366789, -0.251079, -0.168772],
    (2, "aware"): [0.135271, -1.210454, 0.237494, -1.401685],
    (3, "ideal"): [0.116852, 1.031196, 0.374211, -0.134015, -0.332476, -0.483524],
    (3, "aware"): [0.009301, 0.698891, -0.14136, -0.368052, -0.237743, -0.171734],
}

# Noisy-model energies (lambda = 0.0037) printed by Lab 44 for those angles.
MODEL_ENERGY = {
    (1, "ideal"): -9.5734,
    (1, "aware"): -9.5734,
    (2, "ideal"): -9.3954,
    (2, "aware"): -9.3977,
    (3, "ideal"): -9.0308,
    (3, "aware"): -9.1626,
}

# Measured (energy, standard error): 2000 shots per basis, readout-corrected.
MEASURED = {
    (1, "ideal"): (-8.7449, 0.0733),
    (1, "aware"): (-8.9526, 0.0727),
    (2, "ideal"): (-8.6760, 0.0766),
    (2, "aware"): (-8.7781, 0.0758),
    (3, "ideal"): (-8.4362, 0.0802),
    (3, "aware"): (-8.5534, 0.0785),
}

# Run size: 20 circuits (6 angle sets x 3 bases + 2 calibration), 2000 shots.
SHOTS = 2000
CIRCUITS = 20
# The job showed RUNNING for 16 s (queued 6 s, done at 22 s wall clock).
RUNNING_SECONDS = 16.0
# Every live ibm_fez run so far: (circuits, shots per circuit, seconds RUNNING).
# Lab 44 (2026-10-16): 20 x 2000 in 16 s. Lab 46 (2026-10-17): 15 x 4000 in
# 62 s (queued 219 s). The seconds per shot differ by 2.6x between the two.
RUNS = ((20, 2000, 16.0), (15, 4000, 62.0))

# Lab 46 live result (ibm_fez, 2026-10-17, job db9lmkgc77vs7381mdv0 not stored
# here): the depth-0 energy and the three-part refit of all nine energies,
#     loss = c0 + c1 * [depth >= 1] + s * (modelled loss).
# Lab 48 turns these into predictions fixed BEFORE its own run.
LAB46_DEPTH0 = (-8.709, 0.044)
LAB46_FIT = {"c0": (0.259, 0.044), "c1": (0.416, 0.068), "s": (0.934, 0.108)}


def shortfalls(chain, lam):
    """(model, measured, sigma) shortfalls from the noiseless energy, 6 circuits."""
    mod, mea, sig = [], [], []
    for key in ANGLES:
        free = chain.energy(ANGLES[key], 0.0)
        mod.append(chain.energy(ANGLES[key], lam) - free)
        mea.append(MEASURED[key][0] - free)
        sig.append(MEASURED[key][1])
    return np.array(mod), np.array(mea), np.array(sig)


def fit_lines(mod, mea, sig):
    """Weighted fits: scale only, and offset plus scale.

    Returns (s1, chi2_scale, (c, s), cov, chi2_both).
    """
    w = 1 / sig**2
    # One parameter: measured = s * model.
    s1 = float((w * mod * mea).sum() / (w * mod * mod).sum())
    chi_scale = float((w * (mea - s1 * mod) ** 2).sum())
    # Two parameters: measured = c + s * model (normal equations).
    a = np.vstack([np.ones_like(mod), mod]).T
    cov = np.linalg.inv(a.T @ (a * w[:, None]))
    beta = cov @ (a.T @ (w * mea))
    chi_both = float((w * (mea - a @ beta) ** 2).sum())
    return s1, chi_scale, beta, cov, chi_both


def fit_design(columns, mea, sig):
    """Weighted linear fit mea = sum_i beta_i * columns[i]; (beta, cov, chi2)."""
    # Normal equations with weights 1/sigma^2, as in fit_lines but for any
    # number of columns (a constant, a step at depth 1, the modelled loss...).
    w = 1 / np.asarray(sig) ** 2
    a = np.vstack(columns).T
    cov = np.linalg.inv(a.T @ (a * w[:, None]))
    beta = cov @ (a.T @ (w * np.asarray(mea)))
    chi2 = float((w * (np.asarray(mea) - a @ beta) ** 2).sum())
    return beta, cov, chi2
