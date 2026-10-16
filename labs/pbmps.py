#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/pbmps.py

Purpose
-------
The per-bond Heisenberg ansatz (Lab 40) on a matrix product state, with an
exact adjoint gradient. Every bond gate is exp(-i t (XX + YY + ZZ)) with its
OWN angle t, so total spin is conserved however the angles differ. One layer
is the odd bond group followed by the even group, as in Lab 26.

The gradient follows Lab 27: the bonds of one group commute, so
dE/dt_b = 2 Im <phi| h_b |psi_k> for each bond b of the step, where psi_k is the
state after the step and phi is H|psi_final> with the later gates undone. One
forward pass, one MPO application and one backward pass give ALL angles, so
the cost does not grow with the number of parameters (n - 1 per layer).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    from pbmps import perbond_energy_grad, perbond_state, step_sizes

Log files
---------
None. This is a library module. (Logging is the caller's job, through
`from applog import get_logger`; this library only returns numbers.)

Exit Codes
----------
Library module; it never exits the process.
===============================================================================
"""

import numpy as np

from mpslib import (
    BOND,
    MPS,
    apply_mpo,
    bond_overlaps,
    bond_unitary,
    group_terms,
    heisenberg_mpo,
)


def step_sizes(n, layers):
    """Number of bonds in each step: odd group, even group, per layer."""
    # Odd group first, then even, per layer (the brick-wall order of Lab 26).
    sizes = []
    for _ in range(layers):
        sizes += [len(group_terms(n, 1)), len(group_terms(n, 0))]
    return sizes


def split_angles(n, x):
    """Cut a flat angle vector into [(parity, angles of that group), ...]."""
    # i walks along the flat vector; each pass consumes one group's angles.
    steps, i = [], 0
    while i < len(x):
        # Steps alternate odd, even, odd, even, ... starting with the odd group.
        parity = 1 - len(steps) % 2
        count = len(group_terms(n, parity))
        steps.append((parity, np.asarray(x[i : i + count])))
        i += count
    return steps


def apply_step(mps, parity, angles):
    """Apply one group of bond gates, each with its own angle."""
    # Sweep away from the nearer end, as MPS.apply_group does, so the
    # orthogonality centre moves as little as possible.
    bonds = group_terms(mps.n, parity)
    if mps.centre <= mps.n // 2:
        order = range(len(bonds))
        to = "right"
    else:
        order = range(len(bonds) - 1, -1, -1)
        to = "left"
    for k in order:
        mps.apply_bond(bonds[k], bond_unitary(angles[k]), to)


def perbond_state(n, x, chi, cutoff=1e-10):
    """The ansatz state on an MPS (bond dimension limit chi)."""
    # Start from singlets on pairs (0,1), (2,3), ...; every gate keeps S = 0.
    mps = MPS.singlet_product(n, chi)
    mps.cutoff = cutoff
    for parity, angles in split_angles(n, x):
        apply_step(mps, parity, angles)
    return mps


def perbond_energy_grad(x, n, chi, chi_phi=None, cutoff=1e-10):
    """Energy and exact adjoint gradient of the per-bond ansatz on an MPS.

    Returns (energy, gradient, discarded weight of the forward pass).
    """
    # phi = H|psi> has a larger bond dimension than psi, hence the default 2 chi.
    chi_phi = chi_phi or 2 * chi
    steps = split_angles(n, x)
    start = MPS.singlet_product(n, chi)
    start.cutoff = cutoff
    # Forward pass: keep every intermediate state for the backward pass.
    states = [start]
    for parity, angles in steps:
        nxt = states[-1].copy()
        apply_step(nxt, parity, angles)
        states.append(nxt)
    final = states[-1].copy()
    energy = final.energy()
    # H|psi_final> needs more room than |psi_final> itself.
    phi = apply_mpo(states[-1].copy(), heisenberg_mpo(n), chi_phi, cutoff)
    phi.chi_max = chi_phi
    grad = np.zeros(len(x))
    # Offsets of each step inside the flat vector.
    offsets = np.cumsum([0] + [len(a) for _, a in steps])
    for k in range(len(steps) - 1, -1, -1):
        parity, angles = steps[k]
        vals = bond_overlaps(phi, states[k + 1], group_terms(n, parity), BOND)
        grad[offsets[k] : offsets[k + 1]] = 2 * np.imag(vals)
        # Undo this step on phi (inverse gates: negative angles).
        apply_step(phi, parity, -angles)
    return energy, grad, states[-1].discarded
