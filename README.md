# Coupling-Aware QAOA for Maximum-Likelihood Detection in Near-Field XL-MIMO at 140 GHz

Reference implementation and numerical simulation code for coupling-aware Quantum Approximate Optimization Algorithm (QAOA) applied to maximum-likelihood (ML) QPSK detection in 140-GHz near-field extremely large-scale MIMO (XL-MIMO) systems.

---

## Overview

This repository contains the simulation code and benchmarks accompanying the research paper: **"Coupling-Aware QAOA for Maximum-Likelihood Detection in Near-Field XL-MIMO at 140 GHz: NISQ Noise and Resource Analysis"**. 

It investigates how dense Ising graphs induced by correlated near-field XL-MIMO channels can be pruned using a coupling-aware top-$K$ strategy. This reduces two-qubit interaction requirements on Noisy Intermediate-Scale Quantum (NISQ) processors while providing a deterministic surrogate-optimality bound on the full ML objective.

---

## Key Features

- **Exact QUBO & Ising Mapping**: Converts complex-valued ML objectives into exact real-valued binary quadratic programs and Ising Hamiltonians with explicit coefficient conventions.
- **Coupling-Pruned Surrogate Objective**: Implements top-$K$ pairwise interaction pruning with a derived deterministic surrogate-optimality bound ($2\epsilon_K$) on the full objective.
- **NISQ Noise Simulation**: Simulates gate-level two-qubit depolarizing channels via stochastic Pauli trajectories for scalable noise sensitivity analysis without full density-matrix storage overhead.
- **Validation & Resource Analysis**: Includes exact statevector simulation code for 8-qubit validation ($128 \times 4$ system) and analytical resource scaling projections up to 32 qubits.

---

## Repository Structure

- `simulation.py`: Core simulation script containing the geometry-driven 140-GHz near-field channel model, exact ML enumeration, MMSE-SIC baseline, dense QAOA, and coupling-pruned QAOA.
- `noise_study.py`: Stochastic gate-level two-qubit depolarizing noise simulation script.
- `requirements.txt`: Python package dependencies.

---

## Requirements & Installation

The core numerical implementation relies on standard scientific Python libraries and does not require heavy quantum SDKs for statevector evaluations:

```bash
pip install numpy scipy matplotlib
