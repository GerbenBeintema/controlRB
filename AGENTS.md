# AGENTS.md

## Project Purpose
`controlRB` is a research codebase for simulating and controlling 2D Rayleigh-Benard (RB) thermal convection.

The repository combines:
- A high-performance Lattice Boltzmann Method (LBM) simulator with CPU and CUDA backends.
- Python bindings and utilities for running, validating, and visualizing simulations.
- A thesis workspace with scripts and LaTeX sources documenting results on RL-based control of RB convection.

Primary scientific goal:
- Reduce convective heat transport (Nusselt number, Nu) by controlling bottom boundary temperature while keeping mean boundary temperature fixed.

## Core Components
- `controlRB/LBMd2q9ctypes.py`
  - Main Python interface (`lbmd2q9`) to compiled C/CUDA shared libraries via `ctypes`.
  - Builds backend libraries through `make` in `d2q9-cpu` or `d2q9-cuda`.
  - Supports multi-simulation batching (`NSIM`) and configurable physics parameters (`Ra`, `Pr`, boundary conditions, forcing).
- `d2q9-cpu/`
  - C implementation of the RB LBM solver.
- `d2q9-cuda/`
  - CUDA implementation of the same solver for acceleration.
- `controlRB/validation.py`
  - Validation routines for Poiseuille flow, thermal diffusion, and Nu-vs-Ra reference behavior.
- `controlRB/cuda_cpu_compare.py`
  - Regression checks for CPU/CUDA numerical agreement.
- `controlRB/misc.py`
  - Interactive experiments, control waveform utilities, plotting/movie helpers.
- `controlRB/scalings.py`
  - Timing/scaling experiments across domain size, steps, and number of parallel simulations.

## What The Thesis Folder Adds
The `thesis/` folder is not just documentation. It encodes the scientific framing, evaluation protocol, and figure-generation scripts for the same RB control problem.

Key thesis points relevant to this codebase:
- RL (PPO) is applied to boundary-temperature control of RB convection.
- Baselines include linear G and GD controllers and acceleration modulation.
- Reported metric is Nusselt number across Rayleigh numbers (`10^3` to `10^7`).
- Reported outcome: RL is comparable at low Ra and outperforms classical controllers at moderate/high Ra in the thesis experiments.

Important files:
- `thesis/src/chapters/results-control-RB.tex`: main RB-control results and interpretation.
- `thesis/src/appendices/main.tex`: LBM method details, validation setup, and RL environment formulation.
- `thesis/scripts/Ra-Nu-plot.py` and `thesis/scripts/Nu-time.py`: figure generation for Ra-Nu and time-series comparisons.

## Simulation and Control Context
The simulator models RB convection with no-slip boundaries and thermal forcing.
Control acts on bottom boundary temperature with constraints:
- Mean bottom temperature fixed (constant effective Ra setup).
- Amplitude-limited perturbations.

Canonical performance target:
- Drive Nu toward 1 (conductive state), or otherwise reduce Nu as much as possible when full stabilization is not feasible.

## Guidance For AI Coding Agents
When modifying this repository:
- Preserve parity between CPU and CUDA behavior whenever changing solver logic.
- Do not silently change physical parameter semantics (`Ra`, `Pr`, `TBOT0`, `TTOP0`, boundary handling).
- Prefer adding/using validation checks in `controlRB/validation.py` and `controlRB/cuda_cpu_compare.py` for numerical changes.
- Keep the Python-facing API of `lbmd2q9` stable unless explicitly refactoring all call sites.
- Treat thesis scripts as reproducibility artifacts; avoid breaking expected filenames/outputs in `thesis/figs/`.

## Practical Build/Run Notes
- Python package metadata is in `setup.py`.
- Native libs are built through `make` commands invoked by `LBMd2q9ctypes.py`.
- CUDA mode depends on available GPU/CUDA toolchain.
- Some plotting and notebook utilities assume optional dependencies (matplotlib, IPython, tqdm, scipy).

## Scope Clarification
This repository primarily provides:
- RB simulation infrastructure and validation.
- Analysis/figure tooling used in a thesis on RL control.

It does not contain a full, self-contained RL training framework in this tree; thesis scripts consume externally generated datasets for many result figures.
