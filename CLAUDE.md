# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

Everything this project actually consists of lives under `old_version/` (`ML_QM_*.jl`, `definitions.jl`, `problem_prototype.jl`, `HBAR-qubit_problem/`) — despite the folder name, this is the current and only codebase, not a deprecated one. `QuantumDynamics/` (formerly `external/`), at the repo root alongside this file, is a vendored third-party framework pulled in via `git subtree`/squash merge (see git log). It is unrelated to the Julia code here; do not read it for context and do not edit it as part of work on this project.

All paths below are relative to `old_version/`.

## What this is

`ML_QM` is a Julia framework (built on `QuantumOptics.jl` + `Flux.jl`) that does two things:
1. Runs quantum trajectory simulations (unitary/Schrödinger or dissipative/Lindblad dynamics) for a defined quantum system.
2. Uses those trajectories to build supervised-learning datasets and train/test/predict with a neural network (e.g. predicting pulse parameters needed to reach a target state, then scoring the result by state infidelity).

`HBAR-qubit_problem/` is the one concrete physical system currently implemented on top of this framework: a mechanical resonator (HBAR) dispersively/JC-coupled to a superconducting qubit (parameters from Chu et al.), with a two-stage pulse protocol (spin-flip, then SWAP).

## Commands

There is no root Julia environment — the only `Project.toml`/`Manifest.toml` live in `old_version/HBAR-qubit_problem/`. Work from that directory:

```julia
using Pkg
Pkg.activate(".")      # from old_version/HBAR-qubit_problem/
Pkg.instantiate()      # installs pinned deps from Manifest.toml
Pkg.status()
```

Run a workflow via the Julia REPL (or the notebook `old_version/HBAR-qubit_problem/Chu_DFL_execution.ipynb`) by including files in dependency order — this order matters, see "Include order" below:

```julia
include("../definitions.jl")
include("HBAR-qubit_problem.jl")
include("../ML_QM_library.jl")
# ... define dataset_features / NN_features (see ../ML_QM_execution.jl) ...
execute_problem_NN(dataset_features, NN_features)
```

There is no build step, no linter, and no automated test suite in this repo — validation is done by running the notebook/scripts interactively and inspecting plots/infidelity values.

## Architecture

**Layered, dictionary-dispatched design.** The root files form a problem-agnostic library; `HBAR-qubit_problem/` is a concrete plug-in registered into that library via two global `Dict`s in `ML_QM_library.jl`:

- `dataset_problems_dictionary[problem_symbol] => [inputs_fn, outputs_fn]` — used by `dataset_generation` to turn a `:problem` tag into the pair of functions that sample pulse parameters and simulate the resulting trajectories.
- `plot_problem_dictionary[problem_symbol] => dynamics_fn` — used by `trajectory_plot` to know which dynamics function to call for a given problem when plotting/predicting.

To add a new physical problem: implement the four functions sketched in `problem_prototype.jl` (`XXXXX_NN_outputs`, `XXXX_NN_inputs`, `XXXXX_dynamics`, `create_XXXXX_dynamics` — see `HBAR-qubit_problem.jl`'s `FL_1step_*` functions for a working example of each), then add entries for the new problem symbol to both dictionaries in `ML_QM_library.jl`.

**Include order is load-bearing, not just style.** `HBAR-qubit_problem.jl` calls functions defined later, in `ML_QM_library.jl` (e.g. `dynamic_evolution`, `π_pulse_shape`, `qo_infidelity`, `in_qo_infidelity`, `generate_Bspline_basis`). This works because Julia only resolves function bodies at call time, but it means: `definitions.jl` → problem file → `ML_QM_library.jl` must all be included before any function is actually *called*, even though the problem file textually comes before the library.

**Heavy reliance on pre-existing globals.** Problem files (e.g. `HBAR-qubit_problem.jl`) reference module-level globals that must already exist when the file is included or when its functions are called — e.g. `N_mech`, `hbar`, `kb` (from `definitions.jl` / set by the execution script before including), and `basis`, `g`, `n_basis_spline`, `basis_spline` (set in the execution script). There's no dependency injection — new problem files follow the same convention of assuming certain globals are set up by the execution script first.

**Two-struct-family data flow:**
- QM-side structs (`definitions.jl`): `ho`, `qubit`, `qub_ho` wrap `QuantumOptics.jl` bases/operators for a harmonic oscillator, a spin-1/2 qubit, and their tensor product, built via `Harmonic_oscillator(...)`, `Qubit(...)`, `Qubit_HO(...)`.
- Pipeline-config structs, problem-specific but following the `Local_variable1/2/3` contract documented in `ML_QM_library_Documentation.txt`: `dataset_features` (dataset generation config, includes a `problem::Symbol` used for dictionary dispatch), `fl_1step_nn_features_`/`NN_features` (model/training config), plus problem-specific ones like `step_states`, `fl_1step_features`.

**End-to-end entry point:** `execute_problem_NN(dataset_features, NN_features)` in `ML_QM_library.jl` chains: dataset generation/import (`dataset_generation` or `importing_dataset`, per `dataset_features.modality_dataset`) → optional min-max normalization (`train_test_dataset_normalization`, per `dataset_features.norm_dataset`) → `train_test_prediction!` (trains via `training!`, tests via `testing_model_infidelity`, then predicts via the function named in `NN_features.type_prediction`, e.g. `prediction_infidelity`) → returns the dataset, predicted parameters, predicted final state, and infidelity vs. the target state.

**`FL_1step` problem family.** `HBAR-qubit_problem.jl` implements three variants of the same two-pulse (spin-flip + SWAP) protocol, registered under different problem symbols: `:FL_1step` (2 free parameters, fixed Rabi frequency relation `Ω_R = π/τ_exc`), `:FL_1step_3p` (adds a free drive frequency `ωd`), `:FL_1step_2drives` (both pulses parameterized by BSpline coefficients instead of a fixed pulse shape — needs `basis_spline`/`n_basis_spline` set up first). Each variant has its own `..._NN_outputs` (samples pulse parameters), `..._NN_inputs` (runs the dynamics and builds NN input features from Gell-Mann-basis expectation values + infidelities), `...step_dynamics`, and `create_..._dynamics` (builds the time-dependent Hamiltonian + Lindblad operators) functions.

## Reference docs already in the repo

- `READ ME - ML_QM.txt` — step-by-step workflow guide for defining a new QM problem and running the NN pipeline.
- `ML_QM_library_Documentation.txt` — function-by-function reference for `ML_QM_library.jl`, including the field contracts for `Local_variable1`/`Local_variable2`/`Local_variable3`.
