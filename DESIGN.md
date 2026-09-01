# DESIGN.md

Architecture reference for the `ML_QM` framework and the `HBAR-qubit_problem` example built on it. All of it lives under `old_version/` (the folder name is historical — this is the current codebase, not deprecated). `QuantumDynamics/` (formerly `external/`), at the repo root, is a vendored third-party framework, out of scope for this document.

## 1. Purpose

`ML_QM` is a Julia framework combining `QuantumOptics.jl` (quantum dynamics) and `Flux.jl` (neural networks) to solve a specific class of problem: *find the control-pulse parameters that drive a quantum system from a known initial state to a target state, using a neural network trained on simulated trajectories.*

Concretely, for a given physical system and pulse protocol, it:
1. Samples candidate pulse parameters from a parameter space.
2. Simulates the resulting quantum trajectory (unitary or dissipative) for each sample.
3. Builds a supervised dataset: input = features of the resulting state (expectation values on a decomposition basis + intermediate infidelities), output = the pulse parameters that produced it.
4. Trains a NN to invert this map (state features → pulse parameters).
5. Uses the trained NN to predict the pulse parameters for an actual target state, re-simulates with those parameters, and reports the infidelity of the achieved state vs. the target.

## 2. Repository layout

```
old_version/
  definitions.jl                    QM structs (ho, qubit, qub_ho) + pipeline-config structs
  problem_prototype.jl               Commented-out template for defining a new problem
  ML_QM_library.jl                   Generic engine: dynamics, datasets, NN train/test/predict, dictionaries
  ML_QM_execution.jl                 Template driver script (mostly placeholders) showing run order
  READ ME - ML_QM.txt                User-facing workflow guide
  ML_QM_library_Documentation.txt    Function-by-function API reference

  HBAR-qubit_problem/
    HBAR-qubit_problem.jl            Concrete problem: HBAR-qubit "FL_1step" pulse protocol
    Chu_DFL_execution.ipynb          Notebook that runs the pipeline for this problem
    Project.toml / Manifest.toml     Julia environment (only one in the repo)

QuantumDynamics/                    Vendored third-party framework (formerly external/) — out of scope, repo-root level
```

There is exactly one Julia environment in the repo, rooted at `old_version/HBAR-qubit_problem/`.

## 3. Core abstractions

### 3.1 Quantum-system structs (`definitions.jl`)

Thin wrappers around `QuantumOptics.jl` bases/operators, built by constructor-like functions:

- `ho{ho_basis}` — harmonic oscillator: `basis`, `a`, `ad`, `n`, `Id`. Built by `Harmonic_oscillator(N_particle, type_basis)`.
- `qubit` — spin-1/2 system: `basis`, `σm`, `σp`, `σx`, `σy`, `σz`, `Id`. Built by `Qubit(spin)`.
- `qub_ho{ho_basis}` — tensor product of the two (qubit ⊗ oscillator): pre-built composite operators (`zI`, `xI`, `mI`/`pI` for qubit ladder ⊗ identity, `pa`/`mad` for Jaynes-Cummings-style coupling terms, `Ia`/`Iad` for identity ⊗ oscillator ladder, `n_mech`/`n_qubit` number operators). Built by `Qubit_HO(N_mech, type_basis_mech, type_basis_qubit)`, which also returns the underlying `qubit` and `ho` structs.

This pattern (raw basis → struct of pre-tensored operators) is what every new physical system is expected to follow, keeping the frequently-used composite operators pre-computed rather than re-tensored on every call.

### 3.2 Pipeline-config structs

Loosely typed "bag of config" structs, passed through the pipeline instead of long positional argument lists. Documented generically in `ML_QM_library_Documentation.txt` as `Local_variable1`/`2`/`3`; concretely defined per-problem in `definitions.jl`:

- `dataset_features` — dataset generation config: `problem::Symbol` (dictionary dispatch key), `len_dataset` (`[n_samples, n_training]`), `dim_dataset` (`[n_input, n_output]`), parameter-space bounds/dims, sampling probability `pr`, `dynamics` type, `t0`, `initial_state`, problem-specific `problem_features`, dataset save names, `modality_dataset` (`:generating`/`:generating_and_saving`/import), `norm_dataset`.
- `fl_1step_nn_features_` (the `NN_features` used at the call site) — NN config: `model` (a `Flux.Chain`), `N_epochs`, `η`, `optimizer`, `features_prediction` (positional bundle consumed by the prediction function), `loss_func`, `type_prediction`.
- Problem-specific: `step_states` (initial/target kets and density matrices for a protocol step), `fl_1step_features` (target states, decomposition basis, phonon number, correction mode).

These structs are intentionally loosely typed (`Any` fields) — validation is by convention, not the type system.

## 4. Extension mechanism: adding a new problem

The framework has no abstract problem interface; dispatch is done through two global `Dict{Symbol, ...}` in `ML_QM_library.jl`:

```julia
dataset_problems_dictionary[:my_problem] = [my_problem_NN_inputs, my_problem_NN_outputs]
plot_problem_dictionary[:my_problem]     = my_problem_dynamics
```

`problem_prototype.jl` is the template for the four functions a new problem must supply (see `HBAR-qubit_problem.jl`'s `FL_1step_*` functions for a filled-in example of each):

| Function | Role |
|---|---|
| `..._NN_outputs(p, parameters_range, dim_parameters_space, n_samples)` | Samples `n_samples` pulse-parameter tuples from the parameter space, weighted by probability function `p`. This becomes the NN's *output* (training target). |
| `..._NN_inputs(t0, initial_state, dataset_features, pulse_parameters, n_samples, typeofdynamics)` | For each sampled pulse, runs the dynamics and builds the NN's *input* feature vector (typically: expectation values on a Hermitian decomposition basis + intermediate-step infidelities). |
| `..._dynamics(t0, initial_state, pulse_parameters, typeofdynamics, problem_features, modeofdynamics)` | Runs the actual time evolution (via `dynamic_evolution`), branching on unitary (`:schroedinger`/`:schroedinger_dynamic`) vs. dissipative (`:master`/`:master_dynamic`) dynamics. `modeofdynamics` controls what's returned: `:dynamics` (two key states), `:final_state`, `:all_dynamics` (full trajectory), or default (full trajectory + pulse timings, used for plotting). |
| `create_..._dynamics(t0, pulse_parameters, typeofcorrection, n_phonon)` | Builds the time-dependent Hamiltonian closure (mutating a `LazySum`'s factors to avoid reallocating) plus the Lindblad jump operators and their conjugates. |

**Important:** include order matters. Julia only resolves function bodies at call time, so a problem file can be `include`d *before* `ML_QM_library.jl` even though it calls library functions (`dynamic_evolution`, `π_pulse_shape`, `qo_infidelity`, `in_qo_infidelity`, `generate_Bspline_basis`, ...) — as long as the library is included before those functions are actually invoked. Problem files also assume certain globals are already bound when included (e.g. `N_mech`, `hbar`, `kb` from `definitions.jl`, and system-specific ones like `basis`, `g`, `n_basis_spline`, `basis_spline`) — there is no dependency injection, so new problem files should follow the same convention and the execution script must set these globals up first.

## 5. End-to-end pipeline (`execute_problem_NN`)

```
execute_problem_NN(dataset_features, NN_features)
  │
  ├─ modality_dataset == :generating            → dataset_generation(dataset_features)
  ├─ modality_dataset == :generating_and_saving  → dataset_generation(...) + saving_dataset(...)
  └─ otherwise                                   → importing_dataset(dataset_features.names_dataset)
  │
  ├─ norm_dataset == :normalized → train_test_dataset_normalization(...)   [min-max, per-column]
  │                                 (stores maxs/mins into NN_features.features_prediction[end]
  │                                  for later de-normalization at prediction time)
  │
  └─ train_test_prediction!(dataset_vector, dataset_features, NN_features)
       ├─ training!(model, optimizer, N_epochs, loss_func, train_X, train_Y)
       │     per-sample gradient step via Flux.withgradient + Flux.Optimisers.update!
       ├─ testing_model_infidelity(model, loss_func, test_input, test_output)
       │     mean loss over the test set + raw predictions
       └─ prediction_func = D_predictions[NN_features.type_prediction]   (e.g. prediction_infidelity)
             ├─ normalize the real prediction input (if norm_dataset == :normalized)
             ├─ model(input) → denormalize → predicted pulse parameters
             ├─ trajectory_plot(dataset_features, problem_features, predicted_params)
             │     dispatches to plot_problem_dictionary[problem] to re-simulate + plot
             └─ qo_infidelity(achieved_state, target_state)

  Returns: dataset_vector, predicted_parameters, predicted_final_state, infidelity_prediction
```

`dataset_generation` itself (`ML_QM_library.jl`) is the glue that looks up the input/output function pair from `dataset_problems_dictionary[dataset_features.problem]`, calls the output-sampler then the input-builder, assembles rows via `dataset_creation`, and splits into train/test using `len_dataset = [n_samples, n_training]`.

## 6. Worked example: `HBAR-qubit_problem`

Physical system: a mechanical resonator (HBAR) Jaynes-Cummings-coupled to a superconducting qubit, with mechanical and qubit dissipation baths (parameters from Chu et al.). Built as `basis = tensor(SpinBasis(1//2), FockBasis(N_mech))`, `qub, mech_res, qubit_mech = Qubit_HO(...)`.

Protocol ("FL_1step"): a two-stage pulse —
1. **Spin-flip** stage (duration `τ_exc`, Rabi rate `Ω_R`): drives the qubit.
2. **SWAP** stage (duration `τ_SWAP`): swaps qubit and mechanical excitation via the JC coupling, brought into resonance by nulling the detuning.

Three registered variants, differing only in how the pulse is parameterized:

| Problem symbol | Parameters | Notes |
|---|---|---|
| `:FL_1step` | `[τ_exc, τ_SWAP]` | `Ω_R` fixed to `π/τ_exc` (a π-pulse); drive detuned at `Δ0_tilde`. |
| `:FL_1step_3p` | `[τ_exc, ωd, τ_SWAP]` | Adds a free drive frequency `ωd` instead of using `Δ0_tilde`. |
| `:FL_1step_2drives` | BSpline coefficients (×2) + `[Ω_R1, Ω_R2, τ_exc, τ_SWAP]` | Both pulse envelopes are shaped by BSpline curves (`generate_Bspline_basis`/`Bspline_composition`) instead of the fixed `sin²` π-pulse shape, evaluated on `[0, T]` via `drive_from_normalized_spline`. |

All three share `π_pulse_shape` (sine-squared window) for gating, and build a `LazySum` Hamiltonian whose factors are mutated in place at each time step (`Hamiltonian(t, ψ)` closures) rather than reconstructing the sum — this is required for performance under `timeevolution.schroedinger_dynamic`/`master_dynamic`, which call the Hamiltonian function at every solver step.

The NN input features for all variants are: expectation values of the final state on a random orthonormal Hermitian decomposition basis (`rand_hermitian_orthonormal_basis`, generalizing Gell-Mann matrices to arbitrary dimension) + the spin-flip-stage infidelity + the full-protocol infidelity vs. target.

## 7. Self-documented open items

These are marked directly in the code as unfinished/fragile, worth checking before relying on them:

- `FLstep_dynamics` / `create_FLstep_dynamics_3p` etc.: dissipative branch is commented `#to be fixed`.
- `dynamics_n_steps_FL` (multi-step chaining of the protocol) is marked `# fix it`.
- `create_FLstep_dynamics_3p`'s correction-term block (`χ`, `Δ0_tilde`) is commented out — 3-parameter variant currently always uses uncorrected `Δ0`.
- `ML_QM_execution.jl` is a template with `...` placeholders throughout (not a runnable script as-is); a working run requires filling in every physical parameter, dataset size, and NN architecture section.
