# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This file describes only this folder, `NNQuantum/`, a sibling of `old_version/` and `QuantumDynamics/` at the repo root. `NNQuantum/` currently contains **no code** — only this analysis. It is meant to become a new framework that replaces `old_version`'s dynamics engine with `QuantumDynamics`, but that design work has not started yet; this file and `DESIGN.md` are the groundwork for it.

The two parts below analyze `old_version/` and `QuantumDynamics/framework/` **strictly separately** — as two independent, unrelated frameworks, each on its own terms. Nothing here proposes a mapping, translation, or design decision that connects them; that comes later, once both are understood on their own.

---

## Part 1 — `old_version/` (`ML_QM`)

A Julia framework (`QuantumOptics.jl` + `Flux.jl`) that inverts a quantum control problem: given a target quantum state, find the pulse parameters that drive a system to it. Two decoupled halves connected only by the dataset:
- **Dynamics engine** — pure physics. Builds a time-dependent Hamiltonian (and Lindblad operators, for dissipative runs) from pulse parameters and integrates the trajectory (`dynamic_evolution` in `ML_QM_library.jl`, dict-dispatched over `:schroedinger`/`:schroedinger_dynamic`/`:master`/`:master_dynamic`, fixed-step `dt`, `adaptive=false`, `reltol=abstol=1e-9`).
- **Learning engine** — treats the dynamics engine as a black-box data generator: samples pulse parameters, simulates, builds a supervised dataset (state features → pulse params), trains a `Flux.Chain`, and uses it to predict pulses for a target state, then re-simulates and scores by infidelity.

**Extension mechanism**: no abstract problem interface — dispatch is two global `Dict{Symbol,Function}` (`dataset_problems_dictionary`, `plot_problem_dictionary`) mapping a `problem::Symbol` to `[inputs_fn, outputs_fn]` / a plotting-dynamics function. A new problem supplies four functions (`problem_prototype.jl` is the commented-out template): `..._NN_outputs` (samples pulse parameters), `..._NN_inputs` (runs dynamics, builds NN feature vectors), `..._dynamics` (runs the evolution), `create_..._dynamics` (builds the Hamiltonian closure + Lindblad operators).

**QM-side structs** (`definitions.jl`): `ho{ho_basis}` (harmonic oscillator: `basis,a,ad,n,Id`), `qubit` (spin-1/2: `basis,σm,σp,σx,σy,σz,Id`), `qub_ho{ho_basis}` (tensor product with pre-built composite operators: `zI,xI,mI,pI,II,pa,mad,Ia,Iad,n_mech,n_qubit`), built via `Harmonic_oscillator`/`Qubit`/`Qubit_HO`. No caching abstraction beyond this — every problem hand-picks which pre-tensored operator it needs.

**Time-dependent Hamiltonians**: built as a `LazySum` whose `factors` are mutated in place inside a `Hamiltonian(t,ψ)` closure (needed for performance under `timeevolution.schroedinger_dynamic`/`master_dynamic`, which call it every solver step) — not `QuantumOptics.jl`'s newer `TimeDependentSum`.

**The one worked example**: `HBAR-qubit_problem/` — a mechanical resonator (HBAR) Jaynes-Cummings-coupled to a qubit (Chu et al. parameters), Hamiltonian written in the frame rotating at the mechanical frequency `ωm` (so the qubit's bare energy collapses to the detuning `Δ0 = ωq-ωm` and the mechanical bare term vanishes entirely). Two-stage "FL_1step" pulse protocol (spin-flip, then SWAP), in three registered variants differing only in how the pulse is parameterized: `:FL_1step` (2 params, fixed `Ω_R=π/τ_exc`, drive carrier `cos(Δ0_tilde·t)`), `:FL_1step_3p` (3 params, free drive frequency `ωd` replacing the fixed carrier — its optional detuning-correction term `χ=g²/Δ0` is present in the code but commented out, so it's currently always inactive), `:FL_1step_2drives` (both stages independently BSpline-shaped via `generate_Bspline_basis`/`Bspline_composition`/`drive_from_normalized_spline`, needs a `basis_spline` set up first).

**Never run with concrete numbers**: `N_mech` (mechanical Fock cutoff) is a `...` placeholder in the unfinished template `ML_QM_execution.jl`; the JC coupling `g` is commented out (`#g = 258`) in `HBAR-qubit_problem.jl`. There is no known-good numerical output anywhere in the repo to validate against.

**Self-documented unfinished/fragile bits**: dissipative branches marked `#to be fixed`; `dynamics_n_steps_FL` (multi-step chaining) marked `# fix it`; `ML_QM_execution.jl` is a non-runnable template.

**No tests, no linter.** Validated by running `HBAR-qubit_problem/Chu_DFL_execution.ipynb` interactively and inspecting plots/infidelity values. Single Julia environment, rooted at `HBAR-qubit_problem/` (~24 deps: `Flux`, `DifferentialEquations`, `Zygote`, `PlotlyJS`, `CSV`, `DataFrames`, `BSplineKit`, `Distributions`, `StatsBase`, `ProgressMeter`, ...) — loaded together on every run regardless of whether the learning engine is actually used.

---

## Part 2 — `QuantumDynamics/framework`

A general, **tested** (137 passing tests) Julia framework for cavity/circuit QED simulation, built as a thin organizational layer over `QuantumOptics.jl`. Not an ML framework — no dataset/training concept anywhere in it. Reimplements no Hilbert-space or solver machinery; adds reusable subsystems, a composite-system abstraction, Hamiltonian recipes, dissipation channels, and one hardened `evolve` entry point.

**Subsystems** (`subsystems.jl`): `AbstractSubsystem` supertype. `Qubit(name,ω)`, `HarmonicOscillator(name,ω;nmax=10)`, `Transmon(name,ω,α;nmax=4)` — each **precomputes and caches** its operator set (`ops::NamedTuple`) at construction, including a uniformly-defined excitation number `n` (0 = ground, integer-valued for every subsystem type) and a bare local Hamiltonian `h0` (`ω*n`, plus an anharmonic correction for `Transmon`). Load-bearing sign-convention gotcha: `Qubit`'s `n = (id+σz)/2`, not `-σz`, because `QuantumOptics.jl`'s `sigmam` treats spin-up as the excited state — getting this backwards silently breaks excitation-number conservation in coupling terms while still looking Hermitian and well-formed. Uniform accessor: `op(s, key)`.

**Composite systems** (`composite.jl`): `CompositeSystem(subs...)` tensors several subsystems into one joint Hilbert space and **eagerly embeds every subsystem operator** into it once, at construction (not lazily) — the expensive step happens once, not on every Hamiltonian build or solver step. Subsystem names must be unique (checked, `ArgumentError` otherwise). `op(cs,name,key)` looks up a cached embedded operator; `embed(cs,name,local_op)` embeds an arbitrary one-off operator. Extending `QuantumOptics.jl`'s own `embed`/`ptrace` requires an explicit `import` (not just `using`), documented as a easy-to-get-wrong gotcha.

**Hamiltonian recipes** (`hamiltonians.jl`), all built purely from cached embedded operators (lab frame, i.e. bare energies are `ωq·nq + ωo·no`, always): `bare_hamiltonian` (Σ h0); `jaynes_cummings`/`rabi`/`tavis_cummings` (RWA / non-RWA / multi-qubit coupling, via a `_lower`/`_raise` dispatch that treats `Qubit` and `Transmon` uniformly as "qubit-like"); `dispersive_hamiltonian` (qubit-only, `χ=g²/Δ`, rejects `Δ=0`); `quadratic_coupling` (qubit-only, `g(a+ad)²σz`, optomechanical-type, does not conserve excitation number). `add_time_dependence(H, coeff=>op, ...)` turns any static `H` into a `TimeDependentSum` — design rule: coefficients carry the time dependence, operators are always one of the cached, fixed ones.

**Dissipation** (`dissipation.jl`): `Decay(name,rate;nth=0)`, `Gain(name,rate;nth)` (no default `nth` — only meaningful paired with `Decay`), `Dephasing(name,rate)` — each a typed `Dissipator` producing one rate-scaled jump operator via `jump_operator`/`jump_operators`, dispatched per subsystem type. `thermal_bath(name,rate,nth)` builds a matched `[Decay,Gain]` pair so the two can't desync. `Dephasing`'s jump operator is `sqrt(rate/2)*op` specifically so `rate` is directly `1/Tφ` (documented, tested convention).

**Solver wrapper** (`evolution.jl`): a single `evolve(tspan, state, H[, J])` dispatches on presence of `J` (open/closed) and the type of `H` (time-dependent or not) to the matching `QuantumOptics.timeevolution.*` function. Overrides `QuantumOptics.jl`'s own `DP5()`/`1e-6`/`1e-8` defaults with `Vern9()`/`1e-8`/`1e-10`, because `DP5` can silently accumulate O(1) phase error on a Hamiltonian with a sustained fast oscillation (a documented, tested failure mode). Warns (doesn't error) if the final state's norm/trace drifts from 1 by more than `1e-4`. Also documents a *second*, distinct adaptive-solver failure mode — a short, narrow pulse feature the solver's step size can jump straight over without ever sampling it — which tighter tolerance/higher order does **not** fix; the documented fix is `tstops`/`dtmax`/fixed `dt`.

**Measurement** (`measurement.jl`): `ptrace`, `condition_on` (project+trace+renormalize, returns `(ρ_cond, prob)` so a near-zero-probability outcome is visible rather than silently `NaN`), `check_fock_cutoff` (`Ket`-only pre-flight truncation guard).

**Persistence** (`io.jl`): `SimulationResult`/`save_result`/`load_result` (JLD2-backed), stores the minimal reconstructible spec (`subsystems`, `dissipators`) rather than the derived, larger `cs`/`J`; time-dependent `H` only round-trips reliably if its coefficient is a named function, not an anonymous closure (documented, tested limitation).

**Dependencies**: `QuantumOptics.jl`, `OrdinaryDiffEqVerner` (for `Vern9`), `JLD2` — three, deliberately minimal. **137 passing tests** (`test/runtests.jl`), including sanity-control tests that verify the *absence* of a property (e.g. `rabi`/`quadratic_coupling` genuinely don't conserve excitation number) to prove the corresponding positive test isn't vacuous. Three worked examples under `examples/`, each with its own environment.
