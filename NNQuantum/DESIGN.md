# DESIGN.md

Deeper analysis backing `CLAUDE.md`. Same rule applies: the two parts below are independent analyses of `old_version/` and `QuantumDynamics/framework`, kept strictly separate. No mapping or translation between them is proposed here.

---

## Part 1 — `old_version/` (`ML_QM`) in detail

### Struct inventory (`definitions.jl`)

- `ho{ho_basis}`: `basis, a, ad, n, Id`. Built by `Harmonic_oscillator(N_particle::Int64, type_basis::Symbol)` — `type_basis` is looked up in a one-entry `Dict(:FockBasis => FockBasis)`, i.e. a dictionary dispatch used even where there's only one option.
- `qubit`: `basis, σm, σp, σx, σy, σz, Id`, all typed to `SpinBasis{1//2,Int64}` specifically (not generic spin). Built by `Qubit(spin)`.
- `qub_ho{ho_basis}`: the tensor product, with every commonly-needed composite operator pre-built as a field: `zI, xI, mI, pI, II` (qubit operator ⊗ oscillator identity), `pa, mad` (Jaynes-Cummings-style coupling terms: `σp⊗a`, `σm⊗ad`), `Ia, Iad` (qubit identity ⊗ oscillator ladder), `n_mech, n_qubit` (number operators, `n_qubit` built as `0.5*(II + zI)` — i.e. the same `(id+σz)/2` convention). Built by `Qubit_HO(N_mech, type_basis_mech::Symbol, type_basis_qubit)`, returning `(qub, mech_res, qub_ho_struct)` as a 3-tuple.
- Pipeline-config structs, all loosely typed (`Any` fields, validated by convention not the type system): `dataset_features` (`problem::Symbol` is the dictionary-dispatch key; `len_dataset=[n_samples,n_training]`; `dim_dataset=[n_input,n_output]`; parameter-space bounds/dims; sampling probability function `pr`; `dynamics::Symbol`; `t0`; `initial_state`; `problem_features`; `names_dataset`; `modality_dataset` (`:generating`/`:generating_and_saving`/import); `norm_dataset`), `step_states` (initial/target kets+density matrices for one protocol step), `fl_1step_features` (target states, decomposition basis, phonon number, `correction::Symbol`), `fl_1step_nn_features_` (`model::Flux.Chain`, `N_epochs`, `η`, `optimizer`, `features_prediction::Vector{Any}` — a positional bundle consumed by the prediction function, `loss_func::Symbol`, `type_prediction::Symbol`).

### Physical constants (`definitions.jl`)

```julia
const h    = 6.62607015e-34
const hbar = 1.054571817e−34   # note: the character before "34" reads as a Unicode minus (U+2212),
                                # not ASCII "-" — worth double-checking if this file is ever
                                # actually `include`d and evaluated, since Julia numeric literals
                                # need an ASCII sign in the exponent.
const kb   = 1.380649e-23
```

### HBAR-qubit problem (`HBAR-qubit_problem/HBAR-qubit_problem.jl`)

Parameters (Chu et al.): `ωm = 5.9614e6`, `ωq = 5.9456e6` (labeled `[KHz]`), `γm = 0.025` (mechanical dissipation rate), `Teq = kb/(2π·hbar)·1e-3·10e-3`, `nthm = 1/(exp(ωm/Teq)-1)` (mechanical bath thermal occupation), `κ = 19` (qubit decay), `κϕ = 0.25` (qubit dephasing), `Δ0 = ωq - ωm`. `basis = tensor(SpinBasis(1//2), FockBasis(N_mech))`; `qub, mech_res, qubit_mech = Qubit_HO(N_mech, :FockBasis, 1//2)` — both executed at file-include time, so `N_mech` must already be a bound global before this file is included.

Rotating-frame Hamiltonian (frame rotating at `ωm`), shared structure across all three variants:
```julia
H_JC = g * (qubit_mech.pI*qubit_mech.Ia + qubit_mech.mI*qubit_mech.Iad)
H0   = 0.5 * Δ0_tilde * qubit_mech.zI + H_JC
```
where `Δ0_tilde = Δ0` normally, or `Δ0 + 2χ(n_phonon+0.5)` with `χ=g²/Δ0` if `typeofcorrection == :correction_on` (only wired up for the base `:FL_1step` variant — `:FL_1step_3p`'s correction block is present but commented out).

Time-dependent part built as a **mutated `LazySum`**, not `QuantumOptics.jl`'s `TimeDependentSum`:
```julia
Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(carrier_arg*t)   # carrier_arg = Δ0_tilde for :FL_1step, ωd for :FL_1step_3p
Δ(t) = -Δ0_tilde * π_pulse_shape(t, t0+τ_exc, τ_SWAP)
Ht = LazySum([Ω(t0), Δ(t0)/2], [qubit_mech.xI, qubit_mech.zI])
function Hamiltonian(t, ψ)
    Ht.factors[1] = Ω(t)
    Ht.factors[2] = Δ(t)/2
    return H0 + Ht
end
```
`:FL_1step_2drives` extends this to three mutated factors (`Ω1, Ω2, Δ`), with `Ω1`/`Ω2` built from BSpline-composed coefficients (`Bspline_composition(coeffs, basis)` returns `x -> spline(x)`; `drive_from_normalized_spline(spline, T)` remaps `t∈[0,T]` to the spline's `[0,1]` domain) instead of a fixed `cos` carrier.

Dissipators (four fixed jump operators, always the same shape regardless of variant):
```julia
[sqrt(γm*(nthm+1))*qubit_mech.Ia, sqrt(γm*nthm)*qubit_mech.Iad, sqrt(κϕ/2)*qubit_mech.zI, sqrt(κ)*qubit_mech.mI]
```
plus their `dagger.(...)` — both the operators and their daggers are returned from `create_..._dynamics`, i.e. the daggers are computed once up front rather than by the solver.

Two-stage runner (`FLstep_dynamics`/`FLstep_dynamics_3p`/`FLstep_2drives_dynamics` — three near-identical copies, one per variant): builds one Hamiltonian+dissipator set via `create_..._dynamics`, then calls `dynamic_evolution` twice — once over `[t0, t0+τ_exc]` (spin-flip stage), once over `[t0+τ_exc, t0+τ_exc+τ_SWAP]` (SWAP stage), feeding the first stage's final state into the second. Branches on `typeofdynamics ∈ {:master,:master_dynamic}` (dissipative, density-operator path, passes `(H,J,J†)` as the "state derivative input") vs. else (unitary, ket path, passes `H` alone). A second, separate `modeofdynamics::Symbol` argument controls what's returned: `:dynamics` → `(ρ_at_flip, ρ_at_end)`; `:final_state` → `(t_end, ρ_at_end)`; `:all_dynamics` → full concatenated `(tspan, ρ_out)`; anything else (the default, used for plotting) → `(tspan, ρ_out, τ_exc, τ_SWAP)`. Four different return shapes from one function, selected by a runtime symbol.

### Dataset/NN plumbing (`ML_QM_library.jl`)

No `using` statements in this file at all — it relies entirely on whatever the including script/notebook has already loaded (`QuantumOptics`, `PlotlyJS`, `StatsBase`, `Distributions`, `CSV`, `DataFrames`, `Flux`, `ProgressMeter`, `BSplineKit`, ...). Nothing at its own top level requires those beyond `QuantumOptics`-typed function signatures; everything else is only touched inside function bodies, so it works as long as the *caller* did the right `using`s first.

- `dynamic_evolution(time, ψ0, dynamics_input, type_dynamics::Symbol)`: `tspan = time[1]:time[2]:time[end]`; dict-dispatches `type_dynamics` to one of `timeevolution.{schroedinger,schroedinger_dynamic,master,master_dynamic}`; always calls with `adaptive=false, dt=time[2], reltol=1e-9, abstol=1e-9` — i.e. fixed-step integration regardless of what tolerance is requested (the tolerance kwargs are dead in the `adaptive=false` case, since `OrdinaryDiffEq`'s fixed-step solvers ignore them).
- `π_pulse_shape(t,t0,duration,eps=1e-12)`: `sin(π·δt/duration)² / (sin(...)² + eps²)` inside the window, `0.0` outside — the `eps`-regularized ratio forces the value to exactly `1.0` at the window's interior peak and decays the usual sin² shape elsewhere, while the `if 0≤δt<duration` branch forces an exact, non-decaying `0.0` outside (rather than relying on the sin² shape's own zeros, which coincide with the window edges only at those exact points).
- `qo_infidelity`: `1 - min(real(fidelity(ρ,σ)), 1)`, methods for both `Ket`/`Ket` (wraps into density operators first) and `Operator`/`Operator` pairs; `in_qo_infidelity` maps it over a state vector against one fixed target.
- Parameter-space sampling: `twoD_parameter_space`/`threeD_parameter_space` build a dense grid (`LinRange` per linear dimension, `logrange` for the third dimension in the 3D case) then evaluate a caller-supplied probability function `p` pointwise over it; `..._NN_outputs` then does a weighted `sample(..., Weights(prob), n_samples)` from that grid (`FL_1step`/`FL_1step_3p`), or rejection-samples continuously (`FL_1step_2drives`'s `_NN_outputs`, with optional log-uniform sampling per dimension).
- NN input features, same recipe for every variant: expectation values of the final state on a random orthonormal Hermitian operator basis (`rand_hermitian_orthonormal_basis(d, bases)` — generalizes Gell-Mann matrices to arbitrary dimension via QR-orthogonalized random Hermitian matrices) **concatenated with** the spin-flip-stage infidelity and the full-protocol infidelity vs. target.
- Normalization: separate min-max (`normalize_data`/`denormalize_data`/`max_min`) and standardization (`standardize_data`/`mean_variance`) utilities; `train_test_dataset_normalization` fits min-max stats on the **training set only**, applies to both train/test (an alternate fit-on-everything version is present but commented out).
- Training: `training!` does one `Flux.withgradient`+`Flux.Optimisers.update!` step **per individual sample** (not batched), for `N_epochs` full passes; loss function selected from a `Dict{Symbol,Function}` including two custom ones (`relative_mse`, `mape`) and one bespoke `loss_1` (a weighted sum of `|ŷᵢ-yᵢ|^0.5` over the first two output dims specifically — not general-purpose).
- `execute_problem_NN(dataset_features, NN_features)` is the top-level orchestrator: dataset generation/import (branch on `modality_dataset`) → optional normalization → `train_test_prediction!` (train → test → `prediction_func = D_predictions[NN_features.type_prediction]`, currently only `:prediction_infidelity` registered) → returns `(dataset_vector, predicted_parameters, predicted_final_state, infidelity_prediction)`.

---

## Part 2 — `QuantumDynamics/framework` in detail

### Subsystem construction and caching (`src/subsystems.jl`)

Every `AbstractSubsystem` stores `name::Symbol`, `ω::Float64`, its local `basis`, and `ops::NamedTuple` — computed once, at construction, never recomputed. `ops` being a `NamedTuple` (not individual struct fields) lets each subsystem type expose a different key set while staying type-stable per type. `Qubit(name,ω)`: `SpinBasis(1//2)`, `ops=(σx,σy,σz,σp,σm,n,h0,id)`, `n=(id+σz)/2`, `h0=ω*n`. `HarmonicOscillator(name,ω;nmax=10)`: `FockBasis(nmax)`, `ops=(a,ad,n,h0,id)`, `n=number(basis)` (QuantumOptics's own — the reference convention `Qubit`'s `n` was deliberately aligned to), `h0=ω*n`. `Transmon(name,ω,α;nmax=4)`: same basis type as `HarmonicOscillator` but default `nmax=4` (anharmonicity detunes higher levels increasingly out of relevance), `h0 = ω*n + (α/2)*n*(n-id)` (`α` negative for a physical transmon), `n` still the plain number operator so it participates in excitation-number-conservation arguments identically to `HarmonicOscillator`.

### Composite systems (`src/composite.jl`)

`CompositeSystem(subs...)`: `basis = tensor(...)`; `index::Dict{Symbol,Int}`; `ops::Dict{Symbol,NamedTuple}` built by `embed(basis, index[name], local_op)` for **every** operator of **every** subsystem, eagerly, once. Rationale: embedding (building the tensor-product matrix) is the expensive step, strictly more so than local operator construction — doing it once up front means Hamiltonian builders and, critically, a time-dependent solver's per-step callback never repeat it. Duplicate subsystem names throw `ArgumentError` at construction specifically because every later lookup goes through the plain `Dict` `index`, which would otherwise silently let the later subsystem shadow the earlier one (unreachable by name, but still occupying a Hilbert-space factor) with no error.

**Import gotcha, load-bearing**: `embed`/`ptrace` are owned by `QuantumInterface`, only re-exported by `QuantumOptics`. A bare `using QuantumOptics` followed by a local `function embed(new_sig...) = ...` does **not** add a method to the existing generic function — Julia silently creates a new, separate `embed` local to the defining module, shadowing the real one (`methods(embed)` would show only 1 method instead of the full multi-method function). `QuantumDynamics.jl` fixes this with an explicit `import QuantumOptics: embed, ptrace` before any `include`.

### Hamiltonian recipes (`src/hamiltonians.jl`)

All lab-frame (`bare_hamiltonian(cs) = Σ h0`, so a coupled system's bare energy is always `ωq·nq + ωo·no`, never a rotating-frame-reduced form). Qubit-like coupling dispatch (`_lower`/`_raise`, pattern-matching `getsubsystem(cs,name)`'s type: `σm`/`σp` for `Qubit`, `a`/`ad` for `Transmon`) is shared by `jaynes_cummings`, `rabi`, `tavis_cummings` — works because both operator pairs satisfy the same algebraic relation the conservation arguments need: `[n,lower]=-lower`, `[n,raise]=+raise`. `tavis_cummings` checks `length(g)==length(qubit_names)` when `g` isn't scalar (`ArgumentError` otherwise) specifically because `zip`ping mismatched-length iterables would otherwise silently drop trailing entries. `dispersive_hamiltonian` computes `χ=g²/Δ` from the subsystems' own `ω`s (not caller-supplied) and rejects `Δ=0` (`ArgumentError`) since that's not a degenerate corner of the approximation's validity domain, it's the fully-resonant regime the approximation doesn't apply to at all — left unguarded, `χ` would silently blow up to `Inf`/`NaN`. `quadratic_coupling` (`g(a+ad)²σz`) is `Qubit`-only like `dispersive_hamiltonian` (no `σz` analogue for a multi-level `Transmon`) and deliberately does not conserve excitation number (`(a+ad)²` mixes `a²`/`ad²` terms, Δn=±2) — used as a sanity control against the JC conservation test in `test/runtests.jl`, the same role `rabi`'s non-conservation plays.

`add_time_dependence(H, terms::Pair...; init_time=0.0) = TimeDependentSum(1.0=>H, terms...; init_time)` — deliberately not named `add_drive`, since an added term need not be an external drive; it may be a rotating-frame artifact (a counter-rotating term left over after a frame transformation) with no classical field behind it. Every term's operator must already be resolved (`op(cs,name,key)` or a fixed sum/product of cached ops) — `set_time!` (called every solver step) only re-evaluates scalar coefficient closures and re-forms the weighted sum; it never rebuilds an operator.

### Dissipation (`src/dissipation.jl`)

Three `Dissipator` subtypes, each validating `rate≥0` (and `nth≥0` where applicable) via its **sole inner constructor** — deliberately not an outer `Real`-typed guard method, because a bare `Float64` argument dispatches to a struct's auto-generated default inner constructor ahead of any outer method, so an outer-only guard would silently not run. `_decay_op`/`_gain_op`/`_dephasing_op` are separate dispatch helpers from `_lower`/`_raise` (not reused) specifically because decay/gain/dephasing are physically meaningful for a bare `HarmonicOscillator` too (cavity photon loss, thermal absorption), while `_lower`/`_raise` are documented as strictly "qubit-like." `Gain`'s `nth` has no default (unlike `Decay`'s `nth=0.0`) since a bare `Gain` with no paired `Decay` in mind isn't a meaningful thing to construct on its own — but `nth=0` is allowed (produces a zero jump operator) so a temperature sweep down to `nth=0` never needs a special case. `Dephasing`'s `sqrt(rate/2)` factor (vs. `Decay`'s plain `sqrt(rate*(nth+1))`) is chosen so `rate` is directly the coherence decay rate `1/Tφ` for a qubit (`c=sqrt(rate/2)σz` ⇒ `c†c=(rate/2)I` ⇒ `dρ01/dt=-rate·ρ01`); the same operator form applied to `HarmonicOscillator`/`Transmon` (`sqrt(rate/2)*n`) does **not** have that same clean property since `n²∝I` fails for `n` (kept for API symmetry, not because the rate means one universal thing there).

### Solver wrapper (`src/evolution.jl`)

`evolve` structurally dispatches: presence of `J` selects `master`/`master_dynamic` vs `schroedinger`/`schroedinger_dynamic`; the type of `H` (`AbstractOperator` vs `AbstractTimeDependentOperator`) selects dynamic vs static — no `QuantumDynamics`-level flag. Defaults `alg=Vern9()`, `reltol=1e-8`, `abstol=1e-10` (`QuantumOptics.jl`'s own: `DP5()`,`1e-6`,`1e-8`). Two **distinct** documented adaptive-solver failure modes on a time-dependent `H`, requiring different fixes:
1. **Narrow kick** — a short feature the solver's step size (grown large over a preceding smooth stretch) can jump clean over without any sample point landing inside it; both embedded error-estimate orders then agree (on data that already missed the feature), so the step is silently accepted. Neither tighter tolerance nor higher order fixes this (the comparison never saw the feature); fix is `tstops` (cheapest, needs the timing known in advance), `dtmax` (caps step size everywhere), or fixed `dt` with `adaptive=false` (most expensive, no error-based check left running at all).
2. **Sustained fast oscillation** — a carrier seen by every step, not aliased away, but under-resolved per-cycle by a low-order method, accumulating small per-cycle phase error into an O(1) bias over many cycles. Tighter tolerance and higher order **do** fix this — exactly why `Vern9`/tight tolerance is the default.

Post-solve check: warns (`maxlog=1`, not an error) if final norm² (closed) or trace (open) drifts from 1 by more than `NORMALIZATION_ATOL=1e-4`; a correct solve at defaults typically drifts ~`1e-7`, so this reliably flags under-resolution. A custom `fout` returning something other than a state passes through unchecked.

### Measurement (`src/measurement.jl`)

`condition_on(cs,state,name,projector) → (ρ_cond, prob)` — `prob` returned explicitly (not left for the caller to recompute) because silently dividing by a near-zero `prob` would otherwise produce an uninformative `NaN`/`Inf` `ρ_cond`. `check_fock_cutoff` is `Ket`-only, deliberately: under a bath, real steady-state population near a cutoff boundary is expected physics, not a truncation artifact, and the function can't tell the two apart, so it refuses density operators outright rather than risk a false positive (or a loosened tolerance that then hides genuine truncation). `levels` defaults to 3: **2** is the structural minimum (closes the blind spot where a single top level can sit at exactly zero population for parity reasons under a Δn=2 coupling, regardless of true convergence), **3** is one extra margin on top, not derived from any specific Hamiltonian's leakage pattern — a coupling that leaks by more than 2 levels at a time isn't guaranteed to be caught by the default. Bounded by `levels ≤ (nmax+1)÷2` (`ArgumentError` otherwise) so the checked region can't swallow the whole basis and become vacuously always-convergent on a small `nmax`.

### Persistence (`src/io.jl`)

Stores `subsystems`/`dissipators` (small, fully reconstructible specs) rather than `cs`/`J` (large, 100%-derived from those specs) — same "cache vs. spec" distinction `CompositeSystem` makes internally, just at the persistence layer. Time-independent `H` round-trips exactly (JLD2 handles a plain sparse `Operator` natively); time-dependent `H` only round-trips if its coefficient is a **named** top-level function — an anonymous closure (the pattern every time-dependent example in the repo actually uses) throws `MethodError: ... are not callable` on reload, a loud failure by design, consistent with the rest of the codebase's preference for surfacing problems (`condition_on`'s `prob`, `FockCutoffTooSmall`) over producing a quietly-wrong result. Recommended pattern: keep drive parameters in `params`, leave `H` as `nothing`/a fixed reference operator, and rebuild the real `TimeDependentSum` from `params` after `load_result`. Each field is written as a separate top-level JLD2 key (not one serialized struct blob) plus a `format_version`, so a future schema change can be handled explicitly on load rather than depending on JLD2 correctly reconstructing an evolved struct shape.
