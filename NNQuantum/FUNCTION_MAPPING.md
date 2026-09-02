# FUNCTION_MAPPING.md

Concrete function-by-function correspondence between `old_version`'s **dynamics engine only** (the physics half of `ML_QM` — struct builders in `definitions.jl`, the dynamics-related functions in `ML_QM_library.jl`, and the Hamiltonian/dissipator builders in `HBAR-qubit_problem/HBAR-qubit_problem.jl`) and `QuantumDynamics/framework`. This is the mapping `CLAUDE.md`/`DESIGN.md` deliberately deferred ("no mapping ... proposed here"); it's step one of designing `NNQuantum`. The learning engine (`Flux.jl` training/prediction machinery in `ML_QM_library.jl`) is out of scope here — it has no counterpart in `QuantumDynamics` (which has no ML concept at all) and would sit unchanged on top of whichever dynamics engine `NNQuantum` ends up using.

Read against source, not just the `CLAUDE.md`/`DESIGN.md` summaries, to keep exact factors/conventions honest.

---

## 1. System construction

| `old_version` | `QuantumDynamics` | Notes |
|---|---|---|
| `Harmonic_oscillator(N_particle, :FockBasis)` → `ho{basis,a,ad,n,Id}` | `HarmonicOscillator(name, ω; nmax=N_particle)` → caches `(a,ad,n,h0,id)` | Old struct has no `ω`/`h0` field — the bare energy is assembled by hand at the Hamiltonian call site every time. New struct threads `ω` through at construction and caches `h0=ω*n` once. |
| `Qubit(spin)` → `qubit{basis,σm,σp,σx,σy,σz,Id}` | `Qubit(name, ω)` → caches `(σx,σy,σz,σp,σm,n,h0,id)` | Old struct doesn't cache `n`/`h0` at all — `n_qubit` only appears later, hand-built in `qub_ho`. New struct's `n=(id+σz)/2` is byte-for-byte the same convention old version uses for `n_qubit = 0.5*(II+zI)` — **the sign convention is already consistent between the two codebases**, a real point in favor of the swap being low-risk on this axis. |
| `Qubit_HO(N_mech, :FockBasis, spin)` → `(qub, mech_res, qub_ho{zI,xI,mI,pI,II,pa,mad,Ia,Iad,n_mech,n_qubit})` | `CompositeSystem(Qubit(:qubit,ωq), HarmonicOscillator(:osc,ωm; nmax=N_mech))` → `op(cs, name, key)` | Old version hand-picks and names 10 specific pre-tensored operators as struct fields — every new problem that needs a different combination edits `qub_ho`'s field list. New version eagerly embeds **every** operator of **every** subsystem generically at construction; any combination is available via `op(cs, name, key)` with no struct changes. Field-by-field: `zI`→`op(cs,:qubit,:σz)`, `xI`→`op(cs,:qubit,:σx)`, `mI`→`op(cs,:qubit,:σm)`, `pI`→`op(cs,:qubit,:σp)`, `Ia`→`op(cs,:osc,:a)`, `Iad`→`op(cs,:osc,:ad)`, `n_mech`→`op(cs,:osc,:n)`, `n_qubit`→`op(cs,:qubit,:n)`, `pa`(=`σp⊗a`)→`op(cs,:qubit,:σp)*op(cs,:osc,:a)`, `mad`(=`σm⊗ad`)→`op(cs,:qubit,:σm)*op(cs,:osc,:ad)`, `II`→ not needed explicitly (`bare_hamiltonian` sums `h0` directly, never needs a bare identity term). |

## 2. Hamiltonian construction

The old version's `create_..._dynamics` functions (three near-identical copies, one per `FL_1step` variant) build:

```julia
H_JC = g * (qubit_mech.pI*qubit_mech.Ia + qubit_mech.mI*qubit_mech.Iad)   # σp⊗a + σm⊗ad
H0   = 0.5 * Δ0_tilde * qubit_mech.zI + H_JC
```

This is structurally `jaynes_cummings(cs, :qubit, :osc, g)` — `bare_hamiltonian(cs) + g*(ad*lower + a*raise)` — **but in old version's rotating frame** (rotating at `ωm`), where the mechanical bare term has vanished entirely and the qubit's bare term has collapsed from `ωq*nq` to the detuning `Δ0_tilde/2 * σz`. `QuantumDynamics`'s Hamiltonian recipes are unconditionally **lab-frame** (`bare_hamiltonian = ωq*nq + ωo*no`, documented explicitly in `DESIGN.md` Part 2) — there is no rotating-frame builder.

**Gap, not a direct mapping**: to reproduce old version's `H0` exactly, you can't just call `jaynes_cummings` with the physical `ωq`/`ωm`. Two options, neither built into the framework today:
- Construct `Qubit(:qubit, Δ0_tilde)` and `HarmonicOscillator(:osc, 0.0)` (i.e. pre-transform the frequencies yourself before handing them to `QuantumDynamics`) and call `jaynes_cummings(cs,:qubit,:osc,g)`. This reproduces old version's `H_JC` term exactly, but `bare_hamiltonian` then gives `Δ0_tilde*nq = 0.5*Δ0_tilde*(id+σz)`, which differs from old version's `0.5*Δ0_tilde*σz` by a constant `0.5*Δ0_tilde*id` global-phase term — physically inert (a global phase never affects observables/fidelities) but not numerically identical if you diff Hamiltonians directly.
- Write the rotating-frame `H0` by hand from cached ops (`0.5*Δ0_tilde*op(cs,:qubit,:σz) + g*(op(cs,:qubit,:σp)*op(cs,:osc,:a) + op(cs,:qubit,:σm)*op(cs,:osc,:ad))`), bypassing `jaynes_cummings` — legal (the recipe functions are convenience wrappers over cached ops, not the only way to use them) but means `NNQuantum` would need its own rotating-frame Hamiltonian helper; none exists in `QuantumDynamics` today.

The optional `Δ0_tilde = Δ0 + 2χ(n+0.5)`, `χ=g²/Δ0` correction (only active for `:FL_1step`) is the same *idea* as `dispersive_hamiltonian`'s `χ=g²/Δ`, but mechanically different — old version folds `χ` into a detuning used inside a still-fully-JC Hamiltonian, it doesn't swap to the dispersive approximation the way `dispersive_hamiltonian` does (which eliminates the coupling term outright). Not a callable match, just a shared formula.

## 3. Time-dependent drive

Old version (`HBAR-qubit_problem.jl`), a **mutated `LazySum`** inside a `Hamiltonian(t,ψ)` closure:

```julia
Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(carrier_arg*t)
Δ(t) = -Δ0_tilde * π_pulse_shape(t, t0+τ_exc, τ_SWAP)
Ht = LazySum([Ω(t0), Δ(t0)/2], [qubit_mech.xI, qubit_mech.zI])
function Hamiltonian(t, ψ)
    Ht.factors[1] = Ω(t); Ht.factors[2] = Δ(t)/2
    return H0 + Ht
end
```

Maps directly onto `add_time_dependence`, which exists precisely to replace this pattern:

```julia
H = add_time_dependence(H0,
    t -> Ω(t)     => op(cs, :qubit, :σx),
    t -> Δ(t)/2   => op(cs, :qubit, :σz))
```

This is the cleanest 1:1 correspondence in the whole mapping — old version's "coefficients carry the time dependence, in-place-mutated for solver-step performance" pattern is exactly the design rule `TimeDependentSum`/`add_time_dependence` enforce structurally (`DESIGN.md` Part 2, `add_time_dependence` docstring), just done safely by the framework instead of by hand. `:FL_1step_2drives`'s three-factor BSpline-driven version (`Ω1,Ω2,Δ`) maps the same way, one `coeff=>op` pair per factor — the BSpline coefficient functions themselves (`Bspline_composition`, `drive_from_normalized_spline`) have no `QuantumDynamics` equivalent and would carry over unchanged as plain coefficient closures.

### Step-wise / multi-pulse sequencing

`HBAR-qubit_problem.jl` has two flavors of "more than one pulse in time," both of which `evolve` can reproduce — as a single continuous call, not one solver call per stage:

- **Simultaneous multi-term drive** (`FL_1step_2drives`'s `Ω1(t)*xI + Ω2(t)*xI + Δ(t)/2*zI`, three factors in one `LazySum`, [HBAR-qubit_problem.jl:643-649](../old_version/HBAR-qubit_problem/HBAR-qubit_problem.jl#L643-L649)): direct match for `add_time_dependence`'s variadic `coeff=>op` pairs — `add_time_dependence(H0, Ω1=>σx, Ω2=>σx, Δ=>σz)`. `TimeDependentSum` doesn't care that two terms share an operator (`σx` twice); it just sums `coeff(t)*op` per term, same as `LazySum`.
- **Sequential windows, chained solver calls** (`FLstep_dynamics`'s two-stage protocol — spin-flip then SWAP, two `dynamic_evolution` calls with the final state manually threaded from one into the next; and `dynamics_n_steps_FL`'s N-repetition chaining of the same pattern, itself marked `# fix it`/unfinished): `QuantumDynamics`'s own examples already demonstrate the intended replacement, not a hypothetical one. `examples/fock_state_preparation/` and `.../resonant_fock_state_preparation/` build a `Step` struct per stage, a schedule (`build_schedule`), and one coefficient function that's a **sum of windowed contributions across every step** (`envelope_fn`/`detuning_fn`, each stage's own `π_pulse_shape`-style window folded in per term) — then make exactly **one** `evolve` call over the full schedule's `tspan`, rather than one `evolve`/`dynamic_evolution` call per stage with manual state hand-off. `dynamics_n_steps_FL`'s N-step case generalizes the same way that example's `for n in 0:N-1` schedule-builder already does — those examples are effectively the working, generalized version of what `dynamics_n_steps_FL` was attempting.

**Caveat this raises, not present in old version's own approach**: old version's fixed-step (`adaptive=false, dt=1e-6`) integration can't miss a narrow window by construction (at the cost of no real error control at all — see §5). Collapsing HBAR's two/N stages into one continuous `evolve` call switches to an *adaptive* solver, and HBAR's short `τ_exc`/`τ_SWAP` windows are exactly the "narrow kick" failure mode `evolution.jl` documents — the step size can grow across a long quiet stretch and jump clean over the next short pulse with no error raised. The fix is the same one `resonant_fock_state_preparation/system.jl:73` already applies for its own short drive plateaus: pass `dtmax` (or `tstops` at each window boundary) tied to the narrowest pulse feature in the schedule. Skipping this would silently trade old version's "no error control" problem for a "solver misses the pulse entirely" problem — not an improvement.

If the intermediate boundary state is itself needed (old version's `ρ_at_flip`, used for spin-flip infidelity scoring mid-protocol), that doesn't require a second `evolve` call either: pass `tstops=[t0+τ_exc]` so the solver is guaranteed to land exactly there, then index into the single returned `states` array at that time.

## 4. Dissipation

Old version's fixed four-jump-operator list (plus manually precomputed daggers, both passed into `master_dynamic`):

```julia
[sqrt(γm*(nthm+1))*qubit_mech.Ia, sqrt(γm*nthm)*qubit_mech.Iad,
 sqrt(κϕ/2)*qubit_mech.zI, sqrt(κ)*qubit_mech.mI]   # + dagger.(...) of each
```

Maps exactly, factor-for-factor, onto `QuantumDynamics`'s three `Dissipator` types:

| Old jump operator | `QuantumDynamics` equivalent | Formula check |
|---|---|---|
| `sqrt(γm*(nthm+1)) * Ia` (mech. decay) | `Decay(:osc, γm; nth=nthm)` | `jump_operator` gives `sqrt(rate*(nth+1))*a` — identical |
| `sqrt(γm*nthm) * Iad` (mech. thermal gain) | `Gain(:osc, γm; nth=nthm)` | `sqrt(rate*nth)*ad` — identical |
| *(the pair above, together)* | `thermal_bath(:osc, γm, nthm)` | keeps `rate`/`nth` in sync, same as manually pairing the two above |
| `sqrt(κϕ/2) * zI` (qubit dephasing) | `Dephasing(:qubit, κϕ)` | `sqrt(rate/2)*σz` — identical, same `rate=1/Tφ` convention |
| `sqrt(κ) * mI` (qubit decay) | `Decay(:qubit, κ)` (default `nth=0`) | `sqrt(rate*(0+1))*σm = sqrt(κ)*σm` — identical |

All four factors match exactly — a strong validation signal that the two codebases already agree on Lindblad-operator conventions, independent of the framework swap. One simplification falls out for free: old version computes `dagger.(...)` of all four operators up front and threads `(H,J,J†)` through `master_dynamic` by hand; `QuantumDynamics`'s `evolve`/`jump_operators` only ever need `J` — `J†` is computed internally by `QuantumOptics.jl`'s `master`/`master_dynamic`, so that whole bookkeeping step disappears.

`jump_operators(cs, [Decay(:qubit,κ), Dephasing(:qubit,κϕ), Decay(:osc,γm;nth=nthm), Gain(:osc,γm;nth=nthm)])` reproduces old version's four-operator list (in embedded, ready-to-pass-to-`evolve` form) in one call.

## 5. Solver entry point

`dynamic_evolution(time, ψ0, dynamics_input, type_dynamics::Symbol)` ↔ `evolve(tspan, state, H[, J]; ...)`.

- Old version: `Dict{Symbol,Function}` dispatch on `type_dynamics ∈ {:schroedinger,:schroedinger_dynamic,:master,:master_dynamic}`, always called with `adaptive=false, dt=time[2], reltol=1e-9, abstol=1e-9` — the tolerance kwargs are dead code under `adaptive=false` (`OrdinaryDiffEq`'s fixed-step solvers ignore them), so old version's dynamics are, in practice, unchecked fixed-step Euler-grid integration with no real error control.
- New version: ordinary Julia method dispatch — presence of `J` selects `master`/`master_dynamic` vs. `schroedinger`/`schroedinger_dynamic`; the type of `H` (`AbstractOperator` vs. `AbstractTimeDependentOperator`) selects static vs. dynamic. Defaults to **actually adaptive** `Vern9()` at `reltol=1e-8`/`abstol=1e-10`, and warns (doesn't silently pass) if the final state's norm/trace drifts from 1 by more than `1e-4` — a check old version has no equivalent of anywhere.

Old version's `type_dynamics` symbol is redundant information once you're inside `evolve`'s type-dispatch: whether you have `J` and whether `H` is time-dependent are already implied by what you constructed, so the explicit `:schroedinger_dynamic`-style tag a caller has to get right by hand goes away.

`Quantum_solver_ODE(prob)` (bare `solve(prob, Tsit5())`) has no counterpart — it's a generic, disconnected ODE-solve utility not wired into anything QM-specific; nothing in `QuantumDynamics` reimplements raw `DifferentialEquations.jl` solving since `evolve` already owns that entry point.

## 6. Measurement / scoring

| Old version | `QuantumDynamics` | Notes |
|---|---|---|
| `expectation_value(op, states) = real(expect(op,states))` | *(none)* | Thin wrapper over `QuantumOptics.expect`; `QuantumDynamics` doesn't add a framework-level wrapper for this — call `QuantumOptics.expect` directly. |
| `qo_infidelity(ρ,σ)` / `in_qo_infidelity` | *(none)* | No infidelity/fidelity helper anywhere in `QuantumDynamics` — it has no ML/scoring concept. This function would need to be ported into `NNQuantum` as-is (it's a thin, generic wrapper over `QuantumOptics.fidelity`, so porting is trivial) or written fresh; it isn't something the physics framework swap provides. |
| *(none)* | `condition_on(cs,state,name,projector) → (ρ_cond, prob)` | New capability old version has no equivalent of — project+trace+renormalize by subsystem name, returning `prob` explicitly so a near-zero-probability outcome is visible rather than a silent `NaN`. Not a like-for-like mapping, just worth noting as available if `NNQuantum` needs conditional measurement. |
| *(none)* | `check_fock_cutoff` | Old version has no truncation-adequacy check anywhere (`N_mech` is picked by hand, never validated); new version's `Ket`-only pre-flight guard is a free correctness improvement with no old-version counterpart to map from. |

`dm2ket` (extract the pure state from a rank-1 density matrix via `eigenstates`) has no `QuantumDynamics` counterpart — general `QuantumOptics.jl` utility, not framework-specific either side.

## 7. No counterpart on either side (stays problem-level)

These are pulse-shaping/parameterization utilities, not physics-engine machinery — in both codebases they belong at the "problem" layer, not the framework layer, so there's nothing to map, only to carry forward unchanged into whatever `NNQuantum` problem module replaces `HBAR-qubit_problem.jl`:

- `π_pulse_shape(t,t0,duration,eps)` — the `sin²`-window pulse envelope.
- `generate_Bspline_basis` / `Bspline_composition` / `drive_from_normalized_spline` — BSpline drive parameterization for `:FL_1step_2drives`.

`QuantumDynamics/framework/examples/resonant_fock_state_preparation/` uses a conceptually similar tanh-plateau pulse shape at the example layer (not `src/`), reinforcing that pulse-envelope functions are expected to live in problem/example code in the new framework too, not in `src/`.

## 8. Structural/dispatch philosophy difference (not function-level, but load-bearing for the redesign)

Old version dispatches almost everything through global `Dict{Symbol,Function}`/`Dict{Symbol,Type}` lookups: `dataset_problems_dictionary`, `plot_problem_dictionary`, the `dynamics_map` inside `dynamic_evolution`, `basis_Dict` inside `Harmonic_oscillator`, the `loss_functions` dict, `D_predictions`. `QuantumDynamics` uses ordinary Julia multiple dispatch instead — `_lower`/`_raise`/`_decay_op`/`_gain_op`/`_dephasing_op` dispatch on the subsystem's *type* (`Qubit` vs. `HarmonicOscillator` vs. `Transmon`), and `evolve` dispatches on the *type* of `H`/presence of `J`. There is no direct function-to-function mapping for this — it's a design-pattern swap that falls out naturally once each old dictionary's *purpose* (pick dynamics function by problem tag → pick recipe/solver method by argument type) is reproduced using the new framework's types instead of symbols. Worth deciding explicitly for `NNQuantum`'s own problem-registration mechanism (i.e. whether a new `FL_1step`-like problem gets registered via a dict, same as old version, or via a `AbstractProblem` type hierarchy, matching `QuantumDynamics`'s own style).

## 9. Summary of genuine gaps (not just renamings)

Two things old version's dynamics engine does that `QuantumDynamics` has no ready-made equivalent for, and that `NNQuantum` would need to add:

1. **Rotating-frame Hamiltonians** (§2) — `QuantumDynamics`'s recipes are lab-frame only; the HBAR-qubit problem (and likely most control problems, where working in a frame that removes fast carrier terms is standard) needs a rotating-frame construction path that doesn't exist in `src/hamiltonians.jl` today.
2. **Infidelity/fidelity scoring** (§6) — needed by the learning engine's dataset-labeling and prediction-scoring steps; trivial to port (`qo_infidelity` is a few lines over `QuantumOptics.fidelity`) but not present in `QuantumDynamics`.

Everything else in the dynamics engine (system construction, JC-style coupling once in the right frame, time-dependent drive terms, all four dissipation channels, the solver entry point) maps cleanly, and in several places (dissipator factors, the qubit `n` sign convention, the time-dependent-Hamiltonian pattern) the two codebases already agree closely enough that the swap looks low-risk on physics grounds — the main design work left is rotating frames and re-attaching the (unmapped, ML-only) learning engine on top.
