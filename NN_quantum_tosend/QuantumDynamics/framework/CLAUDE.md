# QuantumDynamics

A Julia framework for cavity QED / circuit QED quantum optics simulations,
built on top of [`QuantumOptics.jl`](https://qojulia.org/). QuantumDynamics does not
reimplement Hilbert-space or solver machinery — it adds a thin, consistent
layer for defining reusable subsystems (qubits, oscillators), combining them
into composite systems, and building common Hamiltonians (Jaynes-Cummings,
Rabi, Tavis-Cummings, dispersive approximation) from them.

For full design rationale, API details, and the specific numerical
conventions this code relies on (in particular a qubit sign convention that
is easy to get backwards — see `design.md`), read **[design.md](design.md)**
before making changes to `src/`.

## Layout

```
src/
  QuantumDynamics.jl — module entry point: includes + exports
  subsystems.jl     — AbstractSubsystem, Qubit, HarmonicOscillator, Transmon
  composite.jl      — CompositeSystem (joins subsystems into one Hilbert space), embed(cs,...)
  measurement.jl    — ptrace(cs,...), condition_on (project + trace + renormalize, by subsystem name)
  hamiltonians.jl   — bare_hamiltonian, jaynes_cummings, rabi,
                        tavis_cummings, dispersive_hamiltonian, quadratic_coupling, add_time_dependence
  dissipation.jl    — Decay, Gain, Dephasing, jump_operator(s), thermal_bath
  evolution.jl      — evolve (closed/open, time-(in)dependent solver dispatch)
  io.jl             — SimulationResult, save_result/load_result (JLD2-backed)
test/
  runtests.jl       — Test.jl suite (run via `Pkg.test()`)
examples/
  conditionally_squeezed_states/ — reproduces Fig. 1, 2 & 4 of arXiv:2504.01664,
                                     own Project.toml/environment (see its README.md)
  fock_state_preparation/        — sequential Fock-state preparation via the
                                     JC ladder, own Project.toml/environment
                                     (see its README.md)
  resonant_fock_state_preparation/ — same JC-ladder protocol, but with the
                                     qubit and oscillator always resonantly
                                     coupled (no detuning at all); own
                                     Project.toml/environment (see its
                                     README.md)
design.md           — detailed specs and design decisions
```

## Core ideas (see design.md for why)

- Subsystems (`Qubit`, `HarmonicOscillator`) cache their operators at
  construction time instead of recomputing them on demand.
- `CompositeSystem` eagerly embeds every subsystem operator into the joint
  Hilbert space once, at construction — Hamiltonian builders and
  time-dependent solver callbacks then only do cheap sparse-matrix algebra
  on already-embedded operators, never repeat `embed()`.
- Every subsystem exposes a uniformly-defined excitation number operator
  `n` (0 = ground state), so bare energies are always `ω * n` regardless of
  subsystem type.
- Time-dependent Hamiltonians use `QuantumOptics.jl`'s `TimeDependentSum`:
  cached operators + cheap scalar coefficient functions of `t`, never
  operators rebuilt per solver step.

## Running tests

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

## Status

Hamiltonian construction, dissipation, and solver wrappers are implemented
and tested (137 passing tests). Includes a `Transmon` subsystem with
anharmonicity, usable anywhere a `Qubit` is (Jaynes-Cummings, Rabi,
Tavis-Cummings); a qubit-only `quadratic_coupling` recipe for
optomechanical-type `(a+ad)²σz` coupling (alongside the dispersive
approximation); `add_time_dependence` for turning any static Hamiltonian
into a `TimeDependentSum` by adding one or more time-dependent terms
(operators must already be resolved, e.g. via `op(cs, name, key)`) — named
generically since such a term need not be an external drive (it may be a
rotating-frame artifact instead, see `design.md`);
`Decay`/`Gain`/`Dephasing` dissipators that build collapse
operators the same way Hamiltonians are built (dispatched per subsystem
type) — `Decay`+`Gain` compose into finite-temperature baths via a shared
`(rate, nth)` pair, or via the `thermal_bath(name, rate, nth)` convenience
that keeps the pair in sync; a single `evolve` entry point covering
closed/open and time-independent/time-dependent evolution
(`schroedinger`/`schroedinger_dynamic`/`master`/`master_dynamic`) — which
overrides QuantumOptics' `DP5`/`1e-6` solver defaults with `Vern9`/`1e-8`
(silently inaccurate otherwise on some time-dependent `H`) and warns if the
final state's norm/trace drifted from 1, both covered under "Solver wrappers"
in `design.md`; and a
first state-level API (`embed`/`ptrace` by subsystem name, and
`condition_on` for the project-trace-renormalize pattern behind conditional
state preparation/readout, returning `(ρ_cond, prob)` so a near-zero-
probability outcome is visible rather than silently blowing up to
`NaN`/`Inf`) — `CompositeSystem` itself still manages
operators only, so these live in `measurement.jl` rather than
`composite.jl`. `measurement.jl` also has `check_fock_cutoff`, a
closed-system (`Ket`-only) pre-flight guard that throws
`FockCutoffTooSmall` when population near a Fock subsystem's truncation
boundary exceeds tolerance, so an under-sized `nmax` fails loudly instead
of silently biasing results. `examples/conditionally_squeezed_states/`
reproduces Fig. 1, 2, and 4 of arXiv:2504.01664 using these abstractions,
and is where `add_time_dependence` and the state-level helpers were first
validated before being promoted into `src/`.
`examples/fock_state_preparation/` prepares a target Fock state by climbing
the Jaynes-Cummings ladder one excitation at a time, exercising
`jaynes_cummings`/`add_time_dependence`/`evolve` via a long, multi-segment
pulse train built as one continuous `TimeDependentSum`.
`examples/resonant_fock_state_preparation/` is a standalone variant of that
protocol where the qubit and oscillator are always resonantly coupled
(no `Δ(t)` at all, unlike the sibling example) — the coupling term is a
genuinely time-independent operator, and only a carrier-free qubit drive
`Ω(t)*σx` is switched, in tanh-top-hat plateaus timed to be fast compared to
the local vacuum-Rabi period (`g√N*w ≪ 1`, checked numerically in that
example's `leakage_scan.jl`). Notable pitfall this example surfaced: an
adaptive ODE solver can silently step over a pulse plateau much narrower
than the surrounding evolution without erroring, once its step size has
grown large over a long smooth interval — `run_fock_prep` there passes an
explicit `dtmax` (tied to the pulse's own tanh edge width) to `evolve` to
guard against this. A `SimulationResult` type
(`src/io.jl`) plus `save_result`/`load_result` (JLD2-backed) let a
completed trajectory — times, states, the subsystems/dissipators needed to
reconstruct `cs`/`J`, the Hamiltonian, and free-form description/params —
be persisted and reloaded independently of the session that produced it,
for the "long-running simulation now, post-process later" workflow; see
"Result persistence" in `design.md` for what is and isn't safely
round-trippable (notably: time-dependent Hamiltonians built from anonymous
closures, the pattern `add_time_dependence`'s own examples use, do not
survive a reload — named coefficient functions do). Not yet implemented: a
transmon-specific dispersive approximation, additional subsystem types,
parameter-sweep infrastructure, quantum-trajectory solves, and saving
expectation-value trajectories as an alternative to full states — see "Not
yet implemented" in `design.md`.
