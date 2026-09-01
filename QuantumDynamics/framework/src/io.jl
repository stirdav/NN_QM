const RESULT_FORMAT_VERSION = 1

"""
    SimulationResult

Bundles a completed simulation's trajectory together with everything needed
to make sense of it later — the structural setup that produced it, not just
the raw numbers. Meant for the "long-running simulation now, post-process
later" workflow: build one with [`SimulationResult`](@ref)`(cs, times,
states; ...)`, hand it to [`save_result`](@ref), and reload it with
[`load_result`](@ref) in a completely separate session.

- `times`, `states` — the trajectory itself. `states` may hold `Ket`s or
  density `Operator`s (whichever `evolve` returned), saved as-is;
  coarse-graining in time (e.g. `times[1:10:end]`, `states[1:10:end]`) is the
  caller's job before constructing a `SimulationResult`, not something this
  type or [`save_result`](@ref) does for you. **Not currently supported:**
  saving a *reduced* trajectory (e.g. `ptrace`ing out a subsystem before
  saving). `subsystems` is always the full list `cs` was built from, so
  `CompositeSystem(r)` always reconstructs the full-dimension composite
  system — if `states` were ptrace'd down to fewer subsystems first,
  `CompositeSystem(r)` and `jump_operators(CompositeSystem(r), r.dissipators)`
  would silently reconstruct operators of the wrong dimension for those
  states, with nothing in the saved file recording that a reduction
  happened. Representing a reduced trajectory (e.g. a `retained_subsystems`
  field) is deferred — see "Not yet implemented" in design.md.
- `subsystems` — *not* the `CompositeSystem` itself. `CompositeSystem`
  eagerly embeds every operator into the joint Hilbert space at construction
  (see `composite.jl`); those embedded operators are 100% reconstructible
  from the subsystem list alone (`CompositeSystem(subsystems...)`), so
  saving them too would just be redundant, potentially large, disk usage.
  Use the `CompositeSystem(::SimulationResult)` constructor below to rebuild
  `cs` on load.
- `dissipators` — the `Vector{Dissipator}` passed to [`jump_operators`](@ref)
  to build `J`, for the same reason: small, self-describing, and
  `jump_operators(cs, dissipators)` rebuilds `J` after reconstructing `cs`.
  Empty for a closed-system run.
- `H` — the Hamiltonian actually evolved. A time-independent `Operator`
  round-trips exactly. A `TimeDependentSum` (see "Time-dependent
  Hamiltonians" in design.md) is saved on a best-effort basis only: JLD2
  serializes a coefficient function by name, so a *named* top-level function
  (`mydrive(t) = ...`) reloads as a fully working callable, but an
  *anonymous closure* (`t -> Ω*cos(ωd*t)`, the pattern `add_time_dependence`
  is actually documented and used with throughout this codebase) reloads as
  an inert placeholder that carries the captured values and enclosing
  function name for a human to read, but is not callable — reusing it after
  reload throws `MethodError`, it does not silently misbehave. For a
  time-dependent run, prefer recording the drive parameters in `params` and
  rebuilding `H` with [`add_time_dependence`](@ref) after `load_result`; pass
  `H=nothing` if there is nothing worth saving as a fixed reference.
- `description` — free-text note for a human (e.g. which recipe/frame `H`
  came from) — not parsed or relied on by any code here, purely a hint
  alongside the structured `params`.
- `params` — free-form physical parameters (`g`, `Δ`, drive amplitude, ...).
  Hamiltonian recipes (`jaynes_cummings`, `rabi`, ...) take these as plain
  arguments and return a matrix, with nothing reified as data the way
  `Dissipator` is — so, unlike `subsystems`/`dissipators`, there is no
  automatic way to recover "which recipe, which arguments" from `H` alone.
  This is where the caller records that, structured for later use (as
  opposed to `description`, which is unstructured).
"""
struct SimulationResult
    times::Vector{Float64}
    states::Vector
    subsystems::Vector{AbstractSubsystem}
    dissipators::Vector{Dissipator}
    H::Union{Nothing,AbstractOperator}
    description::String
    params::Dict{Symbol,Any}
end

"""
    SimulationResult(cs, times, states; dissipators=Dissipator[], H=nothing, description="", params=Dict{Symbol,Any}())

Convenience constructor: pulls `subsystems` out of `cs` (see
[`SimulationResult`](@ref)'s docstring for why `cs` itself isn't stored)
instead of requiring the caller to pass `cs.subsystems` directly.
"""
SimulationResult(cs::CompositeSystem, times, states;
    dissipators=Dissipator[], H=nothing, description::AbstractString="", params=Dict{Symbol,Any}()) =
    SimulationResult(collect(float.(times)), collect(states), cs.subsystems, collect(dissipators), H, String(description), params)

"""
    CompositeSystem(r::SimulationResult)

Rebuild the `CompositeSystem` that produced `r`, from its saved
`subsystems` — the same construction `evolve` originally ran against, cheap
to redo since it's only re-embedding a handful of small local operators
(see [`SimulationResult`](@ref)'s docstring). Jump operators, if any, follow
with `jump_operators(cs, r.dissipators)`.
"""
CompositeSystem(r::SimulationResult) = CompositeSystem(r.subsystems...)

"""
    save_result(path, r::SimulationResult)

Write `r` to `path` (a `.jld2` file) via `JLD2.jl`. Fields are stored as
separate top-level keys — `times`, `states`, `subsystems`, `dissipators`,
`H`, `description`, `params`, plus a `format_version` — rather than one
serialized `SimulationResult` blob. This keeps old files loadable if the
struct gains fields later: [`load_result`](@ref) can fall back on a missing
key (e.g. a not-yet-existing `expectations` field), whereas JLD2
reconstructing a whole struct whose shape changed hits the same "type
doesn't exist in workspace, reconstructing" fallback documented on `H`
above — worth avoiding for `SimulationResult` itself, not just for drive
coefficients.
"""
function save_result(path::AbstractString, r::SimulationResult)
    jldsave(path;
        format_version=RESULT_FORMAT_VERSION,
        times=r.times,
        states=r.states,
        subsystems=r.subsystems,
        dissipators=r.dissipators,
        H=r.H,
        description=r.description,
        params=r.params,
    )
    nothing
end

"""
    load_result(path)::SimulationResult

Read back a [`SimulationResult`](@ref) saved by [`save_result`](@ref).
"""
function load_result(path::AbstractString)
    jldopen(path, "r") do f
        SimulationResult(f["times"], f["states"], f["subsystems"], f["dissipators"], f["H"], f["description"], f["params"])
    end
end
