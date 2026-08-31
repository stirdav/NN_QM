"""
    CompositeSystem(subsystems::AbstractSubsystem...)

Joins several subsystems into one Hilbert space (`basis = tensor(...)`) and
precomputes every subsystem operator embedded into that joint space. Embedding
(building the tensor-product matrix) is the expensive step, so it happens once
here at construction time rather than being repeated inside Hamiltonians or,
worse, inside a solver's per-step callback.

Subsystem `name`s must be unique — every lookup (`getsubsystem`, `op(cs,
name, key)`, `_lower`/`_raise`/`_decay_op`/etc.) resolves a name via a
`Dict{Symbol,Int}`, so a duplicate name would otherwise silently shadow the
earlier subsystem's slot (unreachable by name, though it still occupies a
factor of the joint Hilbert space) rather than erroring. Checked and throws
`ArgumentError` at construction.
"""
struct CompositeSystem
    subsystems::Vector{AbstractSubsystem}
    basis::CompositeBasis
    index::Dict{Symbol,Int}
    ops::Dict{Symbol,NamedTuple}
end

function CompositeSystem(subs::AbstractSubsystem...)
    names = [s.name for s in subs]
    length(names) == length(unique(names)) || throw(ArgumentError(
        "CompositeSystem: subsystem names must be unique, got $names",
    ))
    basis = tensor([s.basis for s in subs]...)
    index = Dict(s.name => i for (i, s) in enumerate(subs))
    ops = Dict(s.name => map(o -> embed(basis, index[s.name], o), s.ops) for s in subs)
    CompositeSystem(collect(subs), basis, index, ops)
end

"""
    getsubsystem(cs::CompositeSystem, name::Symbol)

Return the subsystem registered under `name`.
"""
getsubsystem(cs::CompositeSystem, name::Symbol) = cs.subsystems[cs.index[name]]

"""
    op(cs::CompositeSystem, name::Symbol, key::Symbol)

Look up a cached operator belonging to subsystem `name`, already embedded into
the composite Hilbert space (e.g. `op(cs, :cavity, :a)`).
"""
op(cs::CompositeSystem, name::Symbol, key::Symbol) = cs.ops[name][key]

"""
    embed(cs::CompositeSystem, name::Symbol, local_op::AbstractOperator)

Embed an arbitrary operator defined on subsystem `name`'s local basis into
the composite Hilbert space — the same embedding [`CompositeSystem`](@ref)
does internally for every cached subsystem op at construction time, exposed
here for operators that aren't one of those cached ops (e.g. a projector
built from an arbitrary local state, as needed for a conditional
measurement). For a cached op, prefer [`op`](@ref) (no repeated embedding
work); reach for this only when `local_op` isn't already one of them.

Adds a method to `QuantumOptics.jl`'s own `embed` (same generic function,
just dispatched on a `CompositeSystem` first argument) rather than
introducing a new name for the same operation.
"""
embed(cs::CompositeSystem, name::Symbol, local_op::AbstractOperator) = embed(cs.basis, cs.index[name], local_op)
