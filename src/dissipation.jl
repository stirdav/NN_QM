abstract type Dissipator end

# Internal dispatch: the jump operator for a given channel depends on the
# subsystem's type, mirroring the `_lower`/`_raise` dispatch in hamiltonians.jl.
# Kept separate from `_lower`/`_raise` (rather than reusing/extending them)
# because those are documented as specifically "qubit-like" (Qubit/Transmon);
# decay is physically meaningful for a HarmonicOscillator too.
_decay_op(cs::CompositeSystem, name::Symbol) = _decay_op(cs, name, getsubsystem(cs, name))
_decay_op(cs::CompositeSystem, name::Symbol, ::Qubit) = op(cs, name, :σm)
_decay_op(cs::CompositeSystem, name::Symbol, ::Union{HarmonicOscillator,Transmon}) = op(cs, name, :a)

# The "raise" counterpart to _decay_op, used by Gain (and by Decay's nth
# scaling being purely a rate factor on the same _decay_op above — Gain is
# the channel that actually needs the raising operator). Kept separate from
# _raise in hamiltonians.jl for the same reason _decay_op is: gain is
# physically meaningful for a bare HarmonicOscillator too (thermal photon
# absorption), not just qubit-like subsystems.
_gain_op(cs::CompositeSystem, name::Symbol) = _gain_op(cs, name, getsubsystem(cs, name))
_gain_op(cs::CompositeSystem, name::Symbol, ::Qubit) = op(cs, name, :σp)
_gain_op(cs::CompositeSystem, name::Symbol, ::Union{HarmonicOscillator,Transmon}) = op(cs, name, :ad)

_dephasing_op(cs::CompositeSystem, name::Symbol) = _dephasing_op(cs, name, getsubsystem(cs, name))
_dephasing_op(cs::CompositeSystem, name::Symbol, ::Qubit) = op(cs, name, :σz)
_dephasing_op(cs::CompositeSystem, name::Symbol, ::Union{HarmonicOscillator,Transmon}) = op(cs, name, :n)

"""
    Decay(name, rate; nth=0.0)

Amplitude-damping channel on subsystem `name` at rate `rate` (e.g. cavity
photon loss `κ`, qubit relaxation `γ1`). Jump operator
`sqrt(rate*(nth+1)) * lower`, where `lower` is `σm` for a [`Qubit`](@ref) or
`a` for a [`HarmonicOscillator`](@ref)/[`Transmon`](@ref). `rate` and `nth`
must be non-negative (checked at construction, not deferred to a `sqrt`
domain error inside [`jump_operator`](@ref)).

`nth` defaults to `0.0` (a zero-temperature bath), reproducing exactly the
original `sqrt(rate) * lower` behavior — the common case, and the only one
this type supported before thermal baths were added. For `nth>0`, combine
`Decay` with [`Gain`](@ref) on the same subsystem, passing the same `rate`
and `nth`, to build a full finite-temperature bath:

```julia
J = jump_operators(cs, [Decay(:osc, γm; nth=nm_th), Gain(:osc, γm; nth=nm_th)])
```

which reproduces the standard thermal Lindblad terms
`γm*(nth+1)*D[b] + γm*nth*D[b†]`. `Decay` alone (default `nth=0`) only ever
models the relaxation half of that pair — `Gain` is a separate dissipator
rather than something `Decay` produces internally, so each channel stays
single-purpose: `Decay`'s jump operator is always the "loss" one, and its
behavior at `nth=0` never changes based on a keyword the caller might not
have known to pass.
"""
struct Decay <: Dissipator
    name::Symbol
    rate::Float64
    nth::Float64
    function Decay(name::Symbol, rate::Real; nth::Real=0.0)
        rate >= 0 || throw(ArgumentError("Decay rate must be non-negative, got $rate"))
        nth >= 0 || throw(ArgumentError("Decay nth must be non-negative, got $nth"))
        new(name, float(rate), float(nth))
    end
end

"""
    Gain(name, rate; nth)

Amplitude-gain (thermal excitation) channel on subsystem `name` — the
counterpart to [`Decay`](@ref). Jump operator `sqrt(rate*nth) * raise`,
where `raise` is `σp` for a [`Qubit`](@ref) or `ad` for a
[`HarmonicOscillator`](@ref)/[`Transmon`](@ref).

`rate` and `nth` must both be non-negative (checked at construction). `nth`
has no default (unlike `Decay`'s `nth=0.0`) since, unlike `Decay`, a `Gain`
channel is only ever constructed to pair with a `Decay` for a specific
finite-temperature bath — there is no meaningful "just gain, no `nth`
in mind" default to fall back on. `nth=0` is allowed (symmetric with
`Decay`) and simply produces a zero jump operator, so temperature sweeps
that range down to `nth=0` don't need to special-case omitting `Gain`.

Combine with [`Decay`](@ref) on the same subsystem, passing the same `rate`
and `nth` (or use [`thermal_bath`](@ref) to build both together), to build a
finite-temperature bath — see `Decay`'s docstring for the full pattern.
"""
struct Gain <: Dissipator
    name::Symbol
    rate::Float64
    nth::Float64
    function Gain(name::Symbol, rate::Real; nth::Real)
        rate >= 0 || throw(ArgumentError("Gain rate must be non-negative, got $rate"))
        nth >= 0 || throw(ArgumentError("Gain nth must be non-negative, got $nth"))
        new(name, float(rate), float(nth))
    end
end

"""
    thermal_bath(name, rate, nth)

Build the matched `[Decay, Gain]` pair for a finite-temperature bath on
subsystem `name`, coupling rate `rate` and thermal occupation `nth` —
equivalent to `[Decay(name, rate; nth), Gain(name, rate; nth)]` but with
`rate`/`nth` guaranteed to agree between the two (constructing them
separately risks a typo desyncing them, silently producing the wrong
steady-state occupation instead of an error). `nth=0` is a valid,
zero-temperature bath (Gain's jump operator is simply zero), so this is
safe to use directly inside a temperature sweep without special-casing
`nth=0`.

```julia
J = jump_operators(cs, thermal_bath(:osc, γm, nm_th))
```
"""
thermal_bath(name::Symbol, rate::Real, nth::Real) = [Decay(name, rate; nth), Gain(name, rate; nth)]

"""
    Dephasing(name, rate)

Pure-dephasing channel on subsystem `name` at rate `rate` (i.e. `rate =
1/Tφ`). Jump operator `sqrt(rate/2) * σz` for a [`Qubit`](@ref), or
`sqrt(rate/2) * n` for a [`HarmonicOscillator`](@ref)/[`Transmon`](@ref)
(photon-number-dependent dephasing).

**Sign/factor convention (load-bearing detail):** for the `Qubit` case, the
`/2` is chosen so `rate` is *directly* the coherence decay rate, matching how
`Decay`'s `rate` is directly the population decay rate (`sqrt(κ)*a` gives
`d⟨n⟩/dt = -κ⟨n⟩`). Concretely, `D[c]ρ = cρc† - {c†c,ρ}/2` with `c =
sqrt(rate/2)*σz` (so `c†c = (rate/2)*I`) gives off-diagonal decay
`dρ₀₁/dt = -rate*ρ₀₁`. Dropping the `/2` (i.e. `c = sqrt(rate)*σz`) would
instead decay coherence at `2*rate` — silently halving the effective Tφ for
anyone plugging in a literature `1/Tφ` value. Verified in `test/runtests.jl`
("master-equation dissipation matches known solutions"). `rate` must be
non-negative (checked at construction).
"""
struct Dephasing <: Dissipator
    name::Symbol
    rate::Float64
    function Dephasing(name::Symbol, rate::Real)
        rate >= 0 || throw(ArgumentError("Dephasing rate must be non-negative, got $rate"))
        new(name, float(rate))
    end
end

"""
    jump_operator(cs::CompositeSystem, d::Dissipator)

Build the single embedded, rate-scaled jump operator for dissipator `d`
(`sqrt(d.rate*(d.nth+1)) * <channel operator>` for [`Decay`](@ref);
`sqrt(d.rate*d.nth) * <channel operator>` for [`Gain`](@ref);
`sqrt(d.rate/2) * <channel operator>` for [`Dephasing`](@ref) — see their
docstrings for why). Named to match `QuantumOptics.jl`'s own vocabulary: its
`master`/`master_dynamic` take these as the `J` argument, documented there as
"jump operators".
"""
jump_operator(cs::CompositeSystem, d::Decay) = sqrt(d.rate * (d.nth + 1)) * _decay_op(cs, d.name)
jump_operator(cs::CompositeSystem, d::Gain) = sqrt(d.rate * d.nth) * _gain_op(cs, d.name)
jump_operator(cs::CompositeSystem, d::Dephasing) = sqrt(d.rate / 2) * _dephasing_op(cs, d.name)

"""
    jump_operators(cs::CompositeSystem, dissipators)

Build the list of embedded, rate-scaled jump operators for an iterable of
[`Dissipator`](@ref)s, e.g.

```julia
J = jump_operators(cs, [Decay(:cavity, κ), Decay(:qubit, γ1), Dephasing(:qubit, γφ)])
```

Pass the result directly as the `J` argument to [`evolve`](@ref).
"""
jump_operators(cs::CompositeSystem, dissipators) = [jump_operator(cs, d) for d in dissipators]
