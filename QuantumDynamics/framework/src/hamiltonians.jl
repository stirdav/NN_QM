"""
    bare_hamiltonian(cs::CompositeSystem)

Sum of `h0` over every subsystem in `cs`, using each subsystem's cached,
embedded bare-energy operator. `h0` is `ω * n` for [`Qubit`](@ref) and
[`HarmonicOscillator`](@ref), and `ω*n + (α/2)*n*(n-1)` for [`Transmon`](@ref)
(its anharmonic correction) — every `AbstractSubsystem` caches `h0`, so this
is generic across subsystem types without needing to know which one it is.
"""
function bare_hamiltonian(cs::CompositeSystem)
    terms = [op(cs, s.name, :h0) for s in cs.subsystems]
    sum(terms)
end

# Internal dispatch: the "qubit-like" partner in a coupling Hamiltonian may be
# a two-level Qubit (σm/σp) or a multi-level Transmon (a/ad) — both satisfy
# [n, lower] = -lower, [n, raise] = +raise, which is all the excitation-number
# conservation arguments below rely on.
_lower(cs::CompositeSystem, name::Symbol) = _lower(cs, name, getsubsystem(cs, name))
_lower(cs::CompositeSystem, name::Symbol, ::Qubit) = op(cs, name, :σm)
_lower(cs::CompositeSystem, name::Symbol, ::Transmon) = op(cs, name, :a)
_raise(cs::CompositeSystem, name::Symbol) = _raise(cs, name, getsubsystem(cs, name))
_raise(cs::CompositeSystem, name::Symbol, ::Qubit) = op(cs, name, :σp)
_raise(cs::CompositeSystem, name::Symbol, ::Transmon) = op(cs, name, :ad)

"""
    jaynes_cummings(cs, qubit_name, osc_name, g)

Rotating-wave-approximation coupling between an oscillator and a qubit-like
subsystem ([`Qubit`](@ref) or [`Transmon`](@ref)) named `qubit_name`, on top
of the bare energies: `H = h0_q + h0_o + g*(ad*lower + a*raise)`, where
`lower`/`raise` are `σm`/`σp` for a `Qubit` or `a`/`ad` for a `Transmon`.
Conserves total excitation number `nq + no`.
"""
function jaynes_cummings(cs::CompositeSystem, qubit_name::Symbol, osc_name::Symbol, g::Real)
    a, ad = op(cs, osc_name, :a), op(cs, osc_name, :ad)
    lower, raise = _lower(cs, qubit_name), _raise(cs, qubit_name)
    bare_hamiltonian(cs) + g * (ad * lower + a * raise)
end

"""
    rabi(cs, qubit_name, osc_name, g)

Quantum Rabi model: the same bare energies as [`jaynes_cummings`](@ref) but
with the full (non-rotating-wave) coupling `g*(a+ad)*(lower+raise)`, which
does *not* conserve total excitation number and is needed outside the strong
detuning / weak-coupling regime where the RWA breaks down. `qubit_name` may
be a [`Qubit`](@ref) or [`Transmon`](@ref), as in [`jaynes_cummings`](@ref).
"""
function rabi(cs::CompositeSystem, qubit_name::Symbol, osc_name::Symbol, g::Real)
    a, ad = op(cs, osc_name, :a), op(cs, osc_name, :ad)
    lower, raise = _lower(cs, qubit_name), _raise(cs, qubit_name)
    bare_hamiltonian(cs) + g * (a + ad) * (lower + raise)
end

"""
    tavis_cummings(cs, qubit_names, osc_name, g)

Multi-qubit generalization of [`jaynes_cummings`](@ref): every subsystem in
`qubit_names` (each a [`Qubit`](@ref) or [`Transmon`](@ref), independently)
couples to the same oscillator mode with strength `g` (a scalar applied to
all, or a vector/tuple matching `qubit_names` for individual couplings).
`H = Σᵢ h0ᵢ + h0_o + Σᵢ gᵢ*(ad*lowerᵢ + a*raiseᵢ)`.

When `g` isn't a scalar, `length(g)` must equal `length(qubit_names)` (checked,
throws `ArgumentError` otherwise): zipping mismatched-length iterables would
otherwise silently truncate to the shorter one, dropping trailing qubits (or
trailing `g`s) from the coupling with no error.
"""
function tavis_cummings(cs::CompositeSystem, qubit_names, osc_name::Symbol, g)
    gs = g isa Real ? Iterators.repeated(g) : g
    if !(g isa Real) && length(qubit_names) != length(gs)
        throw(ArgumentError(
            "tavis_cummings: length(qubit_names)=$(length(qubit_names)) must equal length(g)=$(length(gs))",
        ))
    end
    a, ad = op(cs, osc_name, :a), op(cs, osc_name, :ad)
    H = bare_hamiltonian(cs)
    for (name, gi) in zip(qubit_names, gs)
        lower, raise = _lower(cs, name), _raise(cs, name)
        H += gi * (ad * lower + a * raise)
    end
    H
end

"""
    dispersive_hamiltonian(cs, qubit_name, osc_name, g)

Second-order dispersive approximation to [`jaynes_cummings`](@ref), valid
when the qubit-oscillator detuning `Δ = ωq - ωc` is large compared to `g`.
The coupling is eliminated in favor of a qubit-state-dependent frequency
shift `χ = g²/Δ` on the oscillator: `H = ωq*nq + ωc*no + χ*σz*no`.

Qubit-only (uses `σz` directly): a transmon's dispersive shift depends on its
anharmonicity too and needs a different formula, not yet implemented here.

Throws `ArgumentError` at `Δ=0`: the dispersive approximation is only valid
far from resonance, so `Δ=0` isn't a degenerate-but-still-meaningful corner
of its own domain — it's the resonant Jaynes-Cummings regime this
approximation doesn't apply to at all. Left unguarded, `χ = g²/Δ` would
silently blow up to `Inf` (or `NaN` if `g=0` too) instead of signaling that
the approximation itself doesn't apply here.
"""
function dispersive_hamiltonian(cs::CompositeSystem, qubit_name::Symbol, osc_name::Symbol, g::Real)
    Δ = getsubsystem(cs, qubit_name).ω - getsubsystem(cs, osc_name).ω
    Δ != 0 || throw(ArgumentError(
        "dispersive_hamiltonian: qubit `$qubit_name` and oscillator `$osc_name` are on resonance (Δ=0) — the dispersive approximation is invalid there (that's the resonant Jaynes-Cummings regime); use jaynes_cummings instead",
    ))
    χ = g^2 / Δ
    σz = op(cs, qubit_name, :σz)
    no = op(cs, osc_name, :n)
    bare_hamiltonian(cs) + χ * σz * no
end

"""
    quadratic_coupling(cs, qubit_name, osc_name, g)

Quadratic (optomechanical-type) coupling between an oscillator and a qubit:
`H = h0_q + h0_o + g*(a+ad)^2*σz`. Unlike [`jaynes_cummings`](@ref)/
[`rabi`](@ref)/[`tavis_cummings`](@ref), the coupling is not "qubit-like"
ladder coupling (`σm`/`σp` or `a`/`ad` raising/lowering) but a direct `σz`
conditioning of the oscillator's squeezing — physically relevant for
qubit-oscillator systems with strong quadratic coupling (e.g.
superconducting qubits coupled to mechanical oscillators). Does not
conserve total excitation number (verified in tests as a sanity control,
the same role `rabi`'s non-conservation plays against the JC conservation
test).

Qubit-only (uses `σz` directly, like [`dispersive_hamiltonian`](@ref)): does
not generalize to `Transmon` the way the ladder-coupling recipes do, since
there is no natural `σz` analogue for a multi-level Transmon.
"""
function quadratic_coupling(cs::CompositeSystem, qubit_name::Symbol, osc_name::Symbol, g::Real)
    a, ad = op(cs, osc_name, :a), op(cs, osc_name, :ad)
    σz = op(cs, qubit_name, :σz)
    bare_hamiltonian(cs) + g * (a + ad)^2 * σz
end

"""
    add_time_dependence(H::AbstractOperator, terms::Pair...; init_time=0.0)

Turn a static Hamiltonian `H` (built by any recipe above, or hand-assembled
from cached ops) into a `TimeDependentSum` by adding one or more `coeff =>
operator` time-dependent terms on top of it — sugar for the pattern
documented in "Time-dependent Hamiltonians" in design.md:

```julia
TimeDependentSum(1.0 => H, coeff1 => op1, coeff2 => op2, ...; init_time)
```

Deliberately *not* called `add_drive`: a time-dependent term added this way
isn't necessarily an external drive — it may just as well be a term whose
time dependence is an artifact of a rotating/interaction-frame
transformation (e.g. a fast counter-rotating term left over after moving to
an interaction frame), with no classical field behind it at all. This
function only owns the "turn a term's coefficient into a function of `t`"
mechanics; what that time dependence physically represents is entirely the
caller's business, same as `projector` in [`condition_on`](@ref) or the
coupling constant in any Hamiltonian recipe above.

`H` keeps a fixed coefficient of `1.0`; each `terms` pair's `coeff` should be
a function of `t` (or a constant, matching `TimeDependentSum`'s own
convention) and its operator should be one of the framework's cached,
embedded operators (or a fixed sum/product of them) — never rebuilt inside
the coefficient closure, per the same "coefficients carry the time
dependence, operators stay fixed" rule as any other `TimeDependentSum`.

```julia
Ht = add_time_dependence(quadratic_coupling(cs, :qubit, :osc, g),
                          (t -> A*cos(ωd*t)) => op(cs, :qubit, :σx))
```
"""
add_time_dependence(H::AbstractOperator, terms::Pair{<:Any,<:AbstractOperator}...; init_time=0.0) =
    TimeDependentSum(1.0 => H, terms...; init_time=init_time)
