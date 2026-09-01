"""
    ptrace(cs::CompositeSystem, rho::AbstractOperator, name::Symbol)

Partial trace of `rho` (a density operator on `cs.basis`) over subsystem
`name`, looked up via `cs.index` rather than requiring the caller to know
Hilbert-space index order. Adds a method to `QuantumOptics.jl`'s own
`ptrace`.
"""
ptrace(cs::CompositeSystem, rho::AbstractOperator, name::Symbol) = ptrace(rho, cs.index[name])

"""
    condition_on(cs::CompositeSystem, state, name::Symbol, projector::AbstractOperator)

Project `state` (a `Ket` or density operator on `cs.basis`; a `Ket` is
converted to `dm(state)` first) onto `projector` on subsystem `name`, trace
`name` out, and renormalize what's left — the "measure subsystem `name`,
keep the rest of the system conditioned on the outcome" pattern common to
conditional state preparation and readout protocols (e.g. heralding an
oscillator state on a qubit measurement outcome). Returns `(ρ_cond, prob)`:
the renormalized conditioned state, and the measurement probability for
`projector`'s outcome (`tr` of the unnormalized `P*ρ*P'`, before
renormalization) — callers that need to check for a near-zero-probability
outcome (renormalizing that would blow up to `NaN`/`Inf`) can do so from
`prob` without a second, duplicate computation.

`projector` must be a *local* operator on `getsubsystem(cs, name).basis`
(typically `projector(ψ)` for measuring in local state `ψ`, or a sum of such
projectors for a coarser-grained outcome) — it is embedded into `cs.basis`
via [`embed`](@ref) before being applied.

```julia
qb = getsubsystem(cs, :qubit).basis
plus = normalize(spinup(qb) + spindown(qb))
ρ_cond, prob = condition_on(cs, ψt, :qubit, projector(plus))
```
"""
function condition_on(cs::CompositeSystem, state, name::Symbol, projector::AbstractOperator)
    ρ = state isa Ket ? dm(state) : state
    P = embed(cs, name, projector)
    ρ_unnorm = P * ρ * P'
    prob = real(tr(ρ_unnorm))
    ρ_cond = ptrace(cs, ρ_unnorm, name)
    ρ_cond / prob, prob
end

"""
    FockCutoffTooSmall <: Exception

Thrown by [`check_fock_cutoff`](@ref) when population near a Fock
subsystem's truncation boundary exceeds its tolerance. A distinct type from
[`check_fock_cutoff`](@ref)'s `ArgumentError`s (caller misuse: wrong basis
type, or a `levels` that doesn't fit the given `nmax`) so that "the cutoff
itself is too small" — the one case callers may want to catch and react to,
e.g. future parameter-sweep infrastructure retrying with a larger `nmax` —
can be caught specifically instead of matched against an error string.
"""
struct FockCutoffTooSmall <: Exception
    name::Symbol
    nmax::Int
    levels::Int
    population::Float64
    tol::Float64
end

function Base.showerror(io::IO, e::FockCutoffTooSmall)
    print(io,
        "FockCutoffTooSmall: subsystem `$(e.name)` has $(round(e.population, sigdigits=3)) population in its top $(e.levels) Fock level(s) (tol=$(e.tol)) — nmax=$(e.nmax) is too small for convergence; increase it.",
    )
end

"""
    check_fock_cutoff(cs::CompositeSystem, name::Symbol, states; tol=1e-5, levels=3)

Guard against a Fock-space cutoff too small for subsystem `name`'s dynamics
to be converged. Sums population in the top `levels` Fock level(s) of
`name`'s basis for each state in `states` (a single `Ket`, or a collection
of them, e.g. a trajectory returned by [`evolve`](@ref)) and throws
[`FockCutoffTooSmall`](@ref) the first time that population exceeds `tol`.

An under-sized cutoff leaks probability into levels right at the truncation
boundary well before that shows up as an obviously wrong answer elsewhere
(e.g. a fidelity that undershoots its converged value by a few parts in a
thousand, with no error) — this checks the truncation directly rather than
relying on a downstream symptom being caught by eye. `name` must be on a
`FockBasis` (a [`HarmonicOscillator`](@ref) or [`Transmon`](@ref)).

Deliberately `Ket`-only, not density operators: this is meant as a
closed-system pre-flight check (e.g. run once on a `schroedinger`
trajectory before adding dissipation), not as a general convergence
diagnostic for open-system evolution. Under a bath (`Decay`/`Gain`/
`thermal_bath`), real steady-state population sitting near the boundary of
an otherwise-adequate cutoff is expected physics, not a truncation
artifact — this function has no way to tell the two apart, so rather than
risk that false positive (or, worse, a `tol` loosened to work around it
that then also hides genuine truncation) it simply doesn't accept density
operators.

`levels` defaults to 3, not 1, for one structural reason and one
otherwise-arbitrary margin — keep those separate:
- *Structural:* a single top level is provably blind to truncation whenever
  the dynamics are parity-restricted and `nmax` has the "wrong" parity for
  it — e.g. a coupling built from `a²`/`ad²` (as in `Hcs`/`H1` in
  `examples/conditionally_squeezed_states/`) only connects Fock levels two
  apart, so starting from an even level leaves *every odd* level, including
  possibly the single top level of an odd `nmax`, at exactly zero
  population regardless of how truncated the simulation actually is. 2
  consecutive levels covers both parities and closes that specific blind
  spot for any Δn=2 coupling.
- *Arbitrary:* the default is 3, one level above that 2-level floor, purely
  as a small margin against silently failing to catch a marginal cutoff —
  it is *not* derived from analyzing any particular Hamiltonian's leakage
  pattern, and shouldn't be read as such. For a coupling that leaks
  differently (e.g. connects levels more than 2 apart, so 2 consecutive
  levels doesn't even close the parity blind spot), the default isn't
  guaranteed to be enough — knowing whether your Hamiltonian needs a larger
  `levels` (or a different `tol`) is the caller's responsibility, not
  something this default can determine for you.

`levels` must leave at least half of `name`'s basis unchecked (`levels <=
(nmax + 1) ÷ 2`) — otherwise "population near the boundary" stops meaning
anything: on a small enough basis, the top `levels` region *is* most or all
of the Hilbert space, so its population is always ≈1 regardless of whether
`nmax` is actually too small, and the check would fail on every call
regardless of convergence. This makes the default unusable as-is on a
small basis (e.g. a default `Transmon`, `nmax=4`) — call with an explicit,
smaller `levels` there rather than relying on the default.

```julia
_, states = evolve(tspan, ψ0, H)
check_fock_cutoff(cs, :osc, states)   # throws if nmax is too small anywhere along the trajectory
```
"""
function check_fock_cutoff(cs::CompositeSystem, name::Symbol, states; tol::Real=1e-5, levels::Int=3)
    basis = getsubsystem(cs, name).basis
    basis isa FockBasis || throw(ArgumentError("check_fock_cutoff: subsystem `$name` is not on a FockBasis"))
    nmax = basis.N
    half = (nmax + 1) ÷ 2
    1 <= levels <= half || throw(ArgumentError(
        "check_fock_cutoff: levels ($levels) must leave at least half of `$name`'s basis unchecked (nmax=$nmax, so levels must be between 1 and $half) — pass a smaller `levels` explicitly for this basis size",
    ))
    states isa AbstractOperator && throw(ArgumentError(
        "check_fock_cutoff: only accepts `Ket` states (got $(typeof(states))) — this is a closed-system pre-flight check, not a general open-system convergence diagnostic; see the docstring.",
    ))
    P_top = embed(cs, name, sum(projector(fockstate(basis, n)) for n in (nmax - levels + 1):nmax))
    for state in (states isa Ket ? (states,) : states)
        state isa Ket || throw(ArgumentError(
            "check_fock_cutoff: only accepts `Ket` states (got $(typeof(state))) — this is a closed-system pre-flight check, not a general open-system convergence diagnostic; see the docstring.",
        ))
        pop = real(expect(P_top, state))
        pop <= tol || throw(FockCutoffTooSmall(name, nmax, levels, pop, tol))
    end
    nothing
end
