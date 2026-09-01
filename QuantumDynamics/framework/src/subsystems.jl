abstract type AbstractSubsystem end

"""
    Qubit(name, ω)

Two-level system with transition frequency `ω`. Caches its operators
(`σx, σy, σz, σp, σm, n, h0, id`) on a `SpinBasis(1//2)`. `n = (id + σz)/2` is
the excitation-number operator (0 for ground, 1 for excited); `h0 = ω*n` is
the bare local Hamiltonian, cached so that [`bare_hamiltonian`](@ref) can sum
`h0` uniformly across any subsystem type without recomputing it.

Note: QuantumOptics.jl's `sigmam`/`sigmap` treat spin-up (`σz = +1`) as the
higher-energy state (`sigmam` lowers spin-up → spin-down), so `n` must align
with that sign (`+σz`, not `-σz`) for `σm`/`σp` to act as proper
lowering/raising operators — otherwise interaction terms like
`ad*σm + a*σp` fail to conserve total excitation number.
"""
struct Qubit <: AbstractSubsystem
    name::Symbol
    ω::Float64
    basis::Basis
    ops::NamedTuple
end

function Qubit(name::Symbol, ω::Real)
    b = SpinBasis(1//2)
    n = (identityoperator(b) + sigmaz(b)) / 2
    ops = (
        σx=sigmax(b), σy=sigmay(b), σz=sigmaz(b),
        σp=sigmap(b), σm=sigmam(b),
        n=n, h0=float(ω) * n,
        id=identityoperator(b),
    )
    Qubit(name, float(ω), b, ops)
end

"""
    HarmonicOscillator(name, ω; nmax=10)

Bosonic mode with frequency `ω`, truncated to Fock states `0:nmax`. Caches its
operators (`a, ad, n, h0, id`) on a `FockBasis(nmax)`; `h0 = ω*n` is the bare
local Hamiltonian (see [`bare_hamiltonian`](@ref)).
"""
struct HarmonicOscillator <: AbstractSubsystem
    name::Symbol
    ω::Float64
    basis::Basis
    ops::NamedTuple
end

function HarmonicOscillator(name::Symbol, ω::Real; nmax::Int=10)
    b = FockBasis(nmax)
    n = number(b)
    ops = (a=destroy(b), ad=create(b), n=n, h0=float(ω) * n, id=identityoperator(b))
    HarmonicOscillator(name, float(ω), b, ops)
end

"""
    Transmon(name, ω, α; nmax=4)

Weakly anharmonic multi-level bosonic mode with transition frequency `ω` and
anharmonicity `α`, truncated to Fock states `0:nmax`. Caches its operators
(`a, ad, n, h0, id`) on a `FockBasis(nmax)`, the same operator set as
[`HarmonicOscillator`](@ref) plus `h0`.

The bare energy ladder is `E_k = k*ω + (α/2)*k*(k-1)`, cached as
`h0 = ω*n + (α/2)*n*(n - id)`. For a physical transmon `α` is **negative**
(the level spacing shrinks with `k`, e.g. `ω/2π ~ 5 GHz`,
`α/2π ~ -200 MHz`); `α = 0` recovers a plain harmonic oscillator. Default
`nmax=4` since anharmonicity pushes higher transitions increasingly
off-resonance, keeping only the first few levels relevant in practice —
override for problems that need more.
"""
struct Transmon <: AbstractSubsystem
    name::Symbol
    ω::Float64
    α::Float64
    basis::Basis
    ops::NamedTuple
end

function Transmon(name::Symbol, ω::Real, α::Real; nmax::Int=4)
    b = FockBasis(nmax)
    n = number(b)
    id = identityoperator(b)
    h0 = float(ω) * n + (float(α) / 2) * n * (n - id)
    ops = (a=destroy(b), ad=create(b), n=n, h0=h0, id=id)
    Transmon(name, float(ω), float(α), b, ops)
end

"""
    op(s::AbstractSubsystem, key::Symbol)

Look up a cached local operator (e.g. `op(qubit, :σz)`, `op(cavity, :a)`).
"""
op(s::AbstractSubsystem, key::Symbol) = s.ops[key]
