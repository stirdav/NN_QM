using QuantumDynamics, QuantumOptics

# Interaction frame co-rotating with the oscillator. Both bare frequencies
# vanish, leaving the resonant Jaynes-Cummings coupling and the qubit drive.
function build_system(; nmax::Int)
    qubit = Qubit(:qubit, 0.0)
    osc = HarmonicOscillator(:osc, 0.0; nmax=nmax)
    CompositeSystem(qubit, osc)
end

# Smooth approximation to a unit step centered at `t0`. A finite τ avoids a
# discontinuous Hamiltonian, which is friendlier to the adaptive ODE solver.
smoothstep(t, t0, τ) = 0.5 * (1 + tanh((t - t0) / τ))

# Difference of two smooth steps: approximately one on [t0,t1] and zero
# outside. Its total area is exactly t1-t0, independent of the edge width τ.
tophat(t, t0, t1, τ) = smoothstep(t, t0, τ) - smoothstep(t, t1, τ)

# Timing information for one ladder step n→n+1. [a,b] is the qubit-drive
# plateau; [c,d] is the following drive-free resonant swap window.
struct Step
    n::Int
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

# Under the JC matrix element g*sqrt(n+1), this duration transfers all
# population from |e,n⟩ to |g,n+1⟩ in the ideal two-state manifold.
swap_time(n, g) = π / (2g * sqrt(n + 1))

function build_schedule(N::Int; g, w, τ)
    # Four edge widths let the tanh drive tail become negligible before a swap
    # begins and before the next nominally drive-only interval begins.
    buffer = 4τ
    steps = Step[]
    t = 0.0
    for n in 0:N-1
        a = t
        b = a + w
        c = b + buffer
        d = c + swap_time(n, g)
        push!(steps, Step(n, a, b, c, d))
        t = d + buffer
    end
    steps, t
end

function envelope_fn(steps, w, τ)
    # Hdrive=Ω(t)σx rotates the Bloch vector by 2∫Ωdt. Hence the area of
    # every qubit-flip pulse must be π/2, giving Ω0=π/(2w).
    Ω0 = π / (2w)
    t -> Ω0 * sum(tophat(t, s.a, s.b, τ) for s in steps)
end

# Master-equation run with independent relaxation and pure-dephasing rates:
#
#   qubit:      sqrt(γ1)*σm, sqrt(γ2/2)*σz
#   oscillator: sqrt(κ1)*a,  sqrt(κ2/2)*n
#
# All rates use the same frequency units as g. Here γ2 is directly the
# qubit pure-coherence decay rate. Bosonic number dephasing is level-separation
# dependent: |m⟩⟨n| decays at κ2*(m-n)^2/4 from that channel.
function run_open_fock_prep(N, hamiltonian_fn;
                            γ1=0.0, γ2=0.0, κ1=0.0, κ2=0.0,
                            nmax_margin=4, n_out=2000 * N, kwargs...)
    cs = build_system(nmax=N + nmax_margin)
    qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
    ρ0 = dm(spindown(qb) ⊗ fockstate(ob, 0))
    H, steps, tfinal, Ωfun, dtmax = hamiltonian_fn(cs, N; kwargs...)
    dissipators = [
        Decay(:qubit, γ1),
        Dephasing(:qubit, γ2),
        Decay(:osc, κ1),
        Dephasing(:osc, κ2),
    ]
    # jump_operators embeds the local σm, σz, a, and n operators into the
    # full qubit⊗oscillator Hilbert space and applies the square-root rates.
    J = jump_operators(cs, dissipators)

    # Dense output is requested for plotting, while dtmax=τ (returned by the
    # Hamiltonian builder) prevents the adaptive integrator from stepping over
    # a narrow drive plateau after taking large steps during a long swap.
    tspan = 0:tfinal/n_out:tfinal
    tout, states = evolve(tspan, ρ0, H, J; dtmax=dtmax)
    (; cs, tout, states, steps, tfinal, Ωfun, dissipators)
end
