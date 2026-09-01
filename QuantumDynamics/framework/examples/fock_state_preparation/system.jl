using QuantumDynamics, QuantumOptics

# Interaction frame co-rotating with the resonator at ωr: both subsystems are
# built with ω=0, so `jaynes_cummings` returns the coupling term alone (no
# bare energies) — the qubit's bare energy is folded into the time-dependent
# detuning Δ(t) added on top by each addressing scheme in
# `selective_pulses.jl`/`fixed_frequency.jl`.
function build_system(; nmax::Int, g::Real=1.0)
    qubit = Qubit(:qubit, 0.0)
    osc = HarmonicOscillator(:osc, 0.0; nmax=nmax)
    CompositeSystem(qubit, osc)
end

# Smooth top-hat (tanh edges of width τ), 1 on the interior of [t0,t1], 0 outside.
smoothstep(t, t0, τ) = 0.5 * (1 + tanh((t - t0) / τ))
tophat(t, t0, t1, τ) = smoothstep(t, t0, τ) - smoothstep(t, t1, τ)

# One step (n -> n+1) of the protocol: a dispersive plateau (Δ=Δbig, a
# π-pulse addressed some way — see `selective_pulses.jl`/`fixed_frequency.jl`)
# followed by a resonant plateau (Δ=0, vacuum-Rabi swap |e,n⟩->|g,n+1⟩).
struct Step
    n::Int
    a::Float64      # dispersive plateau start
    b::Float64      # dispersive plateau end
    tc::Float64     # pulse center
    c::Float64      # swap plateau start
    d::Float64      # swap plateau end
end

swap_time(n, g) = π / (2g * sqrt(n + 1))

function build_schedule(N::Int; g, σt, τramp)
    plateau_width = 8σt
    buffer = 4τramp
    steps = Step[]
    t = 0.0
    for n in 0:N-1
        a = t
        b = a + plateau_width
        tc = (a + b) / 2
        c = b + buffer
        d = c + swap_time(n, g)
        push!(steps, Step(n, a, b, tc, c, d))
        t = d + buffer
    end
    steps, t
end

# Δ(t): sum of top-hats, one per dispersive plateau — shared by both
# addressing schemes, which differ only in the drive coefficient.
function detuning_fn(steps, Δbig, τramp)
    t -> Δbig * sum(tophat(t, s.a, s.b, τramp) for s in steps)
end

# Drive envelope |Ω(t)|: sum of Gaussian pulse envelopes, one per step — the
# amplitude modulating each scheme's carrier, without the fast cos(ωd*t)
# term (which oscillates far too quickly to plot meaningfully against the
# plateau timescale). Shared by both schemes; only the carrier frequency ωd
# differs between them (see `drive_fn_selective`/`drive_fn_fixed`).
function envelope_fn(steps, σt, Ω0)
    t -> sum(Ω0 * exp(-(t - s.tc)^2 / (2σt^2)) for s in steps)
end

# Build a fresh N-photon-margin system, evolve |g,0⟩ under the Fock-ladder
# protocol targeting N (N >= 1) using `hamiltonian_fn` (either
# `fock_ladder_hamiltonian_selective` or `fock_ladder_hamiltonian_fixed`),
# and return everything a caller needs to inspect the run: the composite
# system, the time/state trajectory, the pulse schedule, and the Δ(t)/drive
# coefficient functions (for plotting the drive schedule alongside the
# populations — see `population_ladder.jl`). `nmax_margin` keeps the
# oscillator's Fock cutoff comfortably above the target (see the
# `check_fock_cutoff` call in the example scripts).
function run_fock_prep(N, hamiltonian_fn; nmax_margin=4, n_out=2000 * N, kwargs...)
    nmax = N + nmax_margin
    cs = build_system(nmax=nmax)
    qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
    ψ0 = spindown(qb) ⊗ fockstate(ob, 0)
    H, steps, tfinal, Δfun, Ωenv = hamiltonian_fn(cs, N; kwargs...)
    tspan = 0:tfinal/n_out:tfinal
    tout, states = evolve(tspan, ψ0, H)
    (; cs, tout, states, steps, tfinal, Δfun, Ωenv)
end
