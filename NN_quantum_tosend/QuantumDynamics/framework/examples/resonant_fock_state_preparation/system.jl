using QuantumDynamics, QuantumOptics

# Interaction frame co-rotating with the resonator at ωr: both subsystems are
# built with ω=0, same trick as `fock_state_preparation`. Unlike that
# example, the qubit is *never* detuned away from the oscillator — Δ=0
# throughout, so `jaynes_cummings` below is a genuinely time-independent
# operator, not something toggled on/off via a Δ(t) plateau.
function build_system(; nmax::Int, g::Real=1.0)
    qubit = Qubit(:qubit, 0.0)
    osc = HarmonicOscillator(:osc, 0.0; nmax=nmax)
    CompositeSystem(qubit, osc)
end

# Smooth top-hat (tanh edges of width τ), 0 -> 1 -> 0 over [t0,t1]. Its
# integral over all time is exactly t1-t0, independent of τ — see README.
smoothstep(t, t0, τ) = 0.5 * (1 + tanh((t - t0) / τ))
tophat(t, t0, t1, τ) = smoothstep(t, t0, τ) - smoothstep(t, t1, τ)

# One step (n -> n+1): a drive plateau (Ω(t)=Ω0, an n-independent π-pulse on
# the qubit) followed by a swap window where the drive is off and the
# always-on coupling performs the vacuum-Rabi exchange |e,n⟩ -> |g,n+1⟩.
struct Step
    n::Int
    a::Float64   # drive plateau start
    b::Float64   # drive plateau end
    c::Float64   # swap window start
    d::Float64   # swap window end
end

swap_time(n, g) = π / (2g * sqrt(n + 1))

function build_schedule(N::Int; g, w, τ)
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

# Ω(t): sum of tanh top-hats, one per drive plateau. The drive term is the
# bare H = Ω(t)*σx (no carrier), so the Bloch-sphere rotation angle is
# 2*∫Ω dt — a full π flip therefore needs ∫Ω dt = π/2, i.e. Ω0 = π/(2w).
function envelope_fn(steps, w, τ)
    Ω0 = π / (2w)
    t -> Ω0 * sum(tophat(t, s.a, s.b, τ) for s in steps)
end

# Build a fresh N-photon-margin system, evolve |g,0⟩ under the resonant
# drive/swap protocol targeting N (N >= 1) using `hamiltonian_fn`
# (`fock_ladder_hamiltonian_resonant`), and return everything a caller needs
# to inspect the run.
#
# `hamiltonian_fn` also returns `dtmax`, the tanh edge width `τ` of its
# narrowest pulse — the drive plateau (width `w`) is orders of magnitude
# shorter than the swap windows around it, so a long run's adaptive step
# size grows large during the slow swaps and can silently step over the next
# drive plateau entirely (no error raised — the solver just never evaluates
# H inside it). Passing `dtmax=τ` to `evolve` bounds the step size well
# below every plateau's width, at negligible cost since the swap segments
# don't need small steps for accuracy anyway.
function run_fock_prep(N, hamiltonian_fn; nmax_margin=4, n_out=2000 * N, kwargs...)
    nmax = N + nmax_margin
    cs = build_system(nmax=nmax)
    qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
    ψ0 = spindown(qb) ⊗ fockstate(ob, 0)
    H, steps, tfinal, Ωfun, dtmax = hamiltonian_fn(cs, N; kwargs...)
    tspan = 0:tfinal/n_out:tfinal
    tout, states = evolve(tspan, ψ0, H; dtmax=dtmax)
    (; cs, tout, states, steps, tfinal, Ωfun)
end
