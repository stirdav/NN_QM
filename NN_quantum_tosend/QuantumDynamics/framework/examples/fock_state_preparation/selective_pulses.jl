# Number-resolved addressing: each step's π-pulse is long and narrowband,
# individually tuned to that step's number-split transition frequency, so it
# only drives the |g,n⟩->|e,n⟩ transition and leaves other n-manifolds alone.
# This needs a deep dispersive regime (χ resolvable against the pulse
# bandwidth) but, once achieved, gives a protocol that would still work even
# if pulses fired "out of order" or targeted the wrong manifold by mistake —
# selectivity is a property of the pulse itself, not just of the sequence.

# Drive coefficient: sum of Gaussian-enveloped selective π-pulses, one per
# step, each resonant with the n-th number-split transition. Unlike
# `dispersive_hamiltonian`'s χ*σz*no (a cross term only, meant to sit on top
# of an already-measured bare ωq), here Δbig*nq is the actual bare splitting
# entering the exact Hamiltonian, so the resonance condition must come from
# exact 2x2 block diagonalization of {|e,n⟩,|g,n+1⟩} pairs (coupling
# g√(n+1)) rather than that formula: driving |g,n⟩ (dressed into block n-1,
# energy ≈ -nχ) up to |e,n⟩ (dressed into block n, energy ≈ Δbig+(n+1)χ)
# takes ωd = Δbig + (2n+1)χ.
function drive_fn_selective(steps, Δbig, χ, σt, Ω0)
    t -> sum(steps) do s
        ωd = Δbig + (2 * s.n + 1) * χ
        Ω0 * exp(-(t - s.tc)^2 / (2σt^2)) * cos(ωd * t)
    end
end

# Full time-dependent Hamiltonian for climbing |g,0⟩ -> |g,N⟩ with
# number-selective pulses. Defaults are a numerically-tuned operating point
# (checked against N=1..5): with g=1 fixing the time unit, Δbig/g=30
# keeps the dispersive approximation's per-step leakage small even as
# coupling grows with n (block n's mixing angle scales as √(n+1)g/Δbig, so
# deeper dispersive regimes matter more at higher N); σt/g=60 keeps adjacent
# number-split transitions spectrally resolved (2χ·σt ~ 4) without
# needlessly lengthening the protocol; τramp/g=0.001 makes each Δ(t) ramp
# fast relative to every step's swap time — needed for good fidelity, since
# each step uses the exact ideal swap time π/(2g√(n+1)) with no correction
# factor, so a ramp that isn't fast enough shows up directly as lost
# fidelity.
function fock_ladder_hamiltonian_selective(cs, N; g=1.0, Δbig=30.0, σt=60.0, τramp=0.001)
    χ = g^2 / Δbig
    Ω0 = π / (σt * sqrt(2π))
    steps, tfinal = build_schedule(N; g, σt, τramp)
    Hstatic = jaynes_cummings(cs, :qubit, :osc, g)
    nq, σx = op(cs, :qubit, :n), op(cs, :qubit, :σx)
    Δfun = detuning_fn(steps, Δbig, τramp)
    Ωfun = drive_fn_selective(steps, Δbig, χ, σt, Ω0)
    H = add_time_dependence(Hstatic, Δfun => nq, Ωfun => σx)
    H, steps, tfinal, Δfun, envelope_fn(steps, σt, Ω0)
end
