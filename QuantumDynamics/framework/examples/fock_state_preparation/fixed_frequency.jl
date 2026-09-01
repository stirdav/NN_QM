# Fixed-frequency addressing: the protocol Chu et al. (2018) actually
# demonstrate. A single, broadband, short π-pulse at one fixed
# frequency ωd is used for every step — no per-step retuning. This works
# without spectroscopic selectivity because the sequence is open-loop and
# deterministic: at the moment step n's pulse fires, population only exists
# in |g,n⟩ (everything else has already been swapped away by earlier steps),
# so there is nothing else nearby for an untuned pulse to accidentally
# address. The trade-off is that this determinism is exactly what a real
# device can't fully guarantee — any leftover population from an imperfect
# earlier swap sits in the wrong manifold and gets hit by the next step's
# pulse anyway, since there is no frequency selectivity to shield it.

# Drive coefficient: one Gaussian-enveloped π-pulse per step, all at the same
# fixed frequency ωd (by default the n=0 transition, Δbig+χ, matching the
# paper's single fixed idle detuning ν0 — no retuning between steps).
function drive_fn_fixed(steps, ωd, σt, Ω0)
    t -> sum(steps) do s
        Ω0 * exp(-(t - s.tc)^2 / (2σt^2)) * cos(ωd * t)
    end
end

# Full time-dependent Hamiltonian for climbing |g,0⟩ -> |g,N⟩ with a single
# fixed pulse frequency. Same Δbig/g and τramp/g as the selective scheme (for
# a like-for-like comparison at the same physical operating point), but σt is
# much *shorter* — short enough that the pulse bandwidth comfortably exceeds
# the full number-splitting spread swept over N steps (so it never needs to
# resolve individual n-manifolds), while remaining long enough that Δbig
# still keeps the pulse off the |e,0⟩<->|g,1⟩-type resonance.
# σt=0.4 sits near an empirical fidelity optimum (checked at N=3):
# smaller σt raises Ω0 until it's no longer small relative to ωd, where the
# counter-rotating term neglected by the implicit RWA produces a
# Bloch-Siegert-like shift that detunes the pulse; larger σt narrows the
# bandwidth below the number-splitting spread swept over N steps, leaving
# later steps (whose transition sits off the fixed ωd) under-rotated.
function fock_ladder_hamiltonian_fixed(cs, N; g=1.0, Δbig=30.0, σt=0.4, τramp=0.001,
    ωd=Δbig + g^2 / Δbig)
    Ω0 = π / (σt * sqrt(2π))
    steps, tfinal = build_schedule(N; g, σt, τramp)
    Hstatic = jaynes_cummings(cs, :qubit, :osc, g)
    nq, σx = op(cs, :qubit, :n), op(cs, :qubit, :σx)
    Δfun = detuning_fn(steps, Δbig, τramp)
    Ωfun = drive_fn_fixed(steps, ωd, σt, Ω0)
    H = add_time_dependence(Hstatic, Δfun => nq, Ωfun => σx)
    H, steps, tfinal, Δfun, envelope_fn(steps, σt, Ω0)
end
