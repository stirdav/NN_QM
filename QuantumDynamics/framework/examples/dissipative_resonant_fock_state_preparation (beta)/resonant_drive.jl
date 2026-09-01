# Always-on resonant Jaynes-Cummings coupling plus smooth qubit-drive plateaus:
#
#   H(t) = g*(a†σ⁻ + aσ⁺) + Ω(t)*σx.
#
# Because both subsystem frequencies are zero in the co-rotating frame,
# `jaynes_cummings` contributes only the exchange term. `w` and `τ` must use
# the same time unit that is reciprocal to the frequency unit used for `g`.
function fock_ladder_hamiltonian_resonant(cs, N; g=1.0, w=0.01, τ=0.1w)
    steps, tfinal = build_schedule(N; g, w, τ)
    Hstatic = jaynes_cummings(cs, :qubit, :osc, g)
    σx = op(cs, :qubit, :σx)
    Ωfun = envelope_fn(steps, w, τ)
    H = add_time_dependence(Hstatic, Ωfun => σx)
    H, steps, tfinal, Ωfun, τ
end
