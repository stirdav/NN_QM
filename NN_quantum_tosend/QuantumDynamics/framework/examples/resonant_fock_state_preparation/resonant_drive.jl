# Resonant drive/swap scheme: qubit and oscillator stay on resonance (Δ=0)
# throughout — there is no detuning knob at all, unlike
# `fock_state_preparation`. The coupling g*(ad*σm + a*σp) is therefore a
# genuinely time-independent operator; only the qubit drive Ω(t)*σx is
# switched, in plateaus separated in time from each swap window.
#
# Because the coupling never turns off, sitting in |g,n⟩ during step n's
# drive plateau leaves the system weakly coupled to |e,n-1⟩ (matrix element
# g*sqrt(n)) throughout the pulse — the reverse of the swap that produced
# |g,n⟩ in the first place. The drive plateau must be short enough that this
# stray coupling has no time to act: g*sqrt(N)*w << 1 (worst case at the
# last step), i.e. the pulse must be fast compared to a local vacuum-Rabi
# period, not just short compared to a single swap_time(n,g). See
# `leakage_scan.jl` for a numerical check of this bound and README.md for
# the derivation.
function fock_ladder_hamiltonian_resonant(cs, N; g=1.0, w=0.01, τ=0.1w)
    steps, tfinal = build_schedule(N; g, w, τ)
    Hstatic = jaynes_cummings(cs, :qubit, :osc, g)
    σx = op(cs, :qubit, :σx)
    Ωfun = envelope_fn(steps, w, τ)
    H = add_time_dependence(Hstatic, Ωfun => σx)
    H, steps, tfinal, Ωfun, τ
end
