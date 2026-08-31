include("system.jl")

# Open-system analogue of lab_frame_benchmark.jl: same cost/sanity
# comparison between the lab frame (Hlab, Eq. 1 + Eq. 2) and H1's
# interaction frame (Eq. S4), now with dissipation, at
# fig4_open_system_fidelity.jl's best (lowest-decoherence) rate
# combination, γ1/g, γφ/g = 0.1, 0.1, rather than a sweep over all four
# combinations that script covers.
#
# ωm: NOT fig4_open_system_fidelity.jl's own ωm/g=1e4 (the paper's real
# device ratio). Resolving the lab frame's fast ωq oscillation is far
# more expensive for the open-system master equation (density matrix)
# than for the closed-system Schrödinger equation (state vector) —
# ωq/g=2e5 (fig4's real ωm combined with the paper's ωq/ωm=20 device
# ratio) makes the default explicit ODE solver (DP5) numerically unstable
# before reaching gt=1.2 at all, and even ωq/g=2e4 (matching
# lab_frame_benchmark.jl's closed-system value, but with ωq/ωm only 2 at
# fig4's ωm) is merely marginal (~150s, right at the instability edge).
# Using fig2_sensitivity_sweep.jl's ωm/g=1000 instead — validated there as
# a regime where the RWA (and hence H1's interaction-frame physics) is
# still an excellent approximation — while keeping the paper's ωq/ωm=20
# ratio intact (so ωq/g=2e4) integrates the full range cleanly.
const nmax = 15                                            # matches fig4_open_system_fidelity.jl
const g, g_over_ωm = 1.0, 0.001                            # fig2's validated-good-RWA operating point
const ωm = g / g_over_ωm
const ωd = ωm
const A = 2.405 / 2 * ωd
const ωq = 20ωm                                            # paper's device ratio ωq/ωm=20
const γ1_over_g, γφ_over_g = 0.1, 0.1                      # fig4's best (lowest-decoherence) combination
const tspan = range(0, 1.2; length=200)                    # gt up to 1.2, matching fig4
const SOLVER_TOL = (reltol=1e-9, abstol=1e-11)

cs = build_system(nmax=nmax, ωm=ωm, ωq=ωq)
qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
# Tensor order must match CompositeSystem(qubit, osc)'s subsystem order (qubit, then osc).
ψ0 = normalize(spinup(qb) + spindown(qb)) ⊗ fockstate(ob, 0)

Ht_int = H1(cs, g, ωm, A, ωd)
Ht_lab = Hlab(cs, ωq, ωm, g, A, ωd)
J = jump_operators(cs, vcat(
    [Decay(:qubit, γ1_over_g * g), Dephasing(:qubit, γφ_over_g * g)],
    thermal_bath(:osc, γm_over_g * g, nm_th),
))

# Warm up (JIT-compile) each solver path on a couple of steps before timing
# the full run, so @elapsed reflects integration cost, not compilation.
evolve(tspan[1:2], ψ0, Ht_int, J; SOLVER_TOL...)
evolve(tspan[1:2], ψ0, Ht_lab, J; SOLVER_TOL...)

t_int = @elapsed (_, states_int) = evolve(tspan, ψ0, Ht_int, J; SOLVER_TOL...)
t_lab = @elapsed (_, states_lab) = evolve(tspan, ψ0, Ht_lab, J; SOLVER_TOL...)

# Sanity check: see the frame-covariance argument above.
σz, n = op(cs, :qubit, :σz), op(cs, :osc, :n)
Htrans = ωq / 2 * σz + ωm * n
fids = map(zip(tspan, states_lab, states_int)) do (t, ρlab, ρint)
    V1 = exp(im * t * Htrans)
    real(fidelity(V1 * ρlab * V1', ρint))
end

# A previous run (Apple Silicon, single-threaded) produced: interaction
# frame (H1) 2.68-2.73s, lab frame (Hlab) 37.3s, ~14x cost ratio,
# frame-transform fidelity range [0.999999, 1.00001] — exact numbers will
# vary by machine and Julia version.
println("evolve wall-clock (open system, γ1/g=$γ1_over_g, γφ/g=$γφ_over_g):")
println("  interaction frame (H1):   $(round(t_int, digits=3)) s")
println("  lab frame (Hlab, ωq=$(ωq)):  $(round(t_lab, digits=3)) s")
println("  lab-frame / interaction-frame cost ratio: $(round(t_lab / t_int, digits=1))x")
println("frame-transform fidelity range: [$(minimum(fids)), $(maximum(fids))]")
