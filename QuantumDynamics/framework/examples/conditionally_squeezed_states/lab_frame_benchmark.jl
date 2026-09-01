include("system.jl")

# Sanity check + cost benchmark: simulate the state-preparation protocol
# directly in the lab frame (Hlab, Eq. 1 + Eq. 2 — ωq resolved explicitly,
# no rotating-frame transform) rather than in H1's interaction frame
# (Eq. S4, ωq rotated away). No approximation is involved in going between
# the two — H1 is derived from Hlab by the exact unitary V1 of Eq. S3 — so
# this is a benchmark of simulation cost, not of physics validity: fig2's
# own H1-vs-Hcs comparison already validates the physics against the
# paper's reference implementation.
#
# Single run at Fig. 2's own best operating point (on-resonance drive
# Ā=2.405, the smallest tested g/ωm=0.001) rather than a sweep.

const nmax = 50                            # matches fig2_sensitivity_sweep.jl
const g = 1.0
const g_over_ωm = 0.001                    # Fig. 2 panel (b)'s best-fidelity point
const Abar = 2.405                         # Fig. 2 panel (a)'s ideal (first root of J0)
const ωm = g / g_over_ωm
const ωd = ωm
const A = Abar * ωd / 2
const ωq = 20ωm                            # paper's device ratio ωq/ωm=20 (§Open system dynamics: ωq=20GHz, ωm=1GHz)
const tspan = range(0, 1.2; length=200)    # gt up to 1.2, matching Fig. 1/2/4's |ξ|≈1 end time
const SOLVER_TOL = (reltol=1e-9, abstol=1e-11)

cs = build_system(nmax=nmax, ωm=ωm, ωq=ωq)
qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
# Tensor order must match CompositeSystem(qubit, osc)'s subsystem order (qubit, then osc).
ψ0 = normalize(spinup(qb) - spindown(qb)) ⊗ fockstate(ob, 0)

Ht_int = H1(cs, g, ωm, A, ωd)
Ht_lab = Hlab(cs, ωq, ωm, g, A, ωd)

# Warm up (JIT-compile) each solver path on a couple of steps before timing
# the full run, so @elapsed reflects integration cost, not compilation.
evolve(tspan[1:2], ψ0, Ht_int; SOLVER_TOL...)
evolve(tspan[1:2], ψ0, Ht_lab; SOLVER_TOL...)

t_int = @elapsed (_, states_int) = evolve(tspan, ψ0, Ht_int; SOLVER_TOL...)
t_lab = @elapsed (_, states_lab) = evolve(tspan, ψ0, Ht_lab; SOLVER_TOL...)

check_fock_cutoff(cs, :osc, states_int)
check_fock_cutoff(cs, :osc, states_lab)

# Sanity check: states_lab, transformed into the interaction frame via
# V1(t) = exp[i t (ωq/2 σz + ωm n)] (Eq. S3's first factor — the same,
# exact transform H1 is derived from), should match states_int up to
# solver tolerance regardless of the two Hamiltonians' very different
# computational cost.
σz, n = op(cs, :qubit, :σz), op(cs, :osc, :n)
Htrans = ωq / 2 * σz + ωm * n
fids = map(zip(tspan, states_lab, states_int)) do (t, ψlab, ψint)
    ψ_transformed = exp(im * t * Htrans) * ψlab
    real(abs2(ψ_transformed' * ψint))
end

# A previous run (Apple Silicon, single-threaded) produced: interaction
# frame (H1) 0.068-0.071s, lab frame (Hlab) 1.51-1.55s, ~22x cost ratio,
# frame-transform fidelity range [0.99999, 1.0] — exact numbers will vary
# by machine and Julia version.
println("evolve wall-clock:")
println("  interaction frame (H1):   $(round(t_int, digits=3)) s")
println("  lab frame (Hlab, ωq=$(ωq)):  $(round(t_lab, digits=3)) s")
println("  lab-frame / interaction-frame cost ratio: $(round(t_lab / t_int, digits=1))x")
println("frame-transform fidelity range: [$(minimum(fids)), $(maximum(fids))]")
