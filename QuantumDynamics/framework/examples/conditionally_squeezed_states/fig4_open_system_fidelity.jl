include("system.jl")
using CairoMakie

nmax = 15                              # tuned for Fock-cutoff convergence
cs = build_system(nmax=nmax)
g, ωm, A, ωd = 1.0, ωm_over_g, 2.405 / 2 * ωm_over_g, ωm_over_g   # 2A/ωd=2.405 (first root of J0), ωd=ωm
tspan = range(0, 1.2; length=200)      # gt up to 1.2, matching the paper

qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
# Tensor order must match CompositeSystem(qubit, osc)'s subsystem order (qubit, then osc).
ψ0 = normalize(spinup(qb) + spindown(qb)) ⊗ fockstate(ob, 0)
Ht = H1(cs, g, ωm, A, ωd)

# Default adaptive-solver tolerances aren't tight enough against H1's fast
# 2*ωm counter-rotating term at the paper's ωm/g=1e4 ratio (norm drifted to
# ~0.998 instead of 1 with defaults) — tightened here.
const SOLVER_TOL = (reltol=1e-9, abstol=1e-11)

_, closed_states = evolve(tspan, ψ0, Ht; SOLVER_TOL...)        # ideal, noiseless reference

fig = Figure()
ax = Axis(fig[1, 1], xlabel="gt", ylabel="Fidelity")
for (γ1_over_g, γφ_over_g) in ((0.1, 0.1), (1.0, 0.1), (0.1, 1.0), (1.0, 1.0))
    J = jump_operators(cs, vcat(
        [Decay(:qubit, γ1_over_g * g), Dephasing(:qubit, γφ_over_g * g)],
        thermal_bath(:osc, γm_over_g * g, nm_th),
    ))
    _, open_states = evolve(tspan, ψ0, Ht, J; SOLVER_TOL...)

    fids = map(zip(closed_states, open_states)) do (ψc, ρo)
        ρc_cond = conditional_oscillator_state(cs, ψc, :plus)
        ρo_cond = conditional_oscillator_state(cs, ρo, :plus)
        real(fidelity(ρc_cond, ρo_cond))
    end
    lines!(ax, collect(tspan), fids, label="γ1/g=$γ1_over_g, γφ/g=$γφ_over_g")
end
axislegend(ax)
save("fig4.png", fig)
