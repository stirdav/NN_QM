include("system.jl")
using CairoMakie

nmax = 30                              # tuned for Wigner-function convergence
cs = build_system(nmax=nmax)
gcs = 1.0
tfinal = 1 / (2gcs)                    # ξ = 2i*gcs*t = i, matching the paper

qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
# Tensor order must match CompositeSystem(qubit, osc)'s subsystem order (qubit, then osc).
ψ0 = normalize(spinup(qb) + spindown(qb)) ⊗ fockstate(ob, 0)
_, states = evolve(0:tfinal/200:tfinal, ψ0, Hcs(cs, gcs))
ψt = states[end]

ρ_sym = conditional_oscillator_state(cs, ψt, :plus)
ρ_anti = conditional_oscillator_state(cs, ψt, :minus)

xvec = yvec = -4:0.1:4
fig = Figure()
for (i, (ρ, title)) in enumerate([(ρ_sym, "symmetric"), (ρ_anti, "antisymmetric")])
    ax = Axis(fig[1, i], title=title, xlabel="Re α", ylabel="Im α")
    heatmap!(ax, xvec, yvec, wigner(ρ, xvec, yvec))
end
save("fig1.png", fig)
