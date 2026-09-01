include("system.jl")
include("resonant_drive.jl")
using CairoMakie

N = 3

function level_populations(r, N)
    qb, ob = getsubsystem(r.cs, :qubit).basis, getsubsystem(r.cs, :osc).basis
    check_fock_cutoff(r.cs, :osc, r.states)
    [abs2(dagger(spindown(qb) ⊗ fockstate(ob, k)) * ψ) + abs2(dagger(spinup(qb) ⊗ fockstate(ob, k)) * ψ)
     for ψ in r.states, k in 0:N]
end

function shade_swaps!(ax, r)
    for s in r.steps
        vspan!(ax, s.c, s.d, color=(:gray, 0.15))  # shade each resonant-swap window
    end
end

r = run_fock_prep(N, fock_ladder_hamiltonian_resonant)
pop = level_populations(r, N)

fig = Figure(size=(650, 550))

ax1 = Axis(fig[1, 1], ylabel="population", title="Resonant drive/swap: |g,0⟩ → |g,$N⟩")
for k in 0:N
    lines!(ax1, r.tout, pop[:, k+1], label="n=$k")
end
shade_swaps!(ax1, r)
axislegend(ax1, position=:rc)

ax2 = Axis(fig[2, 1], xlabel="t (1/g)", ylabel="drive Ω(t)")
lines!(ax2, r.tout, r.Ωfun.(r.tout), color=:black)
shade_swaps!(ax2, r)

linkxaxes!(ax1, ax2)
hidexdecorations!(ax1, grid=false)
rowsize!(fig.layout, 2, Relative(0.25))

save("population_ladder.png", fig)
