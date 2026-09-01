include("system.jl")
include("selective_pulses.jl")
include("fixed_frequency.jl")
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

function plot_ladder!(ax, r, levelpop, N)
    for k in 0:N
        lines!(ax, r.tout, levelpop[:, k+1], label="n=$k")
    end
    shade_swaps!(ax, r)
end

function plot_detuning!(ax, r)
    lines!(ax, r.tout, r.Δfun.(r.tout), color=:black)
    shade_swaps!(ax, r)
end

function plot_drive!(ax, r)
    lines!(ax, r.tout, r.Ωenv.(r.tout), color=:black)
    shade_swaps!(ax, r)
end

r_sel = run_fock_prep(N, fock_ladder_hamiltonian_selective)
# The fixed-frequency scheme's broadband pulse leaks slightly further into
# higher Fock levels than the selective scheme's narrowband one (see
# `pulse_scheme_comparison.jl`), so it needs a bit more cutoff margin.
r_fix = run_fock_prep(N, fock_ladder_hamiltonian_fixed; nmax_margin=6)
pop_sel = level_populations(r_sel, N)
pop_fix = level_populations(r_fix, N)

fig = Figure(size=(1000, 750))

ax1 = Axis(fig[1, 1], ylabel="population", title="Number-selective: |g,0⟩ → |g,$N⟩")
plot_ladder!(ax1, r_sel, pop_sel, N)
axislegend(ax1, position=:rc)

ax2 = Axis(fig[1, 2], ylabel="population", title="Fixed-frequency: |g,0⟩ → |g,$N⟩")
plot_ladder!(ax2, r_fix, pop_fix, N)
axislegend(ax2, position=:rc)

ax3 = Axis(fig[2, 1], ylabel="Δ(t)")
plot_detuning!(ax3, r_sel)

ax4 = Axis(fig[2, 2], ylabel="Δ(t)")
plot_detuning!(ax4, r_fix)

ax5 = Axis(fig[3, 1], xlabel="t (1/g)", ylabel="drive envelope |Ω(t)|")
plot_drive!(ax5, r_sel)

ax6 = Axis(fig[3, 2], xlabel="t (1/g)", ylabel="drive envelope |Ω(t)|")
plot_drive!(ax6, r_fix)

linkxaxes!(ax1, ax3, ax5)
linkxaxes!(ax2, ax4, ax6)
hidexdecorations!(ax1, grid=false)
hidexdecorations!(ax2, grid=false)
hidexdecorations!(ax3, grid=false)
hidexdecorations!(ax4, grid=false)
rowsize!(fig.layout, 2, Relative(0.2))
rowsize!(fig.layout, 3, Relative(0.2))

save("population_ladder.png", fig)
