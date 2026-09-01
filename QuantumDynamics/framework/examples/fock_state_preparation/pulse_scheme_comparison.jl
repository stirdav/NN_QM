include("system.jl")
include("selective_pulses.jl")
include("fixed_frequency.jl")
using CairoMakie, Printf

Ns = 1:5

function fidelity_and_duration(N, hamiltonian_fn; kwargs...)
    r = run_fock_prep(N, hamiltonian_fn; kwargs...)
    qb, ob = getsubsystem(r.cs, :qubit).basis, getsubsystem(r.cs, :osc).basis
    check_fock_cutoff(r.cs, :osc, r.states)
    target = spindown(qb) ⊗ fockstate(ob, N)
    fid = abs2(dagger(target) * r.states[end])
    fid, r.tfinal
end

selective = [fidelity_and_duration(N, fock_ladder_hamiltonian_selective) for N in Ns]
# The broadband pulse leaks slightly further into higher Fock levels than the
# selective scheme's narrowband one, so it needs a bit more cutoff margin.
fixed = [fidelity_and_duration(N, fock_ladder_hamiltonian_fixed; nmax_margin=6) for N in Ns]

fig = Figure(size=(900, 400))
ax1 = Axis(fig[1, 1], xlabel="target Fock number N", ylabel="final-state fidelity",
           title="Fidelity: number-selective vs. fixed-frequency")
scatterlines!(ax1, collect(Ns), first.(selective), label="number-selective")
scatterlines!(ax1, collect(Ns), first.(fixed), label="fixed-frequency")
ylims!(ax1, 0, 1.02)
axislegend(ax1, position=:lb)

ax2 = Axis(fig[1, 2], xlabel="target Fock number N", ylabel="protocol duration (1/g)",
           title="Total protocol duration", yscale=log10)
scatterlines!(ax2, collect(Ns), last.(selective), label="number-selective")
scatterlines!(ax2, collect(Ns), last.(fixed), label="fixed-frequency")
axislegend(ax2, position=:lt)

save("pulse_scheme_comparison.png", fig)

println("N   fidelity(selective)  fidelity(fixed)  duration(selective)  duration(fixed)  speedup")
for (N, (fs, ds), (ff, df)) in zip(Ns, selective, fixed)
    @printf("%d   %.4f                %.4f            %8.2f              %8.2f          %.1fx\n",
            N, fs, ff, ds, df, ds / df)
end
