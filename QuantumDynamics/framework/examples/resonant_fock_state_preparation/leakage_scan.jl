include("system.jl")
include("resonant_drive.jl")
using CairoMakie, Printf

# Numerically checks the validity bound argued in README.md/resonant_drive.jl:
# fidelity should stay high as long as g*sqrt(N)*w << 1, and degrade once the
# drive plateau is no longer fast compared to the local vacuum-Rabi period.
N = 3
g = 1.0
ws = exp10.(range(-3, -0.3, length=25))

function fidelity_at(N, g, w)
    r = run_fock_prep(N, fock_ladder_hamiltonian_resonant; g, w, nmax_margin=6)
    qb, ob = getsubsystem(r.cs, :qubit).basis, getsubsystem(r.cs, :osc).basis
    check_fock_cutoff(r.cs, :osc, r.states)
    target = spindown(qb) ⊗ fockstate(ob, N)
    abs2(dagger(target) * r.states[end])
end

fids = [fidelity_at(N, g, w) for w in ws]
leakage_param = g * sqrt(N) .* ws

fig = Figure(size=(650, 400))
ax = Axis(fig[1, 1], xlabel="g√N · w  (drive-plateau leakage parameter)",
          ylabel="final-state fidelity", xscale=log10,
          title="Fidelity vs. drive-plateau width, N=$N")
scatterlines!(ax, leakage_param, fids)
vlines!(ax, [1.0], color=:red, linestyle=:dash, label="g√N·w = 1")
axislegend(ax, position=:lb)

save("leakage_scan.png", fig)

println("w         g√N·w      fidelity")
for (w, lp, f) in zip(ws, leakage_param, fids)
    @printf("%.5f   %.4f     %.5f\n", w, lp, f)
end
