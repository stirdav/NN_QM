include("system.jl")
using CairoMakie

# Visual sanity check: evolve the same open-system lab-frame trajectory as
# lab_frame_benchmark_open_system.jl (Hlab, Fig. 4's best dissipation rates,
# at Fig. 2's ωm/g=1000 operating point), then project the qubit to recover
# the conditionally-squeezed oscillator states of Fig. 1 — checking that
# Hlab reproduces the paper's actual protocol end-to-end, not just the
# state-fidelity numbers lab_frame_benchmark_open_system.jl reports.
#
# Frame adjustment: fig1/fig4's conditional_oscillator_state projects onto
# the FIXED interaction-frame states |±⟩=(|e⟩±|g⟩)/√2 — correct there
# because H1's own frame already has ωq rotated away, so those states are
# time-independent in that frame. In the lab frame the qubit is still
# precessing at ωq, so the physically equivalent measurement is a projector
# that itself rotates with it; concretely, the state must first be rotated
# into the interaction frame via the same exact
# V1(t)=exp[it(ωq/2 σz + ωm n)] used in lab_frame_benchmark[_open_system].jl
# before conditioning on the fixed |±⟩. Skipping this and conditioning the
# raw lab-frame state on the fixed |±⟩ measures the qubit in an arbitrary,
# uncontrolled basis (whatever ωq*t happens to be mod 2π) — both versions
# are computed below so the difference is visible, not just asserted.

const nmax = 15
const g, g_over_ωm = 1.0, 0.001
const ωm = g / g_over_ωm
const ωd = ωm
const A = 2.405 / 2 * ωd
const ωq = 20ωm
const γ1_over_g, γφ_over_g = 0.1, 0.1
const tspan = range(0, 1.2; length=200)
const SOLVER_TOL = (reltol=1e-9, abstol=1e-11)

cs = build_system(nmax=nmax, ωm=ωm, ωq=ωq)
qb, ob = getsubsystem(cs, :qubit).basis, getsubsystem(cs, :osc).basis
ψ0 = normalize(spinup(qb) + spindown(qb)) ⊗ fockstate(ob, 0)

Ht_lab = Hlab(cs, ωq, ωm, g, A, ωd)
J = jump_operators(cs, vcat(
    [Decay(:qubit, γ1_over_g * g), Dephasing(:qubit, γφ_over_g * g)],
    thermal_bath(:osc, γm_over_g * g, nm_th),
))

_, states_lab = evolve(tspan, ψ0, Ht_lab, J; SOLVER_TOL...)
ρ_lab_final = states_lab[end]

σz, n = op(cs, :qubit, :σz), op(cs, :osc, :n)
Htrans = ωq / 2 * σz + ωm * n
V1_final = exp(im * tspan[end] * Htrans)
ρ_int_final = V1_final * ρ_lab_final * V1_final'

ρ_sym_correct = conditional_oscillator_state(cs, ρ_int_final, :plus)
ρ_anti_correct = conditional_oscillator_state(cs, ρ_int_final, :minus)
ρ_sym_naive = conditional_oscillator_state(cs, ρ_lab_final, :plus)
ρ_anti_naive = conditional_oscillator_state(cs, ρ_lab_final, :minus)

xvec = yvec = -4:0.1:4
fig = Figure(size=(800, 800))
rows = [
    ("frame-corrected (V1 applied first)", ρ_sym_correct, ρ_anti_correct),
    ("naive (no frame correction)", ρ_sym_naive, ρ_anti_naive),
]
for (r, (label, ρs, ρa)) in enumerate(rows)
    for (c, (ρ, title)) in enumerate([(ρs, "symmetric"), (ρa, "antisymmetric")])
        ax = Axis(fig[r, c], title="$label: $title", xlabel="Re α", ylabel="Im α")
        heatmap!(ax, xvec, yvec, wigner(ρ, xvec, yvec))
    end
end
save("lab_frame_conditional_squeezing_check.png", fig)
