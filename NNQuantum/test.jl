# test.jl
#
# Step 2 smoke/validation run (see CLAUDE.md / DESIGN.md) for the FL_1step_3p
# HBAR-qubit protocol built in HBAR-qubit_problem.jl: fix a concrete initial
# state, simulate a short test trajectory, and — if the simulation succeeds
# (well-formed states, trace preserved) — plot ⟨n_qubit⟩/⟨n_osc⟩ over time.
#
# Run from NNQuantum/ (REPL: include("test.jl"), or `julia test.jl`).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
haskey(Pkg.project().dependencies, "CairoMakie") || Pkg.add("CairoMakie")

using QuantumDynamics
using QuantumOptics

include("HBAR-qubit_problem.jl")   # defines cs, H0, J, π_pulse_shape, FLstep_dynamics_3p
                                     # (this in turn includes Definition.jl)

# --- 1. Initial state: qubit down ⊗ oscillator Fock |0⟩ -----------------------

qubit_basis = getsubsystem(cs, :qubit).basis
osc_basis   = getsubsystem(cs, :osc).basis
initial_state = spindown(qubit_basis) ⊗ fockstate(osc_basis, 0)

# --- 2. Short test timespan -----------------------------------------------------
#
# FLstep_dynamics_3p itself only samples the solver output at each stage's two
# endpoints (per-stage tspan = (t_start, t_end)), which is far too coarse to
# plot a trajectory. This local runner mirrors FLstep_dynamics_3p's structure
# exactly (same H0/J/π_pulse_shape from the include, same two-stage
# spin-flip-then-SWAP hand-off) but samples each stage at `npoints` points, for
# plotting resolution — not a change to the pipeline itself, just this test's
# own finer output grid. Pulse parameters are arbitrary, short, "does it run"
# values (per this task's request), not physically tuned per Chu et al.

function run_short_dynamics(t0, initial_state, τ_exc, ωd, τ_SWAP; npoints=200)
    Ω_R = π / τ_exc
    Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(ωd * t)
    Δ(t) = -Δ0 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    H = add_time_dependence(H0,
        (t -> Ω(t))   => op(cs, :qubit, :σx),
        (t -> Δ(t)/2) => op(cs, :qubit, :σz))

    tspan1 = range(t0, t0 + τ_exc; length=npoints)
    _, states1 = evolve(tspan1, initial_state, H, J)
    ψ_at_flip = states1[end]

    tspan2 = range(t0 + τ_exc, t0 + τ_exc + τ_SWAP; length=npoints)
    _, states2 = evolve(tspan2, ψ_at_flip, H, J)

    return vcat(collect(tspan1), collect(tspan2[2:end])), vcat(states1, states2[2:end])
end

t0     = 0.0
τ_exc  = 1e-3
τ_SWAP = 1e-3
ωd     = Δ0   # arbitrary free drive frequency for this smoke test — not tuned

tspan, states = run_short_dynamics(t0, initial_state, τ_exc, ωd, τ_SWAP)

# --- 3. Success check, then plot ⟨n_qubit⟩/⟨n_osc⟩ -----------------------------

trace_drift = abs(real(tr(states[end])) - 1.0)
simulation_ok = length(states) == length(tspan) && all(s -> s isa AbstractOperator, states) && trace_drift < 1e-3

if simulation_ok
    println("Simulation succeeded: $(length(states)) states, tr(ρ_final) = $(real(tr(states[end])))")

    using CairoMakie

    n_qubit_expect = real.(expect(op(cs, :qubit, :n), states))
    n_osc_expect   = real.(expect(op(cs, :osc, :n), states))

    fig = Figure(size=(900, 550))
    ax = Axis(fig[1, 1], xlabel="t", ylabel="⟨n⟩",
        title="HBAR-qubit FL_1step_3p — test run (τ_exc=$(τ_exc), τ_SWAP=$(τ_SWAP), ωd=$(ωd))")
    lines!(ax, tspan, n_qubit_expect, label="⟨n_qubit⟩")
    lines!(ax, tspan, n_osc_expect, label="⟨n_osc⟩")
    vlines!(ax, [t0 + τ_exc], color=(:gray, 0.5), linestyle=:dash)
    axislegend(ax, position=:rc)

    save(joinpath(@__DIR__, "test_plot.png"), fig)
    println("Plot saved to test_plot.png")
else
    println("Simulation did NOT succeed (trace drift = $trace_drift) — skipping plot.")
end
