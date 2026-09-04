# test.jl
#
# Step 2 smoke/validation run (see CLAUDE.md / DESIGN.md) for the FL_1step_3p
# HBAR-qubit protocol built in FockLadder_problem.jl (renamed from
# HBAR-qubit_problem.jl): fix a concrete initial
# state, simulate a short test trajectory, and — if the simulation succeeds
# (well-formed states, trace preserved) — plot ⟨n_qubit⟩/⟨n_osc⟩ over time.
#
# Also covers a second, separate smoke test (section 4 below): dataset
# generation + saving for FL_1step_3p's ladder step 1, mirroring
# old_version/HBAR-qubit_problem/Chu_DFL_execution.ipynb's :FL_1step_3p /
# :master_dynamic cell (id dfd162e7) at a tiny n_samples, dataset-generation
# only (no NN training — that's step 3, not this).
#
# Sections 6-7 cover the :FL_1step_Bspline protocol (DESIGN.md Part 19,
# Bspline_Fock_problem.jl/Bspline_Fock_execution.jl): a drive+trajectory
# smoke test (6) and a dataset-generation smoke test targeting Fock state 1
# (7), the same two-part structure as :FL_1step_3p's own sections 1-3/4.
#
# Run from NNQuantum/ (REPL: include("test.jl"), or `julia test.jl`).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using QuantumDynamics
using QuantumOptics

include("FockLadder_execution.jl")     # :FL_1step_3p — includes Definition.jl, NNQuantum.jl,
                                         # FockLadder_problem.jl; gives sections 1-4 below
                                         # cs/H0/J/FLstep_dynamics_3p/_inputs/_outputs/
                                         # run_Fock_ladder etc.
include("Bspline_Fock_execution.jl")   # :FL_1step_Bspline (DESIGN.md Part 19) — its own,
                                         # standalone chain (includes Definition.jl/
                                         # NNQuantum.jl again, harmlessly — see that file's
                                         # own header note on why it isn't `include`-dependent
                                         # on FockLadder_problem.jl); gives sections 6-7 below
                                         # cs_res/Hon/J_res/FLstep_dynamics_Bspline/_inputs/
                                         # _outputs/run_FockTarget_Bspline etc.

# --- 1. Initial state: qubit down ⊗ oscillator Fock |0⟩ -----------------------

qubit_basis = getsubsystem(cs, :qubit).basis
osc_basis   = getsubsystem(cs, :osc).basis
initial_state = spindown(qubit_basis) ⊗ fockstate(osc_basis, 0)

# --- 2. Short test timespan -----------------------------------------------------
#
# FLstep_dynamics_3p itself only samples the solver output at each stage's two
# endpoints (per-stage tspan = (t_start, t_end)), which is far too coarse to
# plot a trajectory — FLstep_dynamics_3p_dense (FockLadder_problem.jl) samples
# each stage at `npoints` points instead, for plotting resolution. Pulse
# parameters are arbitrary, short, "does it run" values (per this task's
# original request), not physically tuned per Chu et al.

t0     = 0.0
τ_exc  = 1e-3
τ_SWAP = 1e-3
ωd     = Δ0   # arbitrary free drive frequency for this smoke test — not tuned

tspan, states = FLstep_dynamics_3p_dense(t0, initial_state, τ_exc, ωd, τ_SWAP)

# --- 3. Success check, then plot ⟨n_qubit⟩/⟨n_osc⟩ -----------------------------

trace_drift = abs(real(tr(states[end])) - 1.0)
simulation_ok = length(states) == length(tspan) && all(s -> s isa AbstractOperator, states) && trace_drift < 1e-3

if simulation_ok
    println("Simulation succeeded: $(length(states)) states, tr(ρ_final) = $(real(tr(states[end])))")

    plot_trajectory(tspan, states,
        ["n_qubit" => op(cs, :qubit, :n), "n_osc" => op(cs, :osc, :n)];
        title="HBAR-qubit FL_1step_3p — test run (τ_exc=$(τ_exc), τ_SWAP=$(τ_SWAP), ωd=$(ωd))",
        ylabel="⟨n⟩", vlines=[t0 + τ_exc],
        save_path=joinpath(@__DIR__, "test_plot.png"))
    println("Plot saved to test_plot.png")
else
    println("Simulation did NOT succeed (trace drift = $trace_drift) — skipping plot.")
end

# --- 4. Dataset generation smoke test: FL_1step_3p, ladder step 1 -------------
#
# Mirrors Chu_DFL_execution.ipynb's :FL_1step_3p / :master_dynamic cell
# (old_version/HBAR-qubit_problem/Chu_DFL_execution.ipynb, cell id
# dfd162e7 — the dissipative variant, matching FockLadder_problem.jl's
# always-dissipative FLstep_dynamics_3p), restricted to step=1 of that
# cell's `for step in 1:N_steps` loop, and to n_samples=5 instead of that
# cell's 800 — a "does dataset generation run and save correctly" smoke
# test, not a real dataset run. Dataset-generation only: the notebook cell
# also trains a NN on the dataset it builds, which has no NNQuantum
# equivalent yet (step 3, unstarted, see CLAUDE.md).

# Step-1 target states — mirrors old_version's
# creation_step_states(ρ0, 1) (HBAR-qubit_problem.jl:241-255): the
# spin-flip stage only flips the qubit, oscillator stays at its pre-step
# Fock level (step_number-1 = 0); the SWAP stage then moves one phonon in
# (n = step_number = 1) and flips the qubit back down — one rung of the
# Fock ladder.
step_number = 1
target_spinflip = spinup(qubit_basis) ⊗ fockstate(osc_basis, step_number - 1)
target_final    = spindown(qubit_basis) ⊗ fockstate(osc_basis, step_number)

# Decomposition basis over the full composite Hilbert space — mirrors
# old_version's SUN_basis = rand_hermitian_orthonormal_basis(dimension, basis)
# (dimension = 2*(N_mech+1), the full Ket dimension, per ML_QM_execution.jl).
d_dataset = length(cs.basis)
decom_basis = rand_hermitian_orthonormal_basis(d_dataset, cs.basis)

# Parameter ranges for step=1, ported from the notebook cell above
# (g there is set to 258 in the "NN Fock ladder (g = 258)" section, matching
# FockLadder_problem.jl's own g — no rescaling needed):
#   τ_exc ∈ [1e-5, 5e-2]
#   ωd    ∈ [Δ0-100, Δ0+100]           (δ0/δ in the notebook)
#   τ_SWAP ∈ [t_swap_th-δ1, t_swap_th+δ2], t_swap_th = π/(2g√step_number)
T_exc_ini, T_exc_fin = 1e-5, 5e-2

δ_ωd = 100.0
ωd_ini, ωd_fin = Δ0 - δ_ωd, Δ0 + δ_ωd

t_swap_th = π / (2 * g * sqrt(step_number))
δ1, δ2 = 0.0005, 0.00015
τ_SWAP_ini, τ_SWAP_fin = t_swap_th - δ1, t_swap_th + δ2

parameters_range = [[T_exc_ini, T_exc_fin], [ωd_ini, ωd_fin], [τ_SWAP_ini, τ_SWAP_fin]]
dim_parameters_space = [100, 100, 100]   # notebook's dim_x,dim_y,dim_z for the :master_dynamic cell
prs(τ_exc, ωd, τ_SWAP) = 1 / prod(dim_parameters_space)   # uniform, matches the notebook's prs

n_samples_dataset = 5

outputs_dataset = FL_1step_3p_NN_outputs(prs, parameters_range, dim_parameters_space, n_samples_dataset; uniform=true)
println("Sampled (τ_exc, ωd, τ_SWAP) triples: ", outputs_dataset)

inputs_dataset = FL_1step_3p_NN_inputs(t0, initial_state, target_spinflip, target_final, decom_basis, outputs_dataset, n_samples_dataset)

dataset_ok = length(inputs_dataset) == n_samples_dataset &&
             all(row -> length(row) == d_dataset^2 + 2, inputs_dataset)

if dataset_ok
    println("Dataset generation succeeded: $(n_samples_dataset) samples, $(length(inputs_dataset[1])) features each.")

    dataset_path = joinpath(@__DIR__, "dataset_step1.jld2")
    save_dataset(dataset_path, inputs_dataset, outputs_dataset;
        description="FL_1step_3p ladder step $(step_number) dataset generation smoke test",
        params=Dict{Symbol,Any}(:step=>step_number, :n_samples=>n_samples_dataset, :N_mech=>N_mech, :g=>g))
    println("Dataset saved to ", dataset_path)
else
    println("Dataset generation did NOT succeed — skipping save.")
end

# Section 5 (staged validation of run_Fock_ladder, (v) in CLAUDE.md's Plan)
# used to live here as a local copy of generate_decom_basis/run_Fock_ladder.
# That copy was validated (N_steps=1 then 2, small/fast hyperparameters,
# user-confirmed against the plotted trajectories — see test_log.md) and has
# since been promoted, as reviewed, into FockLadder_execution.jl — that file
# now holds the one real run_Fock_ladder/generate_decom_basis. See
# DESIGN.md Part 13 for the promotion record.

# --- 6. :FL_1step_Bspline — B-spline drive + single-window dynamics -----------
#
# One instance of a random B-spline-decomposed drive, run once through
# FLstep_dynamics_Bspline_dense, plotted as two subplots in one figure — the
# drive ΩRe(t)/ΩIm(t) itself (top) and the ⟨n_qubit⟩/⟨n_osc⟩ trajectory it
# produces (bottom), so the two can be visually compared directly.
# Coefficients/T are arbitrary "does it run, does it look sensible" values,
# not physically tuned or learned — this section predates run_FockTarget_
# Bspline/section 7 below and isn't meant to exercise it.
#
# cs_res/qubit_res_basis/osc_res_basis/generate_Bspline_basis/
# FLstep_dynamics_Bspline_dense now come from Bspline_Fock_execution.jl
# (DESIGN.md Part 19), not FockLadder_problem.jl — see this file's own
# top-matter comment.

degree_bspline = 4     # cubic
n_basis_bspline = 10
domain_bspline = (0.0, 1.0)
basis_spline = generate_Bspline_basis(degree_bspline, n_basis_bspline, domain_bspline)

initial_state_res = spindown(qubit_res_basis) ⊗ fockstate(osc_res_basis, 0)

T_bspline = 0.03   # arbitrary window duration, comparable order to :FL_1step_3p's τ_exc+τ_SWAP

using Random
Random.seed!(1)
coeffs_Re = 50.0 .* (2 .* rand(n_basis_bspline) .- 1)
coeffs_Im = 50.0 .* (2 .* rand(n_basis_bspline) .- 1)

tspan_bspline, states_bspline = FLstep_dynamics_Bspline_dense(t0, initial_state_res, T_bspline, coeffs_Re, coeffs_Im, basis_spline)

trace_drift_bspline = abs(real(tr(states_bspline[end])) - 1.0)
simulation_ok_bspline = length(states_bspline) == length(tspan_bspline) &&
                         all(s -> s isa AbstractOperator, states_bspline) &&
                         trace_drift_bspline < 1e-3

if simulation_ok_bspline
    println("Bspline simulation succeeded: $(length(states_bspline)) states, tr(ρ_final) = $(real(tr(states_bspline[end])))")

    splineRe = Bspline_composition(coeffs_Re, basis_spline)
    splineIm = Bspline_composition(coeffs_Im, basis_spline)
    drive_ΩRe = [splineRe((t - t0) / T_bspline) for t in tspan_bspline]
    drive_ΩIm = [-splineIm((t - t0) / T_bspline) for t in tspan_bspline]   # -ΩIm(t), matching Hon's σy coefficient

    fig = Figure(size=(900, 700))

    ax1 = Axis(fig[1, 1], xlabel="t", ylabel="drive amplitude", title="FL_1step_Bspline — drive (T=$(T_bspline))")
    lines!(ax1, tspan_bspline, drive_ΩRe, label="ΩRe(t)  [σx coeff]")
    lines!(ax1, tspan_bspline, drive_ΩIm, label="-ΩIm(t)  [σy coeff]")
    axislegend(ax1, position=:rc)

    ax2 = Axis(fig[2, 1], xlabel="t", ylabel="⟨n⟩", title="FL_1step_Bspline — trajectory")
    lines!(ax2, tspan_bspline, real.(expect(op(cs_res, :qubit, :n), states_bspline)), label="n_qubit")
    lines!(ax2, tspan_bspline, real.(expect(op(cs_res, :osc, :n), states_bspline)), label="n_osc")
    axislegend(ax2, position=:rc)

    save(joinpath(@__DIR__, "test_plot_bspline.png"), fig)
    println("Plot saved to test_plot_bspline.png")
else
    println("Bspline simulation did NOT succeed (trace drift = $trace_drift_bspline) — skipping plot.")
end

# --- 7. :FL_1step_Bspline dataset generation smoke test, target Fock = 1 ------
#
# Mirrors section 4's :FL_1step_3p dataset-generation smoke test structurally
# — tiny n_samples, checks shapes, saves — but for FL_1step_Bspline_NN_
# outputs/_inputs (DESIGN.md Part 19): a "does dataset generation run and
# save correctly" check, not a real dataset run (that's run_FockTarget_
# Bspline's own n_samples=800 default, exercised separately below at a
# cheap scale). Reuses basis_spline/n_basis_bspline from section 6.

target_final_bspline = spindown(qubit_res_basis) ⊗ fockstate(osc_res_basis, 1)   # Fock target = 1

d_res = length(cs_res.basis)
decom_basis_bspline = rand_hermitian_orthonormal_basis(d_res, cs_res.basis)

T_max_bspline = 0.03
coeff_range_bspline = (-50.0, 50.0)
n_samples_dataset_bspline = 5

pulse_parameters_bspline, outputs_dataset_bspline =
    FL_1step_Bspline_NN_outputs(T_max_bspline, coeff_range_bspline, n_basis_bspline, n_samples_dataset_bspline)
println("Sampled (T, coeffs_Re, coeffs_Im) triples' T values: ", [pp[1] for pp in pulse_parameters_bspline])

inputs_dataset_bspline = FL_1step_Bspline_NN_inputs(
    t0, initial_state_res, target_final_bspline, decom_basis_bspline, basis_spline,
    pulse_parameters_bspline, n_samples_dataset_bspline)

dataset_ok_bspline = length(inputs_dataset_bspline) == n_samples_dataset_bspline &&
                      all(row -> length(row) == d_res^2 + 2, inputs_dataset_bspline) &&
                      all(row -> length(row) == 2 * n_basis_bspline, outputs_dataset_bspline)

if dataset_ok_bspline
    println("Bspline dataset generation succeeded: $(n_samples_dataset_bspline) samples, " *
            "$(length(inputs_dataset_bspline[1])) input features, $(length(outputs_dataset_bspline[1])) output coefficients each.")

    dataset_path_bspline = joinpath(@__DIR__, "dataset_bspline_target1_smoketest.jld2")
    save_dataset(dataset_path_bspline, inputs_dataset_bspline, outputs_dataset_bspline;
        description="FL_1step_Bspline target=1 dataset generation smoke test",
        params=Dict{Symbol,Any}(:n_target=>1, :n_samples=>n_samples_dataset_bspline,
                                 :N_mech_Bspline=>N_mech_Bspline, :g=>g, :T_max=>T_max_bspline))
    println("Dataset saved to ", dataset_path_bspline)
else
    println("Bspline dataset generation did NOT succeed — skipping save.")
end
