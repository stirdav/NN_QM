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
# Run from NNQuantum/ (REPL: include("test.jl"), or `julia test.jl`).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using QuantumDynamics
using QuantumOptics
using Dates   # timestamping test_log.md, section 5

include("FockLadder_problem.jl")   # defines cs, H0, J, π_pulse_shape, FLstep_dynamics_3p,
                                     # FLstep_dynamics_3p_dense (this in turn includes
                                     # Definition.jl and NNQuantum.jl — the latter brings in
                                     # CairoMakie/plot_trajectory too)

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

# --- 5. Staged validation of run_Fock_ladder ((v), CLAUDE.md's Plan) ------------
#
# NOT a smoke test like sections 1-4 above — this is a real (if deliberately
# small/fast) run of the train/predict/reiterate ladder algorithm, so this
# section is slower than the rest of the file.
#
# Per this session's explicit request, this does NOT touch
# FockLadder_execution.jl's own run_Fock_ladder/generate_decom_basis — those
# stay exactly as they are. What follows is a copy of both, local to this
# file (safe: test.jl never includes FockLadder_execution.jl, so there's no
# name clash), with two things added and one thing changed relative to the
# original:
#   - saves each step's predicted (τ_exc,ωd,τ_SWAP) via NNQuantum.jl's new
#     save_prediction, to predicted_step<n>_b<id>.jld2;
#   - plots each step's predicted trajectory (⟨n_qubit⟩/⟨n_osc⟩ over time,
#     via FLstep_dynamics_3p_dense + plot_trajectory) to
#     trajectory_step<n>_b<id>.png — using the step's own t0/initial_state,
#     captured before step 4 advances them;
#   - returns `(decom_basis, infidelities, predictions, models, final_states)`
#     instead of `nothing` — the same shape old_version's own
#     execution_dynamic_FL returns (Chu_DFL_execution.ipynb, cell dfd162e7,
#     the :FL_1step_3p/:master_dynamic cell this project already mirrors:
#     `return SUN_basis, infidelities, predictions, models, final_states`),
#     per this session's explicit request. `models` holds this project's own
#     richer per-step bundle (train_NN's/load_nn's NamedTuple — model plus
#     normalization stats) rather than old_version's bare Flux.Chain, since
#     that bundle is what a caller actually needs to reuse or inspect a
#     step's NN (e.g. nn.test_error, used below in the test_log.md entry).

_per_step(x, step) = x isa AbstractVector ? x[step] : x

function _check_per_step_length(x, N_steps, name)
    if x isa AbstractVector && length(x) != N_steps
        throw(ArgumentError("run_Fock_ladder: $name has length $(length(x)), expected $N_steps (one entry per step)"))
    end
end

_decom_basis_path(basis_id, save_dir) = joinpath(save_dir, "decom_basis_b$(basis_id).jld2")
_dataset_path(basis_id, step, save_dir) = joinpath(save_dir, "dataset_step$(step)_b$(basis_id).jld2")
_nn_path(basis_id, step, save_dir) = joinpath(save_dir, "nn_step$(step)_b$(basis_id).jld2")
_predicted_path(basis_id, step, save_dir) = joinpath(save_dir, "predicted_step$(step)_b$(basis_id).jld2")
_trajectory_plot_path(basis_id, step, save_dir) = joinpath(save_dir, "trajectory_step$(step)_b$(basis_id).png")
_full_trajectory_plot_path(basis_id, save_dir) = joinpath(save_dir, "trajectory_full_b$(basis_id).png")

"""
    generate_decom_basis(basis_id::Integer; save_dir=@__DIR__, overwrite=false)

Copied unchanged from `FockLadder_execution.jl` (see that function's own
docstring for the full rationale) — makes a new random decomposition basis
and saves it to `save_dir/decom_basis_b<basis_id>.jld2`.
"""
function generate_decom_basis(basis_id::Integer; save_dir::AbstractString=@__DIR__, overwrite::Bool=false)
    path = _decom_basis_path(basis_id, save_dir)
    isfile(path) && !overwrite && throw(ArgumentError(
        "generate_decom_basis: $path already exists — pass overwrite=true to replace it " *
        "(any dataset/NN already saved with basis_id=$(basis_id) would then no longer match it)"))

    d = length(cs.basis)
    basis_ops = rand_hermitian_orthonormal_basis(d, cs.basis)
    save_operator_basis(path, basis_ops; description="FL_1step_3p decomposition basis b$(basis_id)")
    println("Generated and saved decomposition basis b$(basis_id) to $(path)")
    return path
end

"""
    run_Fock_ladder(N_steps, basis_id; ...)

Validation-local copy of `FockLadder_execution.jl`'s `run_Fock_ladder` — see
this section's own header comment above for exactly what's added/changed
relative to that original. Same meaning for every keyword argument as the
original; see that function's own docstring (`FockLadder_execution.jl`) for
the full per-keyword explanation, not repeated here.
"""
function run_Fock_ladder(N_steps::Integer, basis_id::Integer;
                                    n_samples::Integer=800,
                                    dim_parameters_space::Vector{<:Integer}=[100, 100, 100],
                                    save_dir::AbstractString=@__DIR__,
                                    train_fraction::Real=0.9375,
                                    hidden::Integer=500,
                                    η::Real=1e-4,
                                    epochs::Integer=350,
                                    loss::Union{Function,Symbol}=:mse,
                                    batch_size::Integer=32,
                                    uniform::Bool=true,
                                    dataset_mode::Union{Symbol,AbstractVector{Symbol}}=:generate_and_save,
                                    dataset_paths::Union{Nothing,AbstractString,AbstractVector}=nothing,
                                    nn_mode::Union{Symbol,AbstractVector{Symbol}}=:train_new,
                                    nn_paths::Union{Nothing,AbstractString,AbstractVector}=nothing,
                                    save_nn_dir::Union{Nothing,AbstractString}=save_dir)
    _check_per_step_length(dataset_mode, N_steps, "dataset_mode")
    _check_per_step_length(dataset_paths, N_steps, "dataset_paths")
    _check_per_step_length(nn_mode, N_steps, "nn_mode")
    _check_per_step_length(nn_paths, N_steps, "nn_paths")

    basis_path = _decom_basis_path(basis_id, save_dir)
    isfile(basis_path) || throw(ArgumentError(
        "run_Fock_ladder: no saved basis at $basis_path — call " *
        "generate_decom_basis($basis_id; save_dir=\"$save_dir\") first"))
    decom_basis = load_operator_basis(basis_path, cs.basis)

    prs(τ_exc, ωd, τ_SWAP) = 1 / prod(dim_parameters_space)

    t0 = 0.0
    initial_state = spindown(qubit_basis) ⊗ fockstate(osc_basis, 0)

    infidelities = Float64[]
    predictions = Any[]
    models = Any[]
    final_states = Any[]

    # (new) accumulators for the one cumulative full-run plot, spanning every
    # step's dense trajectory concatenated in absolute time — as opposed to
    # the per-step plots below, which only ever show one step's own window.
    # Each step's own dense_tspan/dense_states already sits at the right
    # absolute time (t0 carries forward correctly across steps, never reset),
    # so concatenation needs no time-shifting — only dropping each step-after-
    # the-first's leading point, which duplicates the previous step's very
    # last (t, state) pair (same convention FLstep_dynamics_3p_dense/
    # FLstep_dynamics_3p already use internally to join their own two stages).
    all_tspan = Float64[]
    all_states = Any[]
    boundary_times = Float64[]   # every spin-flip/SWAP split and step-to-step transition, for vlines

    for step in 1:N_steps
        target_spinflip = spinup(qubit_basis) ⊗ fockstate(osc_basis, step - 1)
        target_final    = spindown(qubit_basis) ⊗ fockstate(osc_basis, step)

        step_nn_mode = _per_step(nn_mode, step)

        if step_nn_mode == :fixed
            nn_path = something(_per_step(nn_paths, step), _nn_path(basis_id, step, save_dir))
            nn = load_nn(nn_path)
            println("step $(step)/$(N_steps): loaded fixed NN from $(nn_path), no dataset/training")

        elseif step_nn_mode ∈ (:train_new, :continue_training)
            T_exc_ini, T_exc_fin = 1e-5, 5e-2

            δ_ωd = 100.0
            ωd_ini, ωd_fin = Δ0 - δ_ωd, Δ0 + δ_ωd

            t_swap_th = π / (2 * g * sqrt(step))
            δ1, δ2 = 0.0005, 0.00015
            τ_SWAP_ini, τ_SWAP_fin = t_swap_th - δ1, t_swap_th + δ2

            parameters_range = [[T_exc_ini, T_exc_fin], [ωd_ini, ωd_fin], [τ_SWAP_ini, τ_SWAP_fin]]

            step_dataset_mode = _per_step(dataset_mode, step)
            if step_dataset_mode == :load
                dataset_path = something(_per_step(dataset_paths, step), _dataset_path(basis_id, step, save_dir))
                X, Y = load_dataset(dataset_path)
                println("step $(step)/$(N_steps): loaded dataset from $(dataset_path)")
            elseif step_dataset_mode ∈ (:generate_and_save, :generate_only)
                outputs = FL_1step_3p_NN_outputs(prs, parameters_range, dim_parameters_space, n_samples; uniform=uniform)
                inputs  = FL_1step_3p_NN_inputs(t0, initial_state, target_spinflip, target_final, decom_basis, outputs, n_samples)
                X, Y = inputs, outputs

                if step_dataset_mode == :generate_and_save
                    save_path = _dataset_path(basis_id, step, save_dir)
                    save_dataset(save_path, inputs, outputs;
                        description="FL_1step_3p ladder step $(step) dataset (validation run, N_steps=$(N_steps), basis_id=$(basis_id))",
                        params=Dict{Symbol,Any}(:step=>step, :basis_id=>basis_id, :n_samples=>n_samples,
                                                 :N_mech=>N_mech, :g=>g, :t0=>t0))
                    println("step $(step)/$(N_steps): saved $(n_samples)-sample dataset to $(save_path)")
                else
                    println("step $(step)/$(N_steps): generated $(n_samples)-sample dataset (not saved)")
                end
            else
                throw(ArgumentError("run_Fock_ladder: unknown dataset_mode $(step_dataset_mode) at step $(step)"))
            end

            if step_nn_mode == :continue_training
                nn_path = something(_per_step(nn_paths, step), _nn_path(basis_id, step, save_dir))
                loaded = load_nn(nn_path)
                nn = train_NN(X, Y; train_fraction=train_fraction, hidden=hidden, η=η, epochs=epochs,
                              loss=loss, batch_size=batch_size, model=loaded.model, opt_state=loaded.opt_state)
            else
                nn = train_NN(X, Y; train_fraction=train_fraction, hidden=hidden, η=η,
                              epochs=epochs, loss=loss, batch_size=batch_size)
            end

            if save_nn_dir !== nothing
                save_path = _nn_path(basis_id, step, save_nn_dir)
                save_nn(save_path, nn; description="FL_1step_3p ladder step $(step) trained NN (validation run, basis_id=$(basis_id))",
                        params=Dict{Symbol,Any}(:step=>step, :basis_id=>basis_id, :N_mech=>N_mech, :g=>g))
                println("step $(step)/$(N_steps): saved trained NN to $(save_path) (test error = $(nn.test_error))")
            end
        else
            throw(ArgumentError("run_Fock_ladder: unknown nn_mode $(step_nn_mode) at step $(step)"))
        end

        # --- 3. Predict the drive parameters that reach target_final ---
        predicted_output, predicted_state, prediction_infidelity =
            predict_drive_parameters(nn, t0, initial_state, target_final, decom_basis)

        println("step $(step)/$(N_steps): predicted (τ_exc,ωd,τ_SWAP) = $(predicted_output), infidelity = $(prediction_infidelity)")

        # --- (new) save predicted parameters ---
        predicted_path = _predicted_path(basis_id, step, save_dir)
        save_prediction(predicted_path, predicted_output, prediction_infidelity;
            description="FL_1step_3p ladder step $(step) predicted (τ_exc,ωd,τ_SWAP) (validation run, basis_id=$(basis_id))",
            params=Dict{Symbol,Any}(:step=>step, :basis_id=>basis_id, :t0=>t0))
        println("step $(step)/$(N_steps): saved predicted parameters to $(predicted_path)")

        # --- (new) plot the predicted trajectory, using this step's pre-advance (t0, initial_state) ---
        τ_exc_pred, ωd_pred, τ_SWAP_pred = predicted_output
        dense_tspan, dense_states = FLstep_dynamics_3p_dense(t0, initial_state, τ_exc_pred, ωd_pred, τ_SWAP_pred)
        plot_path = _trajectory_plot_path(basis_id, step, save_dir)
        plot_trajectory(dense_tspan, dense_states,
            ["n_qubit" => op(cs, :qubit, :n), "n_osc" => op(cs, :osc, :n)];
            title="run_Fock_ladder step $(step)/$(N_steps) — predicted trajectory (basis_id=$(basis_id))",
            ylabel="⟨n⟩", vlines=[t0 + τ_exc_pred], save_path=plot_path)
        println("step $(step)/$(N_steps): saved predicted-trajectory plot to $(plot_path)")

        # (new) fold this step's dense trajectory into the running full-trajectory
        # accumulators — drop the leading point for every step after the first,
        # since it duplicates the previous step's own last (t, state) pair.
        if step == 1
            append!(all_tspan, dense_tspan)
            append!(all_states, dense_states)
        else
            append!(all_tspan, dense_tspan[2:end])
            append!(all_states, dense_states[2:end])
        end
        push!(boundary_times, t0 + τ_exc_pred)          # this step's spin-flip/SWAP split
        push!(boundary_times, t0 + τ_exc_pred + τ_SWAP_pred)  # this step's end / next step's start

        push!(infidelities, prediction_infidelity)
        push!(predictions, predicted_output)
        push!(models, nn)
        push!(final_states, predicted_state)

        # --- 4. Reiterate with the state the predicted parameters actually reach ---
        t0 += τ_exc_pred + τ_SWAP_pred
        initial_state = predicted_state
    end

    # (new) one cumulative plot spanning every step's dense trajectory,
    # concatenated in absolute time — shows whether ⟨n_osc⟩ actually climbs
    # 0→1→2→... across the whole run, which no single per-step plot can show
    # on its own. Always produced, even for N_steps=1 (identical to that
    # step's own per-step plot then, just under its own file name).
    full_plot_path = _full_trajectory_plot_path(basis_id, save_dir)
    plot_trajectory(all_tspan, all_states,
        ["n_qubit" => op(cs, :qubit, :n), "n_osc" => op(cs, :osc, :n)];
        title="run_Fock_ladder — full predicted trajectory (basis_id=$(basis_id), N_steps=$(N_steps))",
        ylabel="⟨n⟩", vlines=boundary_times, save_path=full_plot_path)
    println("saved full predicted-trajectory plot ($(N_steps) step(s)) to $(full_plot_path)")

    return decom_basis, infidelities, predictions, models, final_states
end

# --- Driver: step=1 validation only (per this session's explicit request — ---
# --- step=3 is a separate, later step, only after this one is reviewed)   ---

basis_id_v = 1
save_dir_v = @__DIR__

basis_path_v = _decom_basis_path(basis_id_v, save_dir_v)
isfile(basis_path_v) || generate_decom_basis(basis_id_v; save_dir=save_dir_v)

step1_files_exist = isfile(_dataset_path(basis_id_v, 1, save_dir_v)) &&
                     isfile(_nn_path(basis_id_v, 1, save_dir_v)) &&
                     isfile(_predicted_path(basis_id_v, 1, save_dir_v)) &&
                     isfile(_trajectory_plot_path(basis_id_v, 1, save_dir_v))

if step1_files_exist
    println("Step 1 artifacts already exist for basis_id=$(basis_id_v) (from an earlier, already-reviewed run) — not regenerating.")
else
    decom_basis_v, infidelities_v, predictions_v, models_v, final_states_v =
        run_Fock_ladder(1, basis_id_v;
            n_samples=150, hidden=128, η=1e-3, epochs=60,
            batch_size=32, train_fraction=0.85, save_dir=save_dir_v)

    step1_ok = length(infidelities_v) == 1 &&
               final_states_v[1] isa AbstractOperator &&
               abs(real(tr(final_states_v[1])) - 1.0) < 1e-3 &&
               isfile(_dataset_path(basis_id_v, 1, save_dir_v)) &&
               isfile(_nn_path(basis_id_v, 1, save_dir_v)) &&
               isfile(_predicted_path(basis_id_v, 1, save_dir_v)) &&
               isfile(_trajectory_plot_path(basis_id_v, 1, save_dir_v))

    if step1_ok
        log_path = joinpath(@__DIR__, "test_log.md")
        header_needed = !isfile(log_path)
        open(log_path, "a") do io
            header_needed && println(io, "# Fock-ladder validation log (step (v))\n")
            println(io, "## Step 1 validation — ", Dates.now())
            println(io, "")
            println(io, "- basis_id = $(basis_id_v), save_dir = $(save_dir_v)")
            println(io, "- n_samples = 150, epochs = 60, hidden = 128, η = 1e-3, batch_size = 32, train_fraction = 0.85, uniform sampling")
            println(io, "- dataset: `$(basename(_dataset_path(basis_id_v,1,save_dir_v)))`")
            println(io, "- NN: `$(basename(_nn_path(basis_id_v,1,save_dir_v)))` (test_error = $(models_v[1].test_error))")
            println(io, "- predicted (τ_exc, ωd, τ_SWAP) = $(predictions_v[1])")
            println(io, "- prediction infidelity = $(infidelities_v[1])")
            println(io, "- trajectory plot: `$(basename(_trajectory_plot_path(basis_id_v,1,save_dir_v)))`")
            println(io, "- pipeline check: OK (well-formed trajectory, trace preserved to $(abs(real(tr(final_states_v[1]))-1.0)), all files saved)")
            println(io, "- visual validity: pending user review of the trajectory plot")
            println(io, "")
        end
        println("Step 1 validation succeeded (pipeline check OK) — logged to test_log.md.")
    else
        println("Step 1 validation did NOT succeed mechanically — skipping test_log.md.")
    end
end

# --- Driver: step=2 extension, reusing step 1's already-reviewed dataset+NN ---
#
# Per this session's explicit request: only run after step 1 was reviewed and
# approved. Reuses step 1's saved NN via nn_mode=:fixed rather than
# regenerating it — since prediction is a deterministic forward pass given
# fixed weights and a fixed (t0=0, ground-state) input, step 1's prediction
# here reproduces *exactly* the one already reviewed, not a fresh draw.
# Step 2 generates+trains fresh, same hyperparameters as step 1.

decom_basis_v2, infidelities_v2, predictions_v2, models_v2, final_states_v2 =
    run_Fock_ladder(2, basis_id_v;
        n_samples=150, hidden=128, η=1e-3, epochs=60,
        batch_size=32, train_fraction=0.85, save_dir=save_dir_v,
        nn_mode=[:fixed, :train_new])

step2_ok = length(infidelities_v2) == 2 &&
           all(s -> s isa AbstractOperator, final_states_v2) &&
           all(s -> abs(real(tr(s)) - 1.0) < 1e-3, final_states_v2) &&
           isfile(_dataset_path(basis_id_v, 2, save_dir_v)) &&
           isfile(_nn_path(basis_id_v, 2, save_dir_v)) &&
           isfile(_predicted_path(basis_id_v, 2, save_dir_v)) &&
           isfile(_trajectory_plot_path(basis_id_v, 2, save_dir_v)) &&
           isfile(_full_trajectory_plot_path(basis_id_v, save_dir_v))

if step2_ok
    log_path = joinpath(@__DIR__, "test_log.md")
    open(log_path, "a") do io
        println(io, "## Step 1→2 validation — ", Dates.now())
        println(io, "")
        println(io, "- basis_id = $(basis_id_v), save_dir = $(save_dir_v)")
        println(io, "- step 1 reused via nn_mode=:fixed (loads nn_step1_b1.jld2, no retraining — reproduces the already-reviewed step 1 prediction exactly)")
        println(io, "- step 2: n_samples = 150, epochs = 60, hidden = 128, η = 1e-3, batch_size = 32, train_fraction = 0.85, uniform sampling")
        for step in 1:2
            extra = step == 2 ? ", NN test_error = $(models_v2[step].test_error)" : ""
            println(io, "- step $(step): predicted (τ_exc, ωd, τ_SWAP) = $(predictions_v2[step]), infidelity = $(infidelities_v2[step])$(extra)")
        end
        println(io, "- per-step plots: `$(basename(_trajectory_plot_path(basis_id_v,1,save_dir_v)))`, `$(basename(_trajectory_plot_path(basis_id_v,2,save_dir_v)))`")
        println(io, "- cumulative plot: `$(basename(_full_trajectory_plot_path(basis_id_v,save_dir_v)))`")
        println(io, "- pipeline check: OK (well-formed trajectories, trace preserved, all files saved)")
        println(io, "- visual validity: pending user review of the cumulative trajectory plot")
        println(io, "")
    end
    println("Step 1→2 validation succeeded (pipeline check OK) — logged to test_log.md.")
    println("Review $(basename(_full_trajectory_plot_path(basis_id_v,save_dir_v))) for the full picture.")
else
    println("Step 1→2 validation did NOT succeed mechanically — skipping test_log.md.")
end
