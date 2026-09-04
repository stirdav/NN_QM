# FockLadder_execution.jl
#
# (Renamed from execution.jl — see CLAUDE.md / DESIGN.md for the rename note.)
#
# This file is deliberately thin (DESIGN.md Part 17/18 — the project's
# standing three-way rule): NNQuantum.jl holds the generic ML/dataset-
# management engine; FockLadder_problem.jl holds the quantum problem's own
# definition (dynamics functions) *and* its dataset input/output creation
# (FL_1step_3p_NN_outputs/_inputs) and NN-facing glue (predict_drive_
# parameters) — none of that belongs here. This file only:
#   1. Activates the environment and includes FockLadder_problem.jl (which
#      itself includes Definition.jl and NNQuantum.jl — see that file's own
#      header for why it depends on NNQuantum.jl directly).
#   2. Defines run_Fock_ladder(N_steps, basis_id; ...) and
#      generate_decom_basis — the orchestrating "run" functions that call
#      into FockLadder_problem.jl (for settings, dynamics,
#      FL_1step_3p_NN_outputs/_inputs, predict_drive_parameters) and
#      NNQuantum.jl (for train_NN, save/load of datasets/NNs/bases/
#      predictions, plot_trajectory) — never holding either kind of logic
#      itself.
#
#   This is the version staged-validated in test.jl (N_steps=1 then 2, see
#   DESIGN.md Parts 11-13 and test_log.md) and promoted here once reviewed —
#   test.jl itself now only holds the smaller step-1/2 smoke tests it started
#   with.
#
# Run from NNQuantum/ (REPL: include("FockLadder_execution.jl"), or
# `julia FockLadder_execution.jl`). Including this file only activates the
# environment and defines run_Fock_ladder — it does not call it
# (dataset generation at realistic n_samples is expensive; see the bottom
# of this file for how to run it).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using QuantumDynamics
using QuantumOptics

include("Definition.jl")
include("FockLadder_problem.jl")   # (renamed from HBAR-qubit_problem.jl) — this in turn
                                     # includes NNQuantum.jl, so both are available here
                                     # transitively; no separate include needed.

# FockLadder_problem.jl's (a)-(g) are the *setup* of the HBAR+qubit system
# (parameters, subsystems, Hamiltonian, dissipators, protocol runner), plus
# its own FL_1step_3p_NN_outputs/_inputs (dataset creation) and
# predict_drive_parameters (NN-facing glue) — all live code, exercised by
# this include. Step 2's actual validation run (physically-tuned pulse
# parameters, plot ⟨n_qubit⟩/⟨n_mech⟩) lives in test.jl, not here — see
# CLAUDE.md's Scope note on test.jl's role. run_Fock_ladder below is what
# drives the dataset/NN pieces across a full ladder, generic-N_steps.

qubit_basis = getsubsystem(cs, :qubit).basis
osc_basis   = getsubsystem(cs, :osc).basis

# --- Ladder-step dataset generation -------------------------------------------
#
# Mirrors Chu_DFL_execution.ipynb's execution_dynamic_FL (the
# :FL_1step_3p / :master_dynamic cell, id dfd162e7) structurally: one
# dataset per step, target states built the way creation_step_states(ρ0,
# step) builds them, parameter ranges built the way that cell builds them
# (T_exc fixed across steps; τ_SWAP centered on t_swap_th=π/(2g√step), so it
# narrows as the ladder climbs and the JC swap gets faster). decom_basis is
# built once, before the loop, and reused every step — same as that cell's
# own SUN_basis.

# A step's dataset_mode/nn_mode (and their companion path arguments) may be
# given once (applies to every step) or as a Vector with one entry per step
# — _per_step resolves either shape the same way; _check_per_step_length
# fails loudly and early (before any simulation runs) on a length mismatch,
# rather than an obscure BoundsError mid-loop.
_per_step(x, step) = x isa AbstractVector ? x[step] : x

function _check_per_step_length(x, N_steps, name)
    if x isa AbstractVector && length(x) != N_steps
        throw(ArgumentError("run_Fock_ladder: $name has length $(length(x)), expected $N_steps (one entry per step)"))
    end
end

# File names for one basis_id: every file that belongs together (the basis
# itself, and every dataset/NN made using it) ends in "_b<basis_id>", so the
# number in the name is what tells you which basis a file goes with —
# generate_decom_basis (NNQuantum.jl, DESIGN.md Part 18 — relocated and
# generalized from a local copy that used to live here) is what makes that
# file, called explicitly, e.g. `generate_decom_basis(cs, 1)`.
_decom_basis_path(basis_id, save_dir) = joinpath(save_dir, decom_basis_filename(basis_id))
_dataset_path(basis_id, step, save_dir) = joinpath(save_dir, "dataset_step$(step)_b$(basis_id).jld2")
_nn_path(basis_id, step, save_dir) = joinpath(save_dir, "nn_step$(step)_b$(basis_id).jld2")
_predicted_path(basis_id, step, save_dir) = joinpath(save_dir, "predicted_step$(step)_b$(basis_id).jld2")
_trajectory_plot_path(basis_id, step, save_dir) = joinpath(save_dir, "trajectory_step$(step)_b$(basis_id).png")
_full_trajectory_plot_path(basis_id, save_dir) = joinpath(save_dir, "trajectory_full_b$(basis_id).png")

"""
    run_Fock_ladder(N_steps, basis_id; n_samples=800, dim_parameters_space=[100,100,100],
                    save_dir=@__DIR__, train_fraction=0.9375, hidden=500, η=1e-4,
                    epochs=350, loss=:mse, batch_size=32, uniform=true,
                    dataset_mode=:generate_and_save, dataset_paths=nothing,
                    nn_mode=:train_new, nn_paths=nothing, save_nn_dir=save_dir)

Drive the Fock ladder up `N_steps` rungs following the user's train/predict/
reiterate algorithm (see CLAUDE.md's Plan, step 3), not `old_version`'s
`execution_dynamic_FL` directly (structurally similar, but restructured
around this project's own step numbering):

  0. Start from the true ground state (`spindown(qubit) ⊗ fockstate(osc,0)`).
  1. Generate a dataset from the current `(t0, initial_state)` over a
     spectrum of drive values (`FL_1step_3p_NN_outputs`/`_inputs`), and save
     it.
  2. Train a NN on that dataset (state features → drive parameters).
  3. Use the trained NN to predict the drive parameters that reach this
     step's target state.
  4. Reiterate steps 1-3 with the state the predicted drive parameters
     *actually* reach (not the target state itself) as next step's initial
     condition, until `N_steps` is reached.

**Status: all five steps (0-4) are implemented — DESIGN.md Part 6 (iv), settled
per Parts 17-18's three-file rule.** Steps 2-3 are `train_NN` (`NNQuantum.jl`
— fully generic, no problem-specific wrapper needed) and
`predict_drive_parameters` (`FockLadder_problem.jl` — the one piece of
NN-facing glue that is genuinely `:FL_1step_3p`-specific) around
`NNQuantum.jl`'s generic `train_and_test_NN`/`predict_and_score`; the keyword arguments here
(`train_fraction`/`hidden`/`η`/`epochs`/`loss`/`batch_size`) just pass
through to those, defaulting to `Chu_DFL_execution.ipynb`'s own
`:FL_1step_3p`/`:master_dynamic` cell values where that cell has an
equivalent (`batch_size` doesn't — see `NNQuantum.jl`'s own note on
`train_model!` for why mini-batching was added on top of the port). Step 4
no longer needs its own re-simulation call: `predict_drive_parameters`
already returns the state its predicted pulse actually reaches
(`predict_and_score`'s `final_state`, scored against `target_final` by
infidelity in the same call), so that's what's carried forward as next
step's `initial_state` directly — no second `FLstep_dynamics_3p` call over
the same parameters. The earlier placeholder that advanced using each
step's *target* state and nominal pulse timing (assuming the ladder climbs
perfectly) was removed, not fixed, once this real prediction existed to
replace it, per the restructure documented further up this file's own
history in `DESIGN.md`.

`uniform` (default `true`) controls how step 1's `(τ_exc,ωd,τ_SWAP)` triples
are drawn: this function's own sampling weight, `prs`, is a hardcodedbar (same place as in your 
constant (`1/prod(dim_parameters_space)`, independent of the point) — there
is no live path here where the sampling is actually non-uniform — so the
default skips `FL_1step_3p_NN_outputs`'s dense-grid-then-weighted-draw
entirely in favor of direct continuous sampling (`NNQuantum.jl`'s
`uniform_parameter_sample`), which is both cheaper (`O(n_samples)` instead of
`O(∏dim_parameters_space)`) and finer-grained (continuous, not quantized to
`dim_parameters_space`'s resolution). `dim_parameters_space` stays a real,
effective keyword when `uniform=false` is passed explicitly, reverting to the
original grid-based sampling.

**`basis_id` — always a loaded basis, never a made-up one.** `decom_basis`
(the random set of operators every feature vector, training or prediction, is
built from) is no longer made inside `run_Fock_ladder` at all — it is always
loaded from `save_dir/decom_basis_b<basis_id>.jld2`, made ahead of time by
calling `generate_decom_basis(cs, basis_id)` once (`NNQuantum.jl` — DESIGN.md
Part 18). If that file doesn't exist, `run_Fock_ladder` stops with a clear
error telling you to call `generate_decom_basis` first, instead of silently
making a new one. See `generate_decom_basis`'s own docstring for why this is a separate step now —
a dataset or NN only means what it means together with the exact basis it
was built from, and a real bug came from making a fresh, unsaved basis
automatically on every call (`DESIGN.md` Part 8).

Every file this run saves gets `basis_id` at the end of its name, so the
number itself shows which basis it goes with: `decom_basis_b<id>.jld2`
(made once, by `generate_decom_basis`), `dataset_step<n>_b<id>.jld2`,
`nn_step<n>_b<id>.jld2`.

**Dataset/NN reuse** (`DESIGN.md`'s later addition — generate/train fresh
every step was the only option before this): `dataset_mode` and `nn_mode`
are independent per-step controls, each either a single `Symbol` (applies to
every step) or a `Vector{Symbol}` (one per step, checked against `N_steps`).

- `dataset_mode ∈ (:generate_and_save, :generate_only, :load)` — the default
  generates and saves (as before, now named `dataset_step<n>_b<id>.jld2`);
  `:generate_only` skips the save; `:load` skips generation entirely and
  reads that same file name back in (via `load_dataset`) instead —
  `dataset_paths[step]`/`dataset_paths` can be given to load from a
  different file instead, but usually isn't needed.
- `nn_mode ∈ (:train_new, :continue_training, :fixed)` — the default trains
  a fresh model each step; `:continue_training` loads that step's saved
  checkpoint (`nn_step<n>_b<id>.jld2`, or `nn_paths[step]`/`nn_paths` if
  given) and resumes training on the current step's dataset; `:fixed` loads
  it and uses it directly for prediction, with **no training and no dataset
  step at all** — nothing would consume a generated dataset in that case,
  so it's skipped rather than generated and discarded.
- `save_nn_dir` (defaults to `save_dir`) is where each step's resulting NN
  is saved whenever training actually happens
  (`:train_new`/`:continue_training`) — this is what supplies the
  checkpoints `:continue_training`/`:fixed` load in a later step or a later
  run. Pass `nothing` to skip saving.

**Also saves each step's prediction and plots its trajectory.** Every step's
predicted `(τ_exc,ωd,τ_SWAP)` and its infidelity are saved via
`save_prediction` to `predicted_step<n>_b<id>.jld2`; that step's own
predicted trajectory (⟨n_qubit⟩/⟨n_osc⟩, via `FLstep_dynamics_3p_dense` +
`plot_trajectory`) is plotted to `trajectory_step<n>_b<id>.png`; and, once
every step is done, one cumulative plot spanning the whole run (every step's
dense trajectory concatenated in absolute time, so a multi-step run's
`⟨n_osc⟩` climb is visible in one picture, not just isolated per-step
windows) is saved to `trajectory_full_b<id>.png`.

**Returns** `(decom_basis, infidelities, predictions, models, final_states)`
— matching `Chu_DFL_execution.ipynb`'s own `execution_dynamic_FL` return
shape (`return SUN_basis, infidelities, predictions, models, final_states`),
one entry per step for the last four. `models` holds this project's own
richer per-step bundle (`train_NN`'s/`load_nn`'s `NamedTuple` — model plus
normalization stats), not `old_version`'s bare `Flux.Chain`.

This function was validated end-to-end (staged, `N_steps=1` then `N_steps=2`,
at small/fast hyperparameters) before being promoted here from a `test.jl`
validation copy — see `DESIGN.md` Parts 11-13 and `test_log.md` for the run
record. Running it at real scale (`old_version`'s own `n_samples=800`,
`epochs=350`) is left to the user, not yet done.
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

    # decom_basis is always loaded, never made here — see generate_decom_basis's
    # own docstring for why. Every feature vector below (training or
    # prediction) is only meaningful together with this exact basis, so a
    # missing file is a loud, upfront error, not a silent fresh basis.
    basis_path = _decom_basis_path(basis_id, save_dir)
    isfile(basis_path) || throw(ArgumentError(
        "run_Fock_ladder: no saved basis at $basis_path — call " *
        "generate_decom_basis($basis_id; save_dir=\"$save_dir\") first"))
    decom_basis = load_operator_basis(basis_path, cs.basis)

    prs(τ_exc, ωd, τ_SWAP) = 1 / prod(dim_parameters_space)

    t0 = 0.0
    # --- 0. True ground state: the starting initial condition ---
    initial_state = spindown(qubit_basis) ⊗ fockstate(osc_basis, 0)

    infidelities = Float64[]
    predictions = Any[]
    models = Any[]
    final_states = Any[]

    # Accumulators for the one cumulative full-run plot, spanning every
    # step's dense trajectory concatenated in absolute time — as opposed to
    # each step's own per-step plot below, which only ever shows that step's
    # own window. Each step's dense_tspan/dense_states already sits at the
    # right absolute time (t0 carries forward correctly across steps, never
    # reset), so concatenation needs no time-shifting — only dropping each
    # step-after-the-first's leading point, which duplicates the previous
    # step's very last (t, state) pair (same convention FLstep_dynamics_3p/
    # FLstep_dynamics_3p_dense already use to join their own two stages).
    all_tspan = Float64[]
    all_states = Any[]
    boundary_times = Float64[]   # every spin-flip/SWAP split and step-to-step transition, for vlines

    for step in 1:N_steps
        # --- Target states for this step (mirrors creation_step_states) ---
        target_spinflip = spinup(qubit_basis) ⊗ fockstate(osc_basis, step - 1)
        target_final    = spindown(qubit_basis) ⊗ fockstate(osc_basis, step)

        step_nn_mode = _per_step(nn_mode, step)

        if step_nn_mode == :fixed
            # --- :fixed — no dataset, no training, just load and predict ---
            # (loading always looks under save_dir, the one directory that's
            # never nothing — save_nn_dir only controls where *new* checkpoints
            # get written, a separate, optional concern, below)
            nn_path = something(_per_step(nn_paths, step), _nn_path(basis_id, step, save_dir))
            nn = load_nn(nn_path)
            println("step $(step)/$(N_steps): loaded fixed NN from $(nn_path), no dataset/training")

        elseif step_nn_mode ∈ (:train_new, :continue_training)
            # --- Parameter ranges for this step ---
            T_exc_ini, T_exc_fin = 1e-5, 5e-2

            δ_ωd = 100.0
            ωd_ini, ωd_fin = Δ0 - δ_ωd, Δ0 + δ_ωd

            t_swap_th = π / (2 * g * sqrt(step))
            δ1, δ2 = 0.0005, 0.00015
            τ_SWAP_ini, τ_SWAP_fin = t_swap_th - δ1, t_swap_th + δ2

            parameters_range = [[T_exc_ini, T_exc_fin], [ωd_ini, ωd_fin], [τ_SWAP_ini, τ_SWAP_fin]]

            # --- 1. Dataset: generate (+ optionally save), or load a previous one ---
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
                        description="FL_1step_3p ladder step $(step) dataset (N_steps=$(N_steps), basis_id=$(basis_id))",
                        params=Dict{Symbol,Any}(:step=>step, :basis_id=>basis_id, :n_samples=>n_samples,
                                                 :N_mech=>N_mech, :g=>g, :t0=>t0))
                    println("step $(step)/$(N_steps): saved $(n_samples)-sample dataset to $(save_path)")
                else
                    println("step $(step)/$(N_steps): generated $(n_samples)-sample dataset (not saved)")
                end
            else
                throw(ArgumentError("run_Fock_ladder: unknown dataset_mode $(step_dataset_mode) at step $(step)"))
            end

            # --- 2. Train — fresh, or continuing from a loaded checkpoint ---
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
                save_nn(save_path, nn; description="FL_1step_3p ladder step $(step) trained NN (basis_id=$(basis_id))",
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

        # --- Save the prediction ---
        predicted_path = _predicted_path(basis_id, step, save_dir)
        save_prediction(predicted_path, predicted_output, prediction_infidelity;
            description="FL_1step_3p ladder step $(step) predicted (τ_exc,ωd,τ_SWAP) (basis_id=$(basis_id))",
            params=Dict{Symbol,Any}(:step=>step, :basis_id=>basis_id, :t0=>t0))
        println("step $(step)/$(N_steps): saved predicted parameters to $(predicted_path)")

        # --- Plot this step's predicted trajectory, using its pre-advance (t0, initial_state) ---
        τ_exc_pred, ωd_pred, τ_SWAP_pred = predicted_output
        dense_tspan, dense_states = FLstep_dynamics_3p_dense(t0, initial_state, τ_exc_pred, ωd_pred, τ_SWAP_pred)
        plot_path = _trajectory_plot_path(basis_id, step, save_dir)
        plot_trajectory(dense_tspan, dense_states,
            ["n_qubit" => op(cs, :qubit, :n), "n_osc" => op(cs, :osc, :n)];
            title="run_Fock_ladder step $(step)/$(N_steps) — predicted trajectory (basis_id=$(basis_id))",
            ylabel="⟨n⟩", vlines=[t0 + τ_exc_pred], save_path=plot_path)
        println("step $(step)/$(N_steps): saved predicted-trajectory plot to $(plot_path)")

        # Fold this step's dense trajectory into the running full-trajectory
        # accumulators — drop the leading point for every step after the
        # first, since it duplicates the previous step's own last (t, state) pair.
        if step == 1
            append!(all_tspan, dense_tspan)
            append!(all_states, dense_states)
        else
            append!(all_tspan, dense_tspan[2:end])
            append!(all_states, dense_states[2:end])
        end
        push!(boundary_times, t0 + τ_exc_pred)                 # this step's spin-flip/SWAP split
        push!(boundary_times, t0 + τ_exc_pred + τ_SWAP_pred)   # this step's end / next step's start

        push!(infidelities, prediction_infidelity)
        push!(predictions, predicted_output)
        push!(models, nn)
        push!(final_states, predicted_state)

        # --- 4. Reiterate with the state the predicted parameters actually reach ---
        t0 += τ_exc_pred + τ_SWAP_pred
        initial_state = predicted_state
    end

    # One cumulative plot spanning every step's dense trajectory, concatenated
    # in absolute time — shows whether ⟨n_osc⟩ actually climbs 0→1→2→...
    # across the whole run, which no single per-step plot can show on its
    # own. Always produced, even for N_steps=1 (identical to that step's own
    # per-step plot then, just under its own file name).
    full_plot_path = _full_trajectory_plot_path(basis_id, save_dir)
    plot_trajectory(all_tspan, all_states,
        ["n_qubit" => op(cs, :qubit, :n), "n_osc" => op(cs, :osc, :n)];
        title="run_Fock_ladder — full predicted trajectory (basis_id=$(basis_id), N_steps=$(N_steps))",
        ylabel="⟨n⟩", vlines=boundary_times, save_path=full_plot_path)
    println("saved full predicted-trajectory plot ($(N_steps) step(s)) to $(full_plot_path)")

    return decom_basis, infidelities, predictions, models, final_states
end

# Not called on include — dataset generation at realistic n_samples takes
# real time (minutes, at old_version's own n_samples=800/dim_parameters_
# space=[100,100,100], per step). Run explicitly, e.g. from the REPL after
# including this file. A basis has to exist first (generate_decom_basis
# only needs doing once — reuse the same basis_id afterward):
#
#   generate_decom_basis(cs, 1)              # makes decom_basis_b1.jld2
#   run_Fock_ladder(6, 1)                    # old_version's own N_steps, full-size
#   run_Fock_ladder(3, 1; n_samples=20)      # a cheaper multi-step check
