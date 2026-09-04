# Bspline_Fock_execution.jl
#
# Thin orchestrator for the `:FL_1step_Bspline` protocol (DESIGN.md Part
# 19), the counterpart of `FockLadder_execution.jl` for
# `Bspline_Fock_problem.jl`. Follows the project's standing three-way rule
# (DESIGN.md Part 18, CLAUDE.md's Scope note) exactly like
# `FockLadder_execution.jl` does: only environment activation +
# `include("Bspline_Fock_problem.jl")` (which itself includes
# `Definition.jl` and `NNQuantum.jl`), plus `run_FockTarget_Bspline` — the
# orchestrating "run" function, calling into `Bspline_Fock_problem.jl` for
# settings/dynamics/dataset-creation/prediction-glue and `NNQuantum.jl` for
# ML/dataset-management. No dataset-creation or NN-wrapper logic of its own.
#
# `run_FockTarget_Bspline` is deliberately NOT `run_Fock_ladder` with a
# `:FL_1step_Bspline` dynamics function swapped in — it has no `N_steps`,
# no reiteration loop, no per-step target advancing. DESIGN.md Part 19
# explains why: `:FL_1step_3p`'s rigid, 3-parameter pulse can only swap one
# quantum per call, so climbing to Fock state `n` needs `n` reiterated
# one-rung calls, each starting from wherever the previous (imperfect) call
# actually landed. The B-spline drive's `2×n_basis` free coefficients over a
# window `[0,T]` are expressive enough to be aimed directly at a target Fock
# number in one shot, from the ground state, in the same spirit as
# optimal-control Fock-state-preparation techniques in circuit QED — so
# there's no structural reason to re-impose a ladder here. One target Fock
# number, one dataset, one trained NN, one prediction.
#
# Run from NNQuantum/examples/ (REPL: include("Bspline_Fock_execution.jl"), or
# `julia Bspline_Fock_execution.jl`). Including this file only activates the
# environment and defines run_FockTarget_Bspline — it does not call it (see
# the bottom of this file for how to run it).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using QuantumDynamics
using QuantumOptics

include("Bspline_Fock_problem.jl")   # includes Definition.jl and NNQuantum.jl transitively

qubit_res_basis = getsubsystem(cs_res, :qubit).basis
osc_res_basis   = getsubsystem(cs_res, :osc).basis

# --- File names for one basis_id/target -----------------------------------------
#
# Same "basis_id at the end of every related file name" convention
# FockLadder_execution.jl uses, but with a "_bspline" segment throughout
# (decom_basis_filename's own `prefix` keyword, NNQuantum.jl) so this
# variant's files never collide with :FL_1step_3p's own in a shared
# save_dir — genuinely necessary here, not just tidy: the two variants'
# composite dimensions differ (d=8 vs. d=12), and DESIGN.md Part 8's own bug
# (a dataset/NN silently paired with the wrong basis) is exactly the failure
# mode distinct names guard against. "step" becomes "target" throughout,
# since there is no per-step loop here — just one target Fock number.
_decom_basis_path_bspline(basis_id, save_dir) =
    joinpath(save_dir, decom_basis_filename(basis_id; prefix="decom_basis_bspline"))
_dataset_path_bspline(basis_id, n_target, save_dir) =
    joinpath(save_dir, "dataset_bspline_target$(n_target)_b$(basis_id).jld2")
_nn_path_bspline(basis_id, n_target, save_dir) =
    joinpath(save_dir, "nn_bspline_target$(n_target)_b$(basis_id).jld2")
_predicted_path_bspline(basis_id, n_target, save_dir) =
    joinpath(save_dir, "predicted_bspline_target$(n_target)_b$(basis_id).jld2")
_trajectory_plot_path_bspline(basis_id, n_target, save_dir) =
    joinpath(save_dir, "trajectory_bspline_target$(n_target)_b$(basis_id).png")

"""
    run_FockTarget_Bspline(n_target, basis_id; n_samples=800, T_max=0.03,
                            coeff_range=(-50.0,50.0), n_basis=10, degree=4,
                            save_dir=@__DIR__, train_fraction=0.9375, hidden=500, η=1e-4,
                            epochs=350, loss=:mse, batch_size=32, n_T_candidates=10,
                            dataset_mode=:generate_and_save, dataset_path=nothing,
                            nn_mode=:train_new, nn_path=nothing, save_nn_dir=save_dir)

Reach Fock state `n_target` directly from the ground state, in one shot, via
the `:FL_1step_Bspline` protocol — no ladder, no reiteration (see this
file's own header note for why):

  1. Generate a dataset from the true ground state, sweeping sampled
     `(T, coeffs_Re, coeffs_Im)` triples (`FL_1step_Bspline_NN_outputs`/
     `_inputs`, `Bspline_Fock_problem.jl`), and save it.
  2. Train a NN on that dataset (state features ⊕ infidelity ⊕ T →
     B-spline coefficients).
  3. Use the trained NN to predict the coefficients that reach
     `target_final = spindown(qubit_res) ⊗ fockstate(osc_res, n_target)`,
     sweeping `n_T_candidates` values of `T` evenly across `[0,T_max]` and
     keeping whichever achieves the best re-simulated infidelity
     (`predict_drive_parameters_Bspline`).

`basis_id` — always a loaded basis, never a made-up one, same reasoning as
`run_Fock_ladder`'s own (DESIGN.md Part 8): call
`generate_decom_basis(cs_res, basis_id; save_dir=save_dir, prefix="decom_basis_bspline")`
once, ahead of time. If that file doesn't exist, this function stops with a
clear error rather than silently making a new one.

**Dataset/NN reuse**, same shape as `run_Fock_ladder`'s own (`DESIGN.md`),
just without the per-step `Vector` option — there is only one target here:
- `dataset_mode ∈ (:generate_and_save, :generate_only, :load)`.
- `nn_mode ∈ (:train_new, :continue_training, :fixed)` — `:fixed` skips
  dataset generation entirely, same reasoning as `run_Fock_ladder`'s own
  (nothing would consume a generated dataset).
- `save_nn_dir` (defaults to `save_dir`) is where a freshly trained/
  continued NN gets saved; pass `nothing` to skip saving.

**Also saves the prediction and plots its trajectory**, same as
`run_Fock_ladder`'s own per-step behavior: `predicted_bspline_target<n>_b
<id>.jld2` (predicted coefficients + infidelity, with the winning `T`
recorded in its `params`) and `trajectory_bspline_target<n>_b<id>.png`
(`FLstep_dynamics_Bspline_dense` + `plot_trajectory`).

**Returns** `(decom_basis, infidelity, predicted_output, nn, final_state)`
— a flat tuple, not one-entry-per-step vectors like `run_Fock_ladder`
returns, since there is no step dimension here.
"""
function run_FockTarget_Bspline(n_target::Integer, basis_id::Integer;
                                 n_samples::Integer=3000,
                                 T_max::Real=0.04,
                                 coeff_range::Tuple{<:Real,<:Real}=(-300.0, 300.0),
                                 n_basis::Integer=30,
                                 degree::Integer=6,
                                 save_dir::AbstractString=@__DIR__,
                                 train_fraction::Real=0.9375,
                                 hidden::Integer=500,
                                 η::Real=1e-4,
                                 epochs::Integer=350,
                                 loss::Union{Function,Symbol}=:mse,
                                 batch_size::Integer=32,
                                 n_T_candidates::Integer=10,
                                 dataset_mode::Symbol=:generate_and_save,
                                 dataset_path::Union{Nothing,AbstractString}=nothing,
                                 nn_mode::Symbol=:train_new,
                                 nn_path::Union{Nothing,AbstractString}=nothing,
                                 save_nn_dir::Union{Nothing,AbstractString}=save_dir)
    basis_path = _decom_basis_path_bspline(basis_id, save_dir)
    isfile(basis_path) || throw(ArgumentError(
        "run_FockTarget_Bspline: no saved basis at $basis_path — call " *
        "generate_decom_basis(cs_res, $basis_id; save_dir=\"$save_dir\", prefix=\"decom_basis_bspline\") first"))
    decom_basis = load_operator_basis(basis_path, cs_res.basis)

    basis_spline = generate_Bspline_basis(degree, n_basis, (0.0, 1.0))

    t0 = 0.0
    initial_state = spindown(qubit_res_basis) ⊗ fockstate(osc_res_basis, 0)
    target_final  = spindown(qubit_res_basis) ⊗ fockstate(osc_res_basis, n_target)

    if nn_mode == :fixed
        # --- :fixed — no dataset, no training, just load and predict ---
        resolved_nn_path = something(nn_path, _nn_path_bspline(basis_id, n_target, save_dir))
        nn = load_nn(resolved_nn_path)
        println("target=$(n_target): loaded fixed NN from $(resolved_nn_path), no dataset/training")

    elseif nn_mode ∈ (:train_new, :continue_training)
        # --- 1. Dataset: generate (+ optionally save), or load a previous one ---
        if dataset_mode == :load
            resolved_dataset_path = something(dataset_path, _dataset_path_bspline(basis_id, n_target, save_dir))
            X, Y = load_dataset(resolved_dataset_path)
            println("target=$(n_target): loaded dataset from $(resolved_dataset_path)")
        elseif dataset_mode ∈ (:generate_and_save, :generate_only)
            pulse_parameters, outputs = FL_1step_Bspline_NN_outputs(T_max, coeff_range, n_basis, n_samples)
            inputs = FL_1step_Bspline_NN_inputs(t0, initial_state, target_final, decom_basis, basis_spline, pulse_parameters, n_samples)
            X, Y = inputs, outputs

            if dataset_mode == :generate_and_save
                save_path = _dataset_path_bspline(basis_id, n_target, save_dir)
                save_dataset(save_path, inputs, outputs;
                    description="FL_1step_Bspline target=$(n_target) dataset (basis_id=$(basis_id))",
                    params=Dict{Symbol,Any}(:n_target=>n_target, :basis_id=>basis_id, :n_samples=>n_samples,
                                             :N_mech_Bspline=>N_mech_Bspline, :g=>g, :T_max=>T_max,
                                             :coeff_range=>coeff_range, :n_basis=>n_basis, :t0=>t0))
                println("target=$(n_target): saved $(n_samples)-sample dataset to $(save_path)")
            else
                println("target=$(n_target): generated $(n_samples)-sample dataset (not saved)")
            end
        else
            throw(ArgumentError("run_FockTarget_Bspline: unknown dataset_mode $(dataset_mode)"))
        end

        # --- 2. Train — fresh, or continuing from a loaded checkpoint ---
        if nn_mode == :continue_training
            resolved_nn_path = something(nn_path, _nn_path_bspline(basis_id, n_target, save_dir))
            loaded = load_nn(resolved_nn_path)
            nn = train_NN(X, Y; train_fraction=train_fraction, hidden=hidden, η=η, epochs=epochs,
                          loss=loss, batch_size=batch_size, model=loaded.model, opt_state=loaded.opt_state)
        else
            nn = train_NN(X, Y; train_fraction=train_fraction, hidden=hidden, η=η,
                          epochs=epochs, loss=loss, batch_size=batch_size)
        end

        if save_nn_dir !== nothing
            save_path = _nn_path_bspline(basis_id, n_target, save_nn_dir)
            save_nn(save_path, nn; description="FL_1step_Bspline target=$(n_target) trained NN (basis_id=$(basis_id))",
                    params=Dict{Symbol,Any}(:n_target=>n_target, :basis_id=>basis_id, :N_mech_Bspline=>N_mech_Bspline, :g=>g))
            println("target=$(n_target): saved trained NN to $(save_path) (test error = $(nn.test_error))")
        end
    else
        throw(ArgumentError("run_FockTarget_Bspline: unknown nn_mode $(nn_mode)"))
    end

    # --- 3. Predict the (T, coefficients) that reach target_final ---
    result = predict_drive_parameters_Bspline(nn, t0, initial_state, target_final, decom_basis, basis_spline, T_max;
                                               n_candidates=n_T_candidates)
    println("target=$(n_target): predicted T=$(result.T), infidelity = $(result.infidelity)")

    # --- Save the prediction (T recorded in params, not mixed into predicted_output) ---
    predicted_path = _predicted_path_bspline(basis_id, n_target, save_dir)
    save_prediction(predicted_path, result.predicted_output, result.infidelity;
        description="FL_1step_Bspline target=$(n_target) predicted B-spline coefficients (basis_id=$(basis_id))",
        params=Dict{Symbol,Any}(:n_target=>n_target, :basis_id=>basis_id, :t0=>t0, :T=>result.T))
    println("target=$(n_target): saved predicted parameters to $(predicted_path)")

    # --- Plot the predicted trajectory ---
    n_basis_actual = length(result.predicted_output) ÷ 2
    coeffs_Re = result.predicted_output[1:n_basis_actual]
    coeffs_Im = result.predicted_output[n_basis_actual+1:end]
    dense_tspan, dense_states = FLstep_dynamics_Bspline_dense(t0, initial_state, result.T, coeffs_Re, coeffs_Im, basis_spline)
    plot_path = _trajectory_plot_path_bspline(basis_id, n_target, save_dir)
    plot_trajectory(dense_tspan, dense_states,
        ["n_qubit" => op(cs_res, :qubit, :n), "n_osc" => op(cs_res, :osc, :n)];
        title="run_FockTarget_Bspline target=$(n_target) — predicted trajectory (basis_id=$(basis_id), T=$(round(result.T, digits=5)))",
        ylabel="⟨n⟩", save_path=plot_path)
    println("target=$(n_target): saved predicted-trajectory plot to $(plot_path)")

    return decom_basis, result.infidelity, result.predicted_output, nn, result.final_state
end

# Not called on include — dataset generation at realistic n_samples takes
# real time. Run explicitly, e.g. from the REPL after including this file.
# A basis has to exist first (only needs doing once — reuse the same
# basis_id afterward):
#
#   generate_decom_basis(cs_res, 1; prefix="decom_basis_bspline")   # makes decom_basis_bspline_b1.jld2
#   run_FockTarget_Bspline(1, 1)                                    # reach Fock state 1, full-size
#   run_FockTarget_Bspline(1, 1; n_samples=20, epochs=10)           # a cheaper smoke-test-scale check
