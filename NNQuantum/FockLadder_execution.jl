# FockLadder_execution.jl
#
# (Renamed from execution.jl — see CLAUDE.md / DESIGN.md for the rename note.)
#
# Two things live here:
#   1. Environment activation + including the problem definition
#      (FockLadder_problem.jl) — step 1's scope, unchanged from before the
#      rename.
#   2. run_Fock_ladder(N_steps; ...) — a generic-N_steps loop
#      structured around the user's own generate/train/predict/reiterate
#      algorithm (CLAUDE.md's Plan, step 3), not old_version's
#      execution_dynamic_FL directly. Per step: generates and saves one
#      FL_1step_3p dataset, trains a NN on it and predicts drive parameters
#      for the target state (FockLadder_problem.jl's train_NN/
#      predict_drive_parameters, thin wrappers around NNQuantum.jl's
#      problem-agnostic machinery), and reiterates using the state the
#      prediction actually reaches — see run_Fock_ladder's own docstring.
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
include("FockLadder_problem.jl")   # (renamed from HBAR-qubit_problem.jl)

# FockLadder_problem.jl's (a)-(g) are the *setup* of the HBAR+qubit system
# (parameters, subsystems, Hamiltonian, dissipators, protocol runner) — all
# live code, exercised by this include. Step 2's actual validation run
# (physically-tuned pulse parameters, plot ⟨n_qubit⟩/⟨n_mech⟩) lives in
# test.jl, not here — see CLAUDE.md's Scope note on test.jl's role. (h)'s
# dataset-generation functions (FL_1step_3p_NN_outputs/_inputs,
# save_dataset) are also live from this include — run_Fock_ladder
# below is what drives them across a full ladder, generic-N_steps.

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

"""
    run_Fock_ladder(N_steps; n_samples=800, dim_parameters_space=[100,100,100],
                    save_dir=@__DIR__, train_fraction=0.9375, hidden=500, η=1e-4,
                    epochs=350, loss=:mse, batch_size=32)

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

**Status: all five steps (0-4) are implemented — DESIGN.md Part 6 (iv).**
Steps 2-3 are `train_NN`/`predict_drive_parameters` (`FockLadder_problem.jl`),
thin `:FL_1step_3p`-specific wrappers around `NNQuantum.jl`'s generic
`train_and_test_NN`/`predict_and_score`; the keyword arguments here
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
"""
function run_Fock_ladder(N_steps::Integer;
                                    n_samples::Integer=800,
                                    dim_parameters_space::Vector{<:Integer}=[100, 100, 100],
                                    save_dir::AbstractString=@__DIR__,
                                    train_fraction::Real=0.9375,
                                    hidden::Integer=500,
                                    η::Real=1e-4,
                                    epochs::Integer=350,
                                    loss::Union{Function,Symbol}=:mse,
                                    batch_size::Integer=32)
    d_dataset = length(cs.basis)
    decom_basis = rand_hermitian_orthonormal_basis(d_dataset, cs.basis)

    prs(τ_exc, ωd, τ_SWAP) = 1 / prod(dim_parameters_space)

    t0 = 0.0
    # --- 0. True ground state: the starting initial condition ---
    initial_state = spindown(qubit_basis) ⊗ fockstate(osc_basis, 0)

    for step in 1:N_steps
        # --- Target states for this step (mirrors creation_step_states) ---
        target_spinflip = spinup(qubit_basis) ⊗ fockstate(osc_basis, step - 1)
        target_final    = spindown(qubit_basis) ⊗ fockstate(osc_basis, step)

        # --- Parameter ranges for this step ---
        T_exc_ini, T_exc_fin = 1e-5, 5e-2

        δ_ωd = 100.0
        ωd_ini, ωd_fin = Δ0 - δ_ωd, Δ0 + δ_ωd

        t_swap_th = π / (2 * g * sqrt(step))
        δ1, δ2 = 0.0005, 0.00015
        τ_SWAP_ini, τ_SWAP_fin = t_swap_th - δ1, t_swap_th + δ2

        parameters_range = [[T_exc_ini, T_exc_fin], [ωd_ini, ωd_fin], [τ_SWAP_ini, τ_SWAP_fin]]

        # --- 1. Generate a dataset from the current initial_state, over a spectrum of drive values ---
        outputs = FL_1step_3p_NN_outputs(prs, parameters_range, dim_parameters_space, n_samples)
        inputs  = FL_1step_3p_NN_inputs(t0, initial_state, target_spinflip, target_final, decom_basis, outputs, n_samples)

        save_dataset(joinpath(save_dir, "dataset_step$(step).jld2"), inputs, outputs;
            description="FL_1step_3p ladder step $(step) dataset (N_steps=$(N_steps))",
            params=Dict{Symbol,Any}(:step=>step, :n_samples=>n_samples, :N_mech=>N_mech, :g=>g, :t0=>t0))

        println("step $(step)/$(N_steps): saved $(n_samples)-sample dataset to dataset_step$(step).jld2")

        # --- 2. Train a NN on (inputs, outputs) ---
        nn = train_NN(inputs, outputs; train_fraction=train_fraction, hidden=hidden, η=η,
                      epochs=epochs, loss=loss, batch_size=batch_size)

        # --- 3. Predict the drive parameters that reach target_final ---
        predicted_output, predicted_state, prediction_infidelity =
            predict_drive_parameters(nn, t0, initial_state, target_final, decom_basis)

        println("step $(step)/$(N_steps): test error = $(nn.test_error), " *
                "predicted (τ_exc,ωd,τ_SWAP) = $(predicted_output), infidelity = $(prediction_infidelity)")

        # --- 4. Reiterate with the state the predicted parameters actually reach ---
        τ_exc_pred, ωd_pred, τ_SWAP_pred = predicted_output
        t0 += τ_exc_pred + τ_SWAP_pred
        initial_state = predicted_state
    end

    nothing
end

# Not called on include — dataset generation at realistic n_samples takes
# real time (minutes, at old_version's own n_samples=800/dim_parameters_
# space=[100,100,100], per step). Run explicitly, e.g. from the REPL after
# including this file:
#
#   run_Fock_ladder(6)                      # old_version's own N_steps, full-size
#   run_Fock_ladder(3; n_samples=20)         # a cheaper multi-step check
