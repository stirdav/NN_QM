# execution.jl
#
# Execution workflow for NNQuantum, step 1 scope only (see CLAUDE.md / DESIGN.md):
# activates this folder's environment and includes the files needed to define
# the HBAR-qubit :FL_1step_3p problem. There is no learning-engine/dataset
# workflow here yet (that's step 3 of the plan, not started) — old_version's
# ML_QM_execution.jl is a much larger template because it also sets up the NN
# dataset/training pipeline; NNQuantum doesn't have that piece yet.
#
# Run from NNQuantum/ (REPL: include("execution.jl"), or `julia execution.jl`).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using QuantumDynamics
using QuantumOptics

include("Definition.jl")
include("HBAR-qubit_problem.jl")

# (c)-(g) in HBAR-qubit_problem.jl are still commented-out pseudocode (see that
# file) — nothing beyond the includes above is runnable yet. Once they're made
# live, this is where step 2's validation run (build cs/H0/J, call
# FLstep_dynamics_3p, plot ⟨n_qubit⟩/⟨n_mech⟩) will go.
