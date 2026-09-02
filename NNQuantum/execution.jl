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
include("FockLadder_problem.jl")   # (renamed from HBAR-qubit_problem.jl)

# FockLadder_problem.jl's (a)-(g) are the *setup* of the HBAR+qubit system
# (parameters, subsystems, Hamiltonian, dissipators, protocol runner) — all
# live code, exercised by this include. Step 2's actual validation run
# (physically-tuned pulse parameters, plot ⟨n_qubit⟩/⟨n_mech⟩) lives in
# test.jl, not here — see CLAUDE.md's Scope note on test.jl's role.
