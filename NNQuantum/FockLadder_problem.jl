# FockLadder_problem.jl
#
# (Renamed from HBAR-qubit_problem.jl — same file, content unchanged by the rename.)
#
# Step 1 of NNQuantum's plan (see CLAUDE.md / DESIGN.md): rewrite of old_version's
# HBAR-qubit problem (:FL_1step_3p variant only) on top of QuantumDynamics.
#
# What's written below is the *setup* of the HBAR+qubit system: physical
# parameters (b), subsystems/composite system (c), the bare+JC-coupling
# Hamiltonian (d), drive-shape machinery (e), dissipators (f), and the
# two-stage pulse protocol runner (g) built on top of that setup. All of
# (a)-(g) are live code below, not pseudocode.

include("Definition.jl")   # h, hbar, kb — needed below for Teq/nthm

# --- (a) Scope --------------------------------------------------------------
#
# Only the :FL_1step_3p variant of old_version's HBAR-qubit problem is ported.
#
# Pulse protocol: two stages, spin-flip then SWAP, driven by 3 free parameters:
#   τ_exc   — spin-flip stage duration
#   ωd      — drive frequency (free; old_version's :FL_1step fixes this to Δ0_tilde instead)
#   τ_SWAP  — SWAP stage duration
# Ω_R = π / τ_exc is derived, not a free parameter (unchanged from old_version).
#
# Explicitly out of scope:
#   - :FL_1step (fixed-carrier, 2-param variant)
#   - :FL_1step_2drives (BSpline-shaped two-drive variant)
#   - the detuning-correction term χ = g²/Δ0 (old_version's `typeofcorrection`
#     switch): present in old_version's create_FLstep_dynamics_3p only as dead,
#     commented-out code, never exercised — dropped rather than ported.
#     Net effect: the detuning used below is Δ0 unconditionally, no Δ0_tilde
#     correction and no correction-related parameter.

# --- (b) Parameters -----------------------------------------------------------
#
# N_mech (mechanical Fock cutoff) and g (JC coupling rate) were left unbound
# in old_version (N_mech a `...` placeholder, g commented out); both are now
# bound to concrete values below (Chu et al.), along with the rest of the
# physical parameters.
#

#Mechanical resonator
ωm = 5.9614e6 #[KHz];

#Qubit
ωq = 5.9456e6 #[KHz];

#Mechanical bath
γm = 0.025; #dissipation rate
Teq   = kb / (2 * pi * hbar) * 1e-3 * 10e-3;
nthm = 1 / (exp(ωm / Teq) - 1); #mechanical bath population

#Qubit bath
κ = 19; #dissipation rate
κϕ = 0.25; #dephasing rate

#= Qubit-resonator detuning and JC coupling =#
Δ0 = ωq - ωm; #System detuning
g = 258 ; #JC coupling rate
N_mech = 5;

# --- (c) Subsystems and composite system -------------------------------------
#
# Qubit and mechanical resonator as QuantumDynamics subsystems, tensored into
# one composite system. Direct use of QuantumDynamics's own constructors — no
# new subsystem type needed. Number-operator convention (n = (id+σz)/2 for the
# qubit) already matches old_version's n_qubit, so no sign-convention risk here.
#
# Frequencies passed in are already pre-transformed into the frame rotating at
# ωm, per (d) below — Qubit gets Δ0 = ωq - ωm (not the bare ωq), the mechanical
# oscillator gets 0.0 (not the bare ωm).
using QuantumDynamics
using QuantumOptics   # expect — needed directly by (h) below
using JLD2            # jldsave — needed by (h)'s save_dataset

include("NNQuantum.jl")   # qo_infidelity, rand_hermitian_orthonormal_basis, threeD_parameter_space — needed by (h) below

qubit = Qubit(:qubit, Δ0)
osc   = HarmonicOscillator(:osc, 0.0; nmax=N_mech)
cs    = CompositeSystem(qubit, osc)

# --- (d) Bare + JC-coupling Hamiltonian --------------------------------------
#
# Resolves the lab-frame-vs-rotating-frame question left open in earlier
# drafts of this file (FUNCTION_MAPPING.md §2, §9): QuantumDynamics has no
# rotating-frame *builder*, but jaynes_cummings can still be made to produce
# the correct rotating-frame Hamiltonian directly, with no framework changes,
# by pre-transforming the frequencies fed to the subsystems in (c) instead.
#
# Why this is exact, not an approximation: old_version's frame is a common
# rotation U(t) = exp(iωm·t·(n_q+n_mech)) applied to both subsystems together.
# The JC coupling term g*(σp*a + σm*ad) conserves n_q+n_mech (the same
# commutation fact _lower/_raise's docstring relies on), so it commutes with
# U's generator and picks up no time dependence at all under the transform —
# it comes through unchanged. Only the bare energies shift:
#   ωq*n_q + ωm*n_mech  →  (ωq-ωm)*n_q + 0*n_mech  =  Δ0*n_q
# (derivation: H_I = U*H*U' + i*(dU/dt)*U' = H - ωm*(n_q+n_mech) here, since
# U H U' = H). So constructing Qubit(:qubit, Δ0) / HarmonicOscillator(:osc,
# 0.0) and calling jaynes_cummings on them reproduces this exactly.
#
# The one difference from old_version's own H0: Δ0*n_q = 0.5*Δ0*id +
# 0.5*Δ0*σz, so this H0 carries an extra constant 0.5*Δ0*id term that
# old_version's 0.5*Δ0*σz drops. That term is a pure global phase — provably
# inert for any observable, expectation value, or fidelity — so the two
# Hamiltonians are physically identical; only a direct matrix diff would see
# a difference. No Δ0_tilde correction, per (a).
#
# Same reasoning shows (f)'s dissipators need no frame-transforming either:
# Decay/Gain/Dephasing's jump operators (a, ad, σz, σm) all commute with U's
# generator up to a pure phase, which drops out of the Lindblad dissipator
# term LρL' - ... entirely.
H0 = jaynes_cummings(cs, :qubit, :osc, g)

# --- (e) Drive terms ----------------------------------------------------------
#
# π_pulse_shape ported unchanged from old_version/ML_QM_library.jl:188-196 — a
# pure-math sin² window (no framework dependency), so unlike (c)/(d) it's real,
# runnable code rather than commented-out pseudocode: it doesn't depend on any
# of this file's still-unbound quantities.

function π_pulse_shape(t, t0, duration, eps=1e-12)
    δt = t - t0
    if 0.0 <= δt < duration
        s = sin(pi * δt / duration)^2
        return s / (s + eps^2)
    else
        return 0.0
    end
end

# Ω(t) (spin-flip stage, carrier at the free drive frequency ωd, per (a)) and
# Δ(t) (SWAP stage, detuning pulse — Δ0 per (a), no Δ0_tilde correction), wired
# onto H0 via add_time_dependence, are NOT defined here as standalone top-level
# code. Unlike ωq/ωm/g/κ/etc. above, τ_exc/ωd/τ_SWAP/t0 are per-call pulse
# parameters, not fixed physical constants — they only make sense as arguments
# to a function, not as module-level globals. So they're folded directly into
# (g)'s FLstep_dynamics_3p, the one place they're actually used, rather than
# duplicated here as a free-standing (and slightly artificial) demo.

# --- (f) Dissipators -----------------------------------------------------------
#
# Maps exactly onto old_version's four fixed jump operators, factor-for-factor
# (FUNCTION_MAPPING.md §4): thermal_bath(:osc, γm, nthm) replaces the mechanical
# decay+gain pair with one call that keeps rate/nth in sync by construction;
# Dephasing(:qubit, κϕ) and Decay(:qubit, κ) (default nth=0) replace the qubit
# dephasing/decay operators. No dagger bookkeeping needed — evolve/master_dynamic
# computes J† internally (old_version threads (H,J,J†) through by hand).
#
J = jump_operators(cs, [thermal_bath(:osc, γm, nthm)..., Dephasing(:qubit, κϕ), Decay(:qubit, κ)])

# --- (g) Two-stage protocol runner ----------------------------------------------
#
# Two evolve calls (spin-flip stage, then SWAP stage), final state of stage 1
# feeding stage 2 — direct port of old_version's FLstep_dynamics_3p orchestration,
# per DESIGN.md's "as-is" choice. Ω_R = π/τ_exc derived, per (a). (e)'s drive
# terms are folded in directly here (τ_exc/ωd/τ_SWAP/t0 are this function's
# arguments, not module-level globals — see the note where (e) used to be).
#
# Kept as two separate evolve calls rather than one continuous evolve over a
# combined schedule (the more idiomatic QuantumDynamics style used in its own
# fock_state_preparation examples, FUNCTION_MAPPING.md §3) — not resolved here,
# a genuinely open question distinct from (d)'s (now-resolved) frame gap.
# Collapsing to one call would need tstops/dtmax at the stage boundary to avoid
# the "narrow kick" adaptive-solver failure mode evolve documents, since
# τ_exc/τ_SWAP are short relative to the full trajectory.
#
# Verified end-to-end with a scratch initial state (arbitrary, non-physical
# τ_exc/ωd/τ_SWAP): returns a well-formed (tspan, states) trajectory with
# tr(ρ_final) = 1.0 exactly. Real physically-motivated pulse parameters and an
# actual initial_state are step 2's job, not this one.

function FLstep_dynamics_3p(t0, initial_state, τ_exc, ωd, τ_SWAP)
    Ω_R = π / τ_exc
    Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(ωd * t)
    Δ(t) = -Δ0 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    H = add_time_dependence(H0,
        (t -> Ω(t))   => op(cs, :qubit, :σx),
        (t -> Δ(t)/2) => op(cs, :qubit, :σz))

    # stage 1: spin-flip
    tspan1, states1 = evolve((t0, t0 + τ_exc), initial_state, H, J)
    ψ_at_flip = states1[end]

    # stage 2: SWAP
    tspan2, states2 = evolve((t0 + τ_exc, t0 + τ_exc + τ_SWAP), ψ_at_flip, H, J)
    ψ_at_end = states2[end]

    return vcat(tspan1, tspan2[2:end]), vcat(states1, states2[2:end])
end

# --- Densely-sampled two-stage trajectory (for plotting) -----------------------
#
# FLstep_dynamics_3p's own (tspan, states) is only 3 points — each stage's
# evolve call above is passed a bare 2-point tspan, correct for what (h)/
# run_Fock_ladder actually need (endpoints only), but far too coarse to plot
# a curve against. Promoted here (unchanged) from test.jl's own
# run_short_dynamics, a test-local helper that mirrored FLstep_dynamics_3p's
# construction line-for-line just to get plottable resolution — needed as
# reusable, non-test-local code once approaching step 2's (v) staged
# validation raised a real need to plot a *predicted* ladder-step trajectory,
# not just the fixed smoke-test one test.jl itself already covers.
#
# Kept as its own function rather than adding an `npoints` keyword to
# FLstep_dynamics_3p itself: the two functions serve different callers with
# different needs (h)/run_Fock_ladder want the fast, 3-point endpoints-only
# path on every dataset sample; a validation/plotting call wants the slower,
# densely-sampled one, occasionally. Same H0/J/π_pulse_shape, same two-stage
# spin-flip-then-SWAP hand-off as FLstep_dynamics_3p — only each stage's
# tspan differs (range(...; length=npoints) instead of a bare (t_start,
# t_end) tuple).
"""
    FLstep_dynamics_3p_dense(t0, initial_state, τ_exc, ωd, τ_SWAP; npoints=200)

Same two-stage protocol as `FLstep_dynamics_3p`, but sampling each stage at
`npoints` points instead of just its two endpoints — for plotting a
trajectory (e.g. with `NNQuantum.jl`'s `plot_trajectory`), not for dataset
generation (`FLstep_dynamics_3p` remains the right, cheaper choice there).
Returns `(tspan, states)`, the same shape `FLstep_dynamics_3p` returns, just
with `2*npoints - 1` points instead of 3.
"""
function FLstep_dynamics_3p_dense(t0, initial_state, τ_exc, ωd, τ_SWAP; npoints::Integer=200)
    Ω_R = π / τ_exc
    Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(ωd * t)
    Δ(t) = -Δ0 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    H = add_time_dependence(H0,
        (t -> Ω(t))   => op(cs, :qubit, :σx),
        (t -> Δ(t)/2) => op(cs, :qubit, :σz))

    # stage 1: spin-flip
    tspan1 = range(t0, t0 + τ_exc; length=npoints)
    _, states1 = evolve(tspan1, initial_state, H, J)
    ψ_at_flip = states1[end]

    # stage 2: SWAP
    tspan2 = range(t0 + τ_exc, t0 + τ_exc + τ_SWAP; length=npoints)
    _, states2 = evolve(tspan2, ψ_at_flip, H, J)

    return vcat(collect(tspan1), collect(tspan2[2:end])), vcat(states1, states2[2:end])
end

# --- (h) Dataset generation (NN inputs/outputs) -------------------------------
#
# NOT part of the (a)-(g) system setup above. Step 3 of NNQuantum's plan
# ("translate old_version's ML engine, kept as general as possible") hasn't
# formally started (see CLAUDE.md) — this is a narrow, explicitly-requested
# port of just the two dataset-generation functions old_version defines for
# :FL_1step_3p (FL_1step_3p_NN_outputs / FL_1step_3p_NN_inputs, in
# old_version/HBAR-qubit_problem/HBAR-qubit_problem.jl:325-358), kept under
# the same names for direct correspondence.
#
# The generic helpers those two depend on (qo_infidelity,
# rand_hermitian_orthonormal_basis, threeD_parameter_space) now live in
# NNQuantum.jl (included above) — DESIGN.md Part 6 (i): they were
# provisionally colocated here until that library file existed; it now
# does, so they moved out. weighted_sample below is the one exception,
# per NNQuantum.jl's own header note.
#
# weighted_sample replaces old_version's `sample(space, Weights(prob), n)`
# (StatsBase) with a small hand-rolled cumulative-weight sampler, to avoid
# pulling in a new dependency for one weighted-sampling call.
function weighted_sample(items, weights, n)
    cumw = cumsum(weights)
    total = cumw[end]
    total > 0 || throw(ArgumentError("weighted_sample: all weights are zero"))
    return [items[searchsortedfirst(cumw, rand() * total)] for _ in 1:n]
end

# Direct port of old_version's FL_1step_3p_NN_outputs
# (HBAR-qubit_problem.jl:325-329): weighted-samples n_samples
# (τ_exc, ωd, τ_SWAP) triples from a 3D grid over parameters_range,
# weighted by the caller-supplied probability function p(τ_exc, ωd, τ_SWAP).
#
# `uniform=true` bypasses the grid entirely, via NNQuantum.jl's
# `uniform_parameter_sample` — O(n_samples) instead of the grid-based path's
# O(∏dim_parameters_space), and continuous instead of quantized to
# dim_parameters_space's resolution. Only equivalent to the default path when
# `p` really is uniform over the box; every caller in this project (`prs` in
# both `run_Fock_ladder` and `test.jl`'s dataset-generation smoke test) is,
# so both pass `uniform=true`. Default stays `false` so this function's own
# behavior for any other caller/`p` is unchanged.
function FL_1step_3p_NN_outputs(p, parameters_range, dim_parameters_space, n_samples; uniform::Bool=false)
    if uniform
        return uniform_parameter_sample(parameters_range, n_samples)
    end
    parameters_space, prob = threeD_parameter_space(p, parameters_range, dim_parameters_space)
    return weighted_sample(parameters_space, prob, n_samples)
end

# Direct port of old_version's FL_1step_3p_NN_inputs
# (HBAR-qubit_problem.jl:331-358). Runs each of the n_samples
# (τ_exc, ωd, τ_SWAP) triples in pulse_parameters — as produced by
# FL_1step_3p_NN_outputs above; that function's "outputs" become this
# function's "pulse_parameters" input, old_version's naming, kept as-is —
# through FLstep_dynamics_3p, and builds one NN input row per sample:
# expectation values of the final state on decom_basis, concatenated with
# the spin-flip-stage infidelity and the full-protocol infidelity vs.
# target_final.
#
# Flattens old_version's dataset_features::fl_1step_features struct
# (state_target_spinflip, state_target_1step, decom_basis, phonon_n,
# correction) down to the three fields this variant actually needs —
# phonon_n/correction were old_version's detuning-correction knobs, dropped
# per (a). Whether/how these get bundled back into a struct is left to
# step 3 ("kept as general as possible" — an open question, see CLAUDE.md).
#
# FLstep_dynamics_3p's own (tspan, states) return already samples exactly
# the 3 points needed here — states[1] the initial state, states[2] the
# spin-flip-stage endpoint, states[end] the full-protocol endpoint —
# because each stage's evolve call underneath is passed a bare 2-point
# tspan (see (g)'s status note in DESIGN.md). So this needs no change to
# FLstep_dynamics_3p itself.
function FL_1step_3p_NN_inputs(t0, initial_state, target_spinflip, target_final, decom_basis, pulse_parameters, n_samples)
    inputs = Vector{Vector{Float64}}(undef, n_samples)
    final_states = Vector{Any}(undef, n_samples)
    infidelity_spin_flip = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples
        τ_exc, ωd, τ_SWAP = pulse_parameters[i]
        _, states = FLstep_dynamics_3p(t0, initial_state, τ_exc, ωd, τ_SWAP)
        ψ_at_flip, ψ_at_end = states[2], states[end]

        final_states[i] = ψ_at_end
        infidelity_spin_flip[i] = qo_infidelity(ψ_at_flip, target_spinflip)
        inputs[i] = [real(expect(matrix, ψ_at_end)) for matrix in decom_basis]
    end

    swap_infidelity = qo_infidelity.(final_states, Ref(target_final))
    return [vcat(inputs[i], infidelity_spin_flip[i], swap_infidelity[i]) for i in 1:n_samples]
end

# --- Dataset persistence (JLD2) -----------------------------------------------
#
# Still (h), not step 3 proper (see this section's own opening note above).
# Saves the dataset FL_1step_3p_NN_outputs/FL_1step_3p_NN_inputs produce to a
# .jld2 file, one row per sample: vcat(inputs[i], outputs[i]) — the NN input
# features (decom_basis expectations + 2 infidelities) followed by the
# (τ_exc, ωd, τ_SWAP) pulse-parameter triple that produced them. Same row
# layout as old_version's dataset_creation (ML_QM_library.jl:232-253, input
# columns then output columns), but as a standalone function here rather
# than dataset_creation's own role folded into old_version's larger
# dataset_generation orchestrator — no such orchestrator exists in
# NNQuantum yet (step 3, unstarted).
#
# jldsave with separate top-level keys plus a format_version, not one
# serialized blob, follows QuantumDynamics/framework's own io.jl convention
# (see save_result there) — a schema change later can then fall back on a
# missing key instead of depending on JLD2 correctly reconstructing an
# evolved struct shape.
const DATASET_FORMAT_VERSION = 1

function dataset_rows(inputs, outputs)
    length(inputs) == length(outputs) || throw(ArgumentError(
        "dataset_rows: inputs and outputs have different lengths ($(length(inputs)) vs $(length(outputs)))"))
    return [Float64.(vcat(inputs[i], collect(outputs[i]))) for i in eachindex(inputs)]
end

"""
    save_dataset(path, inputs, outputs; description="", params=Dict{Symbol,Any}())

Save an FL_1step_3p dataset to `path` (a `.jld2` file). `inputs`/`outputs`
are `FL_1step_3p_NN_inputs`/`FL_1step_3p_NN_outputs`'s own return values —
same length, row `i` of the saved `dataset` matrix is
`vcat(inputs[i], outputs[i])`. `dim_input`/`dim_output` are saved alongside
so a later `dataset[:, 1:dim_input]`/`dataset[:, dim_input+1:end]` split
doesn't need `inputs`/`outputs` themselves still in memory.
"""
function save_dataset(path::AbstractString, inputs, outputs;
                       description::AbstractString="", params::Dict{Symbol,Any}=Dict{Symbol,Any}())
    rows = dataset_rows(inputs, outputs)
    jldsave(path;
        format_version=DATASET_FORMAT_VERSION,
        dataset=permutedims(reduce(hcat, rows)),
        dim_input=length(inputs[1]),
        dim_output=length(outputs[1]),
        description=description,
        params=params,
    )
    nothing
end

# --- (iv) NN wrappers: train_NN / predict_drive_parameters --------------------
#
# DESIGN.md Part 6 (iv): closes run_Fock_ladder's steps 2-3 stubs
# (FockLadder_execution.jl). These two functions are deliberately thin —
# everything problem-agnostic (dataset splitting, normalization, model
# architecture, the training/testing loop, the predict-then-resimulate-
# then-score orchestration) lives in NNQuantum.jl (i)/(ii)/(iv) and is
# called from here, not reimplemented. train_NN/predict_drive_parameters
# are the only two places in this file that connect the two: everything
# NNQuantum.jl itself touches is a plain matrix, vector, model, or a
# `simulate` function value — it never sees a pulse parameter, a
# decom_basis, or FLstep_dynamics_3p by name. That boundary is what makes
# NNQuantum.jl's machinery reusable as-is for a different quantum problem,
# or a different drive parameterization (e.g. :FL_1step_2drives's BSpline
# coefficients, were it ever ported) — swapping in a new problem only ever
# means writing a new pair of thin wrappers like these, never touching
# NNQuantum.jl.
#
# train_NN converts FL_1step_3p_NN_outputs/_inputs's own return shapes —
# inputs a Vector{Vector{Float64}} (one feature row per sample, from
# FL_1step_3p_NN_inputs), outputs a Vector{NTuple{3,Float64}} (one
# (τ_exc,ωd,τ_SWAP) sample per row, from weighted_sample drawing straight
# out of threeD_parameter_space's tuple grid, not vectors) — into the
# plain Float64 matrices NNQuantum.jl's train_and_test_NN expects (hence
# `collect.(outputs)` below, turning each tuple into a vector `reduce(hcat,
# ...)` can stack), then delegates to the matrix method below.
#
# A second method, taking X/Y as plain matrices directly, was added
# alongside run_Fock_ladder's dataset_mode option (DESIGN.md): a dataset
# loaded from disk via NNQuantum.jl's load_dataset already comes back as
# plain Float64 matrices, not FL_1step_3p_NN_outputs/_inputs's own
# vector-of-vector/vector-of-tuple shapes — this lets run_Fock_ladder call
# `train_NN` the same way regardless of whether the dataset was just
# generated or loaded from a previous run's saved file, via ordinary Julia
# multiple dispatch rather than a runtime branch inside one method.
# train_fraction=0.9375 matches Chu_DFL_execution.ipynb's own
# :FL_1step_3p/:master_dynamic cell (n_training=750 of n_samples=800);
# clamped to leave at least one sample on each side of the split so a small
# smoke-test n_samples doesn't produce an empty train or test set.
function train_NN(X::AbstractMatrix, Y::AbstractMatrix; train_fraction::Real=0.9375, kwargs...)
    n_training = clamp(round(Int, train_fraction * size(X, 1)), 1, size(X, 1) - 1)
    return train_and_test_NN(X, Y, n_training; kwargs...)
end

function train_NN(inputs::AbstractVector, outputs::AbstractVector; kwargs...)
    X = permutedims(reduce(hcat, inputs))
    Y = permutedims(reduce(hcat, collect.(outputs)))
    return train_NN(X, Y; kwargs...)
end

# predict_drive_parameters builds the one problem-specific piece
# NNQuantum.jl's predict_and_score needs but can't supply itself: the
# target state's own input-feature vector (decom_basis expectation values,
# concatenated with [0,0] for the two infidelities a state has with
# itself — matches Chu_DFL_execution.ipynb's target_input_1step exactly,
# and the row layout FL_1step_3p_NN_inputs builds for every other sample),
# and a `simulate` closure over FLstep_dynamics_3p/t0/initial_state that
# turns a predicted (τ_exc,ωd,τ_SWAP) into the state it actually reaches.
# `nn` is train_NN's own return value (a NamedTuple: model + the four
# normalization stats) — passed straight through, not re-derived.
function predict_drive_parameters(nn, t0, initial_state, target_final, decom_basis)
    x_target = vcat([real(expect(matrix, target_final)) for matrix in decom_basis], 0.0, 0.0)

    simulate = predicted_output -> begin
        τ_exc, ωd, τ_SWAP = predicted_output
        _, states = FLstep_dynamics_3p(t0, initial_state, τ_exc, ωd, τ_SWAP)
        states[end]
    end

    return predict_and_score(nn.model, x_target, nn.maxs_input, nn.mins_input,
                              nn.maxs_output, nn.mins_output, simulate, target_final)
end
