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
#
# `:FL_1step_Bspline` (the resonant single-stage B-spline drive variant,
# DESIGN.md Part 14) used to live further down in this same file (Parts
# 15-16) — since moved out to its own `Bspline_Fock_problem.jl`/
# `Bspline_Fock_execution.jl` pair (DESIGN.md Part 19), once it became clear
# the two protocols don't share a "ladder" structure at all (the B-spline
# drive targets a Fock number directly, no rung-by-rung reiteration) and
# duplicating a couple of physical constants into a fully standalone file
# was preferable to an inter-problem `include` dependency between two
# nominally-independent protocols.
#
# This file follows the project's standing three-way rule (DESIGN.md Part
# 18, restated in CLAUDE.md's Scope note): NNQuantum.jl holds the fully
# generic ML/dataset-management engine (training/testing/prediction, plus
# dataset/basis/NN persistence) and must never reference this problem by
# name; THIS file holds the quantum problem's own definition (physical
# setup, raw dynamics/protocol runners — (a)-(g)) *and* its own dataset
# input/output creation (FL_1step_3p_NN_outputs/_inputs) plus any other
# NN-facing glue that is genuinely problem-specific
# (predict_drive_parameters) — none of that belongs in NNQuantum.jl, and
# none of it belongs in FockLadder_execution.jl either, which stays a thin
# orchestrator that only calls into this file and NNQuantum.jl.

include("../src/Definition.jl")   # h, hbar, kb — needed below for Teq/nthm

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
using QuantumOptics   # expect — needed by (h) below (dataset input/output creation)

include("../src/NNQuantum.jl")   # qo_infidelity, rand_hermitian_orthonormal_basis, threeD_parameter_space,
                            # weighted_sample, uniform_parameter_sample, train_NN, predict_and_score —
                            # this file's own dataset-creation/prediction-glue functions (h) need these

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
# evolve call above is passed a bare 2-point tspan, correct for what
# FockLadder_execution.jl's dataset generation actually needs (endpoints
# only), but far too coarse to plot a curve against. Promoted here (unchanged)
# from test.jl's own run_short_dynamics, a test-local helper that mirrored
# FLstep_dynamics_3p's construction line-for-line just to get plottable
# resolution — needed as reusable, non-test-local code once approaching step
# 2's (v) staged validation raised a real need to plot a *predicted*
# ladder-step trajectory, not just the fixed smoke-test one test.jl itself
# already covers.
#
# Kept as its own function rather than adding an `npoints` keyword to
# FLstep_dynamics_3p itself: the two functions serve different callers with
# different needs — dataset generation wants the fast, 3-point endpoints-only
# path on every sample; a validation/plotting call wants the slower,
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
# Per this project's standing rule (DESIGN.md Part 18): dataset input/output
# *creation* — sampling pulse parameters, then running them through this
# problem's own dynamics function to build NN feature rows — is
# problem-specific and lives here, not in NNQuantum.jl (fully generic) or
# FockLadder_execution.jl (thin orchestrator only). Direct port of
# old_version's FL_1step_3p_NN_outputs/FL_1step_3p_NN_inputs
# (old_version/HBAR-qubit_problem/HBAR-qubit_problem.jl:325-358), kept under
# the same names for direct correspondence.
#
# The generic helpers these two depend on (qo_infidelity,
# rand_hermitian_orthonormal_basis, threeD_parameter_space,
# uniform_parameter_sample, weighted_sample) all live in NNQuantum.jl,
# included above.

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
# per (a). Whether/how these get bundled back into a struct is left open
# (CLAUDE.md).
#
# FLstep_dynamics_3p's own (tspan, states) return already samples exactly
# the 3 points needed here — states[1] the initial state, states[2] the
# spin-flip-stage endpoint, states[end] the full-protocol endpoint —
# because each stage's evolve call underneath is passed a bare 2-point
# tspan. So this needs no change to FLstep_dynamics_3p itself.
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

# --- (i) NN-facing glue: predict_drive_parameters ------------------------------
#
# train_NN itself is fully generic (NNQuantum.jl, no problem-specific
# content) so no wrapper for it lives here. predict_drive_parameters is
# genuinely problem-specific glue, per the same Part 18 rule as (h) above: it
# builds the one input NNQuantum.jl's predict_and_score needs but can't
# supply itself — the target state's own input-feature vector (decom_basis
# expectation values, concatenated with [0,0] for the two infidelities a
# state has with itself — matches Chu_DFL_execution.ipynb's
# target_input_1step exactly, and the row layout FL_1step_3p_NN_inputs
# builds for every other sample) — and a `simulate` closure over
# FLstep_dynamics_3p/t0/initial_state that turns a predicted
# (τ_exc,ωd,τ_SWAP) into the state it actually reaches. `nn` is train_NN's
# own return value (a NamedTuple: model + the four normalization stats) —
# passed straight through, not re-derived.
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
