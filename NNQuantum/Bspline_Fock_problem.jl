# Bspline_Fock_problem.jl
#
# The `:FL_1step_Bspline` protocol — a resonant, single-stage, B-spline-drive
# variant of the same qubit+HBAR system `FockLadder_problem.jl`'s
# `:FL_1step_3p` uses, introduced in CLAUDE.md's "Resonant single-stage
# B-spline drive" Plan section and worked out in DESIGN.md Parts 14-16.
# Split into its own file (DESIGN.md Part 19), moved out of
# `FockLadder_problem.jl` where it used to live (Parts 15-16), once it
# became clear the two protocols don't actually share a "ladder" structure:
# `:FL_1step_3p`'s pulse is too rigid to do more than swap one quantum per
# call, so reaching Fock state `n` needs `n` reiterated calls; the B-spline
# drive is expressive enough (2×n_basis free coefficients over a window
# `[0,T]`, vs. `:FL_1step_3p`'s 3 numbers) to be aimed directly at a target
# Fock number in one shot — no rungs, no reiteration (DESIGN.md Part 19).
#
# Standalone, not `include`-dependent on `FockLadder_problem.jl`: this file
# duplicates the small handful of physical constants (γm, κ, κϕ, g, nthm)
# `:FL_1step_3p` also uses (deliberately reused *unchanged*, per user
# confirmation — DESIGN.md Part 15) rather than pulling in that entire
# file's own unrelated apparatus (cs, H0, J, FLstep_dynamics_3p, its own
# dataset functions). The two problems coexist, run independently, from
# separate execution files.
#
# Follows the project's standing three-way rule (DESIGN.md Part 18,
# restated in CLAUDE.md's Scope note) exactly like FockLadder_problem.jl
# does: this file holds the quantum problem's own definition (physical
# setup, raw dynamics/protocol runner) *and* its own dataset input/output
# creation (FL_1step_Bspline_NN_outputs/_inputs) plus NN-facing glue that is
# genuinely problem-specific (predict_drive_parameters_Bspline) — none of
# that belongs in NNQuantum.jl (fully generic) or Bspline_Fock_execution.jl
# (thin orchestrator only).

include("Definition.jl")   # h, hbar, kb — needed below for Teq/nthm

using QuantumDynamics
using QuantumOptics   # expect — needed by (h) below (dataset input/output creation)

include("NNQuantum.jl")   # qo_infidelity, rand_hermitian_orthonormal_basis, train_NN,
                            # predict_and_score, generate_decom_basis — this file's own
                            # dataset-creation/prediction-glue functions need these

# --- Physical parameters --------------------------------------------------------
#
# Same Chu et al. values as FockLadder_problem.jl's :FL_1step_3p (b) —
# duplicated here, not shared via `include`, since this file is meant to be
# fully standalone (see header note above). If these ever need to change,
# update both files.
γm = 0.025;  # mechanical bath dissipation rate
Teq  = kb / (2 * pi * hbar) * 1e-3 * 10e-3;
nthm = 1 / (exp(5.9614e6 / Teq) - 1);  # mechanical bath population (ωm — see below)
κ  = 19;     # qubit decay rate
κϕ = 0.25;   # qubit dephasing rate
g  = 258;    # JC coupling rate

# Note on nthm: Teq's/nthm's defining formula only ever needs a mechanical
# frequency to set the bath's thermal occupation at that frequency — it is
# *not* the same ωm that would appear in this variant's own Hamiltonian
# (which has no bare-energy term at all, see Hon below); it is Chu et al.'s
# physical mechanical resonator frequency, reused for exactly the same
# reason :FL_1step_3p's own nthm uses it. Written as a literal here (rather
# than binding a module-level `ωm`) specifically so this file doesn't
# introduce its own `ωm`/`ωq`/`Δ0` globals that could be mistaken for having
# any bearing on Hon's construction (they don't — Hon's subsystems are both
# built at ω=0, unconditionally, per the resonance condition below).

# N_mech is NOT shared with :FL_1step_3p's N_mech=5 (DESIGN.md Part 15): a
# small "for now" value, not physically derived. Composite-system dimension
# d=2×(N_mech_Bspline+1)=8, vs. :FL_1step_3p's d=12 — the two variants'
# decom_basis therefore can't be shared (confirmed, DESIGN.md Part 15/16).
N_mech_Bspline = 3

# --- Subsystems and composite system --------------------------------------------
#
# Fully resonant regime (DESIGN.md Part 14, point 1): ωm = ωq = ωd, so
# Δ0 = 0 exactly — both subsystems built at ω=0.
qubit_res = Qubit(:qubit, 0.0)
osc_res   = HarmonicOscillator(:osc, 0.0; nmax=N_mech_Bspline)
cs_res    = CompositeSystem(qubit_res, osc_res)

# --- Hamiltonian (Hon) -----------------------------------------------------------
#
# Hon = g(σ+b + σ−b†) + ΩRe(t)σx − ΩIm(t)σy — static JC coupling plus a
# purely time-dependent drive, no bare-energy term at all: both subsystems
# are at ω=0, so bare_hamiltonian(cs_res) ≡ 0 and jaynes_cummings(cs_res,...)
# alone already *is* Hon's complete time-independent part. Verified
# algebraically (DESIGN.md Part 14): with σ± = (σx±iσy)/2 and Ω=ΩRe+iΩIm,
# Ωσ+ + Ω*σ− = ΩReσx − ΩImσy exactly — no extra carrier term needed, the
# resonant frame already absorbs it.
Hon = jaynes_cummings(cs_res, :qubit, :osc, g)

# --- Dissipators -------------------------------------------------------------------
#
# Same four dissipators as :FL_1step_3p, same rates (γm, nthm, κ, κϕ —
# user-confirmed, unchanged), rebuilt on cs_res.
J_res = jump_operators(cs_res, [thermal_bath(:osc, γm, nthm)..., Dephasing(:qubit, κϕ), Decay(:qubit, κ)])

# --- B-spline decomposition of ΩRe(t)/ΩIm(t) --------------------------------------
#
# Ports old_version's BSplineKit-based machinery unchanged
# (ML_QM_library.jl:413-427: generate_Bspline_basis, Bspline_composition,
# drive_from_normalized_spline). degree=4 (cubic), n_basis=10 — old_version's
# own (only-ever-sketched, never exercised with concrete numbers) defaults,
# reused directly (DESIGN.md Part 16). generate_Bspline_basis's
# BSplineBasis(BSplineOrder(degree), knots) constructor produces a *clamped*
# basis (repeated boundary knots), which is what lets the spline settle to a
# fixed value at t=0/t=T with no separate envelope function needed (unlike
# :FL_1step_3p's π_pulse_shape). ΩRe(t)/ΩIm(t) share one basis (one knot
# vector); only their coefficient vectors differ.
using BSplineKit

function generate_Bspline_basis(degree::Int, n_basis::Int, domain::Tuple{Float64,Float64})
    knots = range(domain[1], domain[2], length=n_basis - degree + 2)
    return BSplineBasis(BSplineOrder(degree), knots)
end

function Bspline_composition(coeffs::Vector{Float64}, basis::BSplineBasis)
    spline = Spline(basis, coeffs)
    return x -> spline(x)
end

# t ∈ [0,T] → τ = t/T ∈ [0,1], the spline's own domain (`domain` above).
function drive_from_normalized_spline(spline, T::Float64)
    return t -> spline(t / T)
end

# --- Dynamics: single continuous-window protocol runner ---------------------------
#
# One evolve call over [t0,t0+T] — no spin-flip/SWAP hand-off, per
# DESIGN.md Part 14 point 2. ΩRe/ΩIm are built from B-spline coefficients
# via drive_from_normalized_spline, which expects an argument in [0,T]
# (elapsed time since t0) — so the closures below shift evolve's absolute t
# by t0 before calling it.
function FLstep_dynamics_Bspline(t0, initial_state, T, coeffs_Re::Vector{Float64}, coeffs_Im::Vector{Float64}, basis_spline::BSplineBasis)
    ΩRe = drive_from_normalized_spline(Bspline_composition(coeffs_Re, basis_spline), T)
    ΩIm = drive_from_normalized_spline(Bspline_composition(coeffs_Im, basis_spline), T)

    H = add_time_dependence(Hon,
        (t -> ΩRe(t - t0))  => op(cs_res, :qubit, :σx),
        (t -> -ΩIm(t - t0)) => op(cs_res, :qubit, :σy))

    return evolve((t0, t0 + T), initial_state, H, J_res)
end

# --- Densely-sampled single-window trajectory (for plotting) ----------------------
#
# Same role as FockLadder_problem.jl's FLstep_dynamics_3p_dense:
# FLstep_dynamics_Bspline's own (tspan, states) is only 2 points (a bare
# 2-tuple tspan, correct for dataset generation, far too coarse to plot).
"""
    FLstep_dynamics_Bspline_dense(t0, initial_state, T, coeffs_Re, coeffs_Im, basis_spline; npoints=200)

Same single-window `:FL_1step_Bspline` protocol as `FLstep_dynamics_Bspline`,
but sampling `npoints` points across `[t0,t0+T]` instead of just the two
endpoints — for plotting a trajectory, not for dataset generation
(`FLstep_dynamics_Bspline` remains the right, cheaper choice there). Returns
`(tspan, states)`.
"""
function FLstep_dynamics_Bspline_dense(t0, initial_state, T, coeffs_Re::Vector{Float64}, coeffs_Im::Vector{Float64}, basis_spline::BSplineBasis; npoints::Integer=200)
    ΩRe = drive_from_normalized_spline(Bspline_composition(coeffs_Re, basis_spline), T)
    ΩIm = drive_from_normalized_spline(Bspline_composition(coeffs_Im, basis_spline), T)

    H = add_time_dependence(Hon,
        (t -> ΩRe(t - t0))  => op(cs_res, :qubit, :σx),
        (t -> -ΩIm(t - t0)) => op(cs_res, :qubit, :σy))

    tspan = range(t0, t0 + T; length=npoints)
    return evolve(tspan, initial_state, H, J_res)
end

# --- (h) Dataset generation (NN inputs/outputs) -----------------------------------
#
# Per DESIGN.md Part 14's NN input/output layout — a deliberate role
# reversal from :FL_1step_3p's own: the pulse *duration* here is sampled per
# row and becomes an NN **input** feature (not an output), while the NN's
# **output** is the B-spline coefficients themselves.
#
#   NN input  = decom_basis expectation values ⊕ one infidelity ⊕ T
#   NN output = coeffs_Re ⊕ coeffs_Im  (2×n_basis numbers total)
#
# This is why, unlike :FL_1step_3p_NN_outputs (whose sampled tuples serve
# directly, unchanged, as `outputs` for save_dataset/train_NN),
# FL_1step_Bspline_NN_outputs returns TWO things: `pulse_parameters`
# (T, coeffs_Re, coeffs_Im — everything FLstep_dynamics_Bspline needs to
# actually simulate a row) and `outputs` (coeffs_Re⊕coeffs_Im only — the
# genuine training target, with T deliberately excluded since T belongs on
# the input side, built by FL_1step_Bspline_NN_inputs below instead).
#
# T is sampled uniformly in [0,T_max) (DESIGN.md Part 19 — the user's own
# design: T_max fixed, each row's T sampled below it, used both as that
# row's own dynamics timespan and as an NN input feature — motivated by
# letting shorter, less-dissipation-exposed pulses be favored where the
# dataset shows they achieve lower infidelity, rather than fixing one
# duration for every row). Each of the 2×n_basis coefficients is sampled
# independently, uniformly in `coeff_range` — no threeD_parameter_space/
# weighted_sample reuse here (those are fixed at exactly 3 dimensions);
# straightforward independent uniform sampling is simplest for this
# `1+2×n_basis`-dimensional space and needs no new NNQuantum.jl utility.
function FL_1step_Bspline_NN_outputs(T_max::Real, coeff_range::Tuple{<:Real,<:Real}, n_basis::Integer, n_samples::Integer)
    lo, hi = coeff_range
    pulse_parameters = Vector{Tuple{Float64,Vector{Float64},Vector{Float64}}}(undef, n_samples)
    outputs = Vector{Vector{Float64}}(undef, n_samples)

    for i in 1:n_samples
        T = rand() * T_max
        coeffs_Re = lo .+ rand(n_basis) .* (hi - lo)
        coeffs_Im = lo .+ rand(n_basis) .* (hi - lo)
        pulse_parameters[i] = (T, coeffs_Re, coeffs_Im)
        outputs[i] = vcat(coeffs_Re, coeffs_Im)
    end

    return pulse_parameters, outputs
end

# Runs each of the n_samples (T, coeffs_Re, coeffs_Im) triples in
# pulse_parameters — as produced by FL_1step_Bspline_NN_outputs above —
# through FLstep_dynamics_Bspline, and builds one NN input row per sample:
# decom_basis expectation values ⊕ one infidelity (final state vs.
# target_final) ⊕ T. No spin-flip-stage infidelity the way
# FL_1step_3p_NN_inputs has one — there is no intermediate stage in this
# protocol (DESIGN.md Part 14 point 2), so only one infidelity exists to
# report.
function FL_1step_Bspline_NN_inputs(t0, initial_state, target_final, decom_basis, basis_spline::BSplineBasis, pulse_parameters, n_samples::Integer)
    inputs = Vector{Vector{Float64}}(undef, n_samples)

    for i in 1:n_samples
        T, coeffs_Re, coeffs_Im = pulse_parameters[i]
        _, states = FLstep_dynamics_Bspline(t0, initial_state, T, coeffs_Re, coeffs_Im, basis_spline)
        ψ_at_end = states[end]

        infidelity = qo_infidelity(ψ_at_end, target_final)
        inputs[i] = vcat([real(expect(matrix, ψ_at_end)) for matrix in decom_basis], infidelity, T)
    end

    return inputs
end

# --- (i) NN-facing glue: predict_drive_parameters_Bspline -------------------------
#
# The B-spline counterpart of FockLadder_problem.jl's
# predict_drive_parameters. Genuinely problem-specific glue, per the same
# Part 18 rule: builds the input NNQuantum.jl's predict_and_score needs but
# can't supply itself, and a `simulate` closure over
# FLstep_dynamics_Bspline/t0/initial_state.
#
# Needs a `T` the original :FL_1step_3p version never did (DESIGN.md Part
# 19's own discussion): since T is an NN *input*, not an output, asking
# "what coefficients reach the target?" is really "...over a window of
# length T?" — a choice that has to be made before asking, not something
# the model infers. Per the user's own resolution: sweep `n_candidates`
# values of T evenly across [0,T_max] (skipping the degenerate T=0 point —
# a zero-length evolve call trivially returns the initial state unchanged,
# never a meaningful candidate), build the target's input row and predict
# coefficients at each one, actually re-simulate each candidate with
# FLstep_dynamics_Bspline (not just trust the raw prediction — same
# "verify by re-simulating" principle predict_and_score already applies
# once, here swept over T too), and keep whichever candidate achieves the
# best real infidelity. This lets the dissipation-vs-expressiveness
# tradeoff the dataset itself encodes (shorter T, generally less
# decoherence exposure) decide empirically, rather than guessing one T.
#
# Returns a NamedTuple (predicted_output, final_state, infidelity, T) — the
# extra `T` field (predict_and_score itself only ever returns three values)
# is needed by the caller (run_FockTarget_Bspline) to know which duration
# the winning candidate actually used, e.g. to plot its trajectory.
function predict_drive_parameters_Bspline(nn, t0, initial_state, target_final, decom_basis, basis_spline::BSplineBasis, T_max::Real; n_candidates::Integer=10)
    best = nothing

    for T in range(0.0, T_max; length=n_candidates)
        T <= 0.0 && continue   # skip the degenerate zero-duration candidate

        x_target = vcat([real(expect(matrix, target_final)) for matrix in decom_basis], 0.0, T)

        simulate = predicted_output -> begin
            n_basis = length(predicted_output) ÷ 2
            coeffs_Re = predicted_output[1:n_basis]
            coeffs_Im = predicted_output[n_basis+1:end]
            _, states = FLstep_dynamics_Bspline(t0, initial_state, T, coeffs_Re, coeffs_Im, basis_spline)
            states[end]
        end

        predicted_output, final_state, infidelity = predict_and_score(
            nn.model, x_target, nn.maxs_input, nn.mins_input, nn.maxs_output, nn.mins_output, simulate, target_final)

        if best === nothing || infidelity < best.infidelity
            best = (predicted_output=predicted_output, final_state=final_state, infidelity=infidelity, T=T)
        end
    end

    return best
end
