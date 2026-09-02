# HBAR-qubit_problem.jl
#
# Step 1 of NNQuantum's plan (see CLAUDE.md / DESIGN.md): rewrite of old_version's
# HBAR-qubit problem (:FL_1step_3p variant only) on top of QuantumDynamics.
#
# Sub-points (c)-(g) not implemented yet — code goes here only when explicitly requested.

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
# using QuantumDynamics
#
# qubit = Qubit(:qubit, ωq)
# osc   = HarmonicOscillator(:osc, ωm; nmax=N_mech)
# cs    = CompositeSystem(qubit, osc)

# --- (d) Bare + JC-coupling Hamiltonian --------------------------------------
#
# Provisional, not a resolution of the lab-frame-vs-rotating-frame question
# (FUNCTION_MAPPING.md §2, §9): QuantumDynamics's Hamiltonian recipes
# (bare_hamiltonian, jaynes_cummings) are lab-frame only, with no
# rotating-frame builder anywhere in the framework. old_version's H0 is
# written in the frame rotating at ωm — mechanical bare term vanishes, qubit
# bare term collapses to 0.5*Δ0*σz.
#
# H0 here is hand-built from cs's cached embedded operators to structurally
# mirror old_version's rotating-frame H0 exactly, bypassing jaynes_cummings
# (which would need pre-transformed frequencies fed to Qubit/HarmonicOscillator
# and would leave a harmless-but-nonzero global-phase mismatch — see
# FUNCTION_MAPPING.md §2's two options). No Δ0_tilde correction, per (a).
#
# Δ0 = ωq - ωm
#
# H_JC = g * (op(cs,:qubit,:σp) * op(cs,:osc,:a) + op(cs,:qubit,:σm) * op(cs,:osc,:ad))
# H0   = 0.5 * Δ0 * op(cs,:qubit,:σz) + H_JC

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

# Ω(t): spin-flip stage, carrier at the free drive frequency ωd (per (a), no
# fixed Δ0_tilde carrier). Δ(t): SWAP stage, detuning pulse — Δ0 per (a), no
# Δ0_tilde correction. Wired onto H0 via add_time_dependence, replacing
# old_version's mutated-LazySum/closure pattern (FUNCTION_MAPPING.md §3, the
# cleanest 1:1 correspondence in the whole mapping).
#
# Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(ωd * t)
# Δ(t) = -Δ0 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)
#
# H = add_time_dependence(H0,
#     t -> Ω(t)   => op(cs, :qubit, :σx),
#     t -> Δ(t)/2 => op(cs, :qubit, :σz))

# --- (f) Dissipators -----------------------------------------------------------
#
# Maps exactly onto old_version's four fixed jump operators, factor-for-factor
# (FUNCTION_MAPPING.md §4): thermal_bath(:osc, γm, nthm) replaces the mechanical
# decay+gain pair with one call that keeps rate/nth in sync by construction;
# Dephasing(:qubit, κϕ) and Decay(:qubit, κ) (default nth=0) replace the qubit
# dephasing/decay operators. No dagger bookkeeping needed — evolve/master_dynamic
# computes J† internally (old_version threads (H,J,J†) through by hand).
#
# J = jump_operators(cs, [thermal_bath(:osc, γm, nthm)..., Dephasing(:qubit, κϕ), Decay(:qubit, κ)])

# --- (g) Two-stage protocol runner ----------------------------------------------
#
# Two evolve calls (spin-flip stage, then SWAP stage), final state of stage 1
# feeding stage 2 — direct port of old_version's FLstep_dynamics_3p orchestration,
# per DESIGN.md's "as-is" choice. Ω_R = π/τ_exc derived, per (a).
#
# Kept as two separate evolve calls rather than one continuous evolve over a
# combined schedule (the more idiomatic QuantumDynamics style used in its own
# fock_state_preparation examples, FUNCTION_MAPPING.md §3) — not resolved here,
# same "provisional, not decided" status as (d)'s frame question. Collapsing to
# one call would need tstops/dtmax at the stage boundary to avoid the "narrow
# kick" adaptive-solver failure mode evolve documents, since τ_exc/τ_SWAP are
# short relative to the full trajectory.
#
# function FLstep_dynamics_3p(t0, initial_state, τ_exc, ωd, τ_SWAP)
#     Ω_R = π / τ_exc
#     Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(ωd * t)
#     Δ(t) = -Δ0 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)
#
#     H = add_time_dependence(H0,
#         t -> Ω(t)   => op(cs, :qubit, :σx),
#         t -> Δ(t)/2 => op(cs, :qubit, :σz))
#
#     # stage 1: spin-flip
#     tspan1, states1 = evolve((t0, t0 + τ_exc), initial_state, H, J)
#     ψ_at_flip = states1[end]
#
#     # stage 2: SWAP
#     tspan2, states2 = evolve((t0 + τ_exc, t0 + τ_exc + τ_SWAP), ψ_at_flip, H, J)
#     ψ_at_end = states2[end]
#
#     return vcat(tspan1, tspan2[2:end]), vcat(states1, states2[2:end])
# end
