# Definition.jl
#
# Physical constants for NNQuantum, ported from old_version/definitions.jl.
# QuantumDynamics has no physical constants of its own (see FUNCTION_MAPPING.md
# discussion) — it only consumes plain numbers (ω, rate, nth) that the problem
# layer already computed, so this is the only piece of old_version's
# definitions.jl carried over here. The QM-side struct builders (ho, qubit,
# qub_ho / Harmonic_oscillator, Qubit, Qubit_HO) are NOT ported — QuantumDynamics's
# own Qubit/HarmonicOscillator/CompositeSystem replace that role (see (c) in
# FockLadder_problem.jl, renamed from HBAR-qubit_problem.jl).
#
# Note: old_version/definitions.jl:10 writes hbar's exponent with a Unicode
# minus (U+2212, "−") instead of ASCII "-" — not a valid Julia numeric literal
# sign, fixed below.

# Physical constants in SI units
const h    = 6.62607015e-34   # Planck constant (J*s)
const hbar = 1.054571817e-34  # Reduced Planck constant (J*s)
const kb   = 1.380649e-23     # Boltzmann constant (J/K)
