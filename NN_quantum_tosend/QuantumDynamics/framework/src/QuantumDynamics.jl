module QuantumDynamics

using QuantumOptics
using JLD2
import OrdinaryDiffEqVerner  # Vern9 (evolve's default ODE algorithm)
# `embed`/`ptrace` are QuantumInterface functions re-exported by QuantumOptics;
# a plain `using QuantumOptics` doesn't let a bare `function embed(...) = ...`
# extend the existing generic function — it silently creates a new, separate
# one local to this module instead. Explicit `import` is required to add
# methods (see `embed(cs::CompositeSystem, ...)` in composite.jl and
# `ptrace(cs::CompositeSystem, ...)` in measurement.jl).
import QuantumOptics: embed, ptrace

include("subsystems.jl")
include("composite.jl")
include("measurement.jl")
include("hamiltonians.jl")
include("dissipation.jl")
include("evolution.jl")
include("io.jl")

export AbstractSubsystem, Qubit, HarmonicOscillator, Transmon
export CompositeSystem, getsubsystem
export op, embed
export bare_hamiltonian, jaynes_cummings, rabi, tavis_cummings, dispersive_hamiltonian, quadratic_coupling, add_time_dependence
export Dissipator, Decay, Dephasing, Gain, jump_operator, jump_operators, thermal_bath
export evolve
export ptrace, condition_on, check_fock_cutoff, FockCutoffTooSmall
export SimulationResult, save_result, load_result

end # module QuantumDynamics
