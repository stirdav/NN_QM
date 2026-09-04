# Solver defaults for `evolve`. QuantumOptics.jl's own (`DP5()`, `reltol=1e-6`,
# `abstol=1e-8`) can be silently inaccurate on time-dependent Hamiltonians;
# `Vern9()` plus tighter tolerances converge on those cases at comparable cost.
# Override per call via `alg`/`reltol`/`abstol`. `OrdinaryDiffEqVerner.Vern7()`
# is the lighter-memory swap for very large Hilbert spaces. See "Solver
# wrappers" in design.md.
const DEFAULT_ALG = OrdinaryDiffEqVerner.Vern9()
const DEFAULT_RELTOL = 1e-8
const DEFAULT_ABSTOL = 1e-10

# Post-solve guard: `evolve` warns if the final state's norm² (closed) or trace
# (open) has drifted from 1 by more than this. See "Solver wrappers" in design.md.
const NORMALIZATION_ATOL = 1e-4

# A custom `fout` returning something other than a state (expectation values, a
# tuple, ...) is not checkable and passes through silently.
function _check_normalization((tout, states))
    isempty(states) && return (tout, states)
    final = last(states)
    drift, quantity = if final isa StateVector
        abs(norm(final)^2 - 1), "norm²"
    elseif final isa AbstractOperator
        abs(real(tr(final)) - 1), "trace"
    else
        return (tout, states)
    end
    if drift > NORMALIZATION_ATOL
        @warn """
        evolve: final-state $quantity deviates from 1 by $drift (> NORMALIZATION_ATOL \
        = $NORMALIZATION_ATOL). The ODE solve is likely under-resolving a feature in a \
        time-dependent H — try a smaller reltol/abstol, or a dtmax/tstops tied to the \
        sharpest feature in H (see "Solver wrappers" in design.md). Pass \
        check_normalization=false if the evolution is non-norm-preserving by design.
        """ maxlog = 1
    end
    return (tout, states)
end

"""
    evolve(tspan, psi0, H; alg, reltol, abstol, check_normalization=true, kwargs...)

Closed-system evolution. Dispatches on `H`: a plain, time-independent
`AbstractOperator` goes to `QuantumOptics.timeevolution.schroedinger`; an
`AbstractTimeDependentOperator` (e.g. a `TimeDependentSum`, see
"Time-dependent Hamiltonians" in design.md) goes to `.schroedinger_dynamic`.

`alg` / `reltol` / `abstol` default to [`DEFAULT_ALG`](@ref) /
[`DEFAULT_RELTOL`](@ref) / [`DEFAULT_ABSTOL`](@ref) — higher-order and tighter
than `QuantumOptics.jl`'s own defaults. See "Solver wrappers" in design.md for
the reasoning, for a separate narrow-feature failure mode that these defaults
do not fix, and for the recommended solver when `H` is stiff. These and any
other `kwargs` (`fout`, `dtmax`, `tstops`, ...) are forwarded to the
underlying solver.

Unless `check_normalization=false`, warns if the final state's norm² drifted
from 1 by more than [`NORMALIZATION_ATOL`](@ref).
"""
function evolve(tspan, psi0, H::AbstractOperator;
                alg=DEFAULT_ALG, reltol=DEFAULT_RELTOL, abstol=DEFAULT_ABSTOL,
                check_normalization::Bool=true, kwargs...)
    result = timeevolution.schroedinger(tspan, psi0, H; alg, reltol, abstol, kwargs...)
    check_normalization ? _check_normalization(result) : result
end

function evolve(tspan, psi0, H::AbstractTimeDependentOperator;
                alg=DEFAULT_ALG, reltol=DEFAULT_RELTOL, abstol=DEFAULT_ABSTOL,
                check_normalization::Bool=true, kwargs...)
    result = timeevolution.schroedinger_dynamic(tspan, psi0, H; alg, reltol, abstol, kwargs...)
    check_normalization ? _check_normalization(result) : result
end

"""
    evolve(tspan, rho0, H, J; rates=nothing, alg, reltol, abstol, check_normalization=true, kwargs...)

Open-system evolution via the master equation. Dispatches on `H` the same way
as the closed-system [`evolve`](@ref): time-independent goes to
`QuantumOptics.timeevolution.master`, time-dependent goes to `.master_dynamic`.
`rho0` may be a density operator or a state vector (converted automatically).
`J` is a vector of embedded jump operators — build it with
[`jump_operators`](@ref) — matching `QuantumOptics.jl`'s own `J` argument name.

`alg` / `reltol` / `abstol` default as in the closed-system method; `rates`,
`Jdagger`, and other `kwargs` are forwarded. See "Solver wrappers" in
design.md, which also covers the recommended solver when `H` is stiff — common
for open systems with widely separated rates. Unless
`check_normalization=false`, warns if `tr(rho)` drifted from 1 by more than
[`NORMALIZATION_ATOL`](@ref).
"""
function evolve(tspan, rho0, H::AbstractOperator, J::AbstractVector;
                alg=DEFAULT_ALG, reltol=DEFAULT_RELTOL, abstol=DEFAULT_ABSTOL,
                check_normalization::Bool=true, kwargs...)
    result = timeevolution.master(tspan, rho0, H, J; alg, reltol, abstol, kwargs...)
    check_normalization ? _check_normalization(result) : result
end

function evolve(tspan, rho0, H::AbstractTimeDependentOperator, J::AbstractVector;
                alg=DEFAULT_ALG, reltol=DEFAULT_RELTOL, abstol=DEFAULT_ABSTOL,
                check_normalization::Bool=true, kwargs...)
    result = timeevolution.master_dynamic(tspan, rho0, H, J; alg, reltol, abstol, kwargs...)
    check_normalization ? _check_normalization(result) : result
end
