using QuantumDynamics, QuantumOptics

# Paper's Fig. 4 reference parameters, expressed as ratios to g (matching how
# the paper itself parameterizes Fig. 4: γ1/g, γφ/g, γm/g, and gt on the axes)
# — ωq never appears, since both Hcs/H1 are already in ωq's interaction frame.
const ωm_over_g = 1e4      # paper: ωm=1 GHz, g=100 kHz
const γm_over_g = 0.01     # paper: γm=1 kHz
const nm_th = 1.0

function build_system(; nmax::Int, ωm::Real=ωm_over_g, ωq::Real=0.0)
    qubit = Qubit(:qubit, ωq)                      # ω unused by Hcs/H1 (interaction frame); only Hlab's bare energy needs it
    osc   = HarmonicOscillator(:osc, ωm; nmax=nmax)
    CompositeSystem(qubit, osc)
end

# Hcs = gcs*(a²+ad²)*σz, Eq. 4 — the closed-system protocol Hamiltonian.
# Not `quadratic_coupling` (that's g*(a+ad)²*σz = g*(a²+ad²+2n+1)*σz, the
# lab-frame Eq. 1 term with the extra 2n+1 piece; Hcs is the further RWA'd,
# interaction-frame form with no bare energies).
function Hcs(cs, gcs)
    a, ad, σz = op(cs, :osc, :a), op(cs, :osc, :ad), op(cs, :qubit, :σz)
    gcs * (a^2 + ad^2) * σz
end

# H1, Eq. S4 — interaction-frame Hamiltonian with the drive, used for Fig. 4.
# Built via `add_time_dependence`: a static term (g*(2n+id)*σz) plus three
# time-dependent terms — of which only the last (the classical σx drive) is
# actually a drive. The a²σz/ad²σz terms are counter-rotating leftovers of
# moving Hcs into the interaction frame (see `Hcs` above): their time
# dependence is a frame artifact, not an external field.
function H1(cs, g, ωm, A, ωd)
    a, ad, n, id = op(cs, :osc, :a), op(cs, :osc, :ad), op(cs, :osc, :n), op(cs, :osc, :id)
    σz, σx = op(cs, :qubit, :σz), op(cs, :qubit, :σx)
    Hstatic = g * (2n + id) * σz
    add_time_dependence(Hstatic,
        (t -> g * exp(-2im * ωm * t)) => a^2 * σz,
        (t -> g * exp(2im * ωm * t))  => ad^2 * σz,
        (t -> A * cos(ωd * t))        => σx,
    )
end

# Hlab, Eq. 1 + Eq. 2 — the lab-frame Hamiltonian H1 is derived from (via the
# exact, non-approximate V1 transform of Eq. S3), with ωq resolved explicitly
# rather than rotated away. `quadratic_coupling` already is Eq. 1's bare-energy
# + coupling part (ωq*n_q + ωm*n_o + g*(a+ad)²*σz — equal to ωq/2*σz + ωm*b†b +
# g(b+b†)²σz up to the global-phase constant ωq/2 folded into `Qubit`'s `n`);
# only the lab-frame drive (Eq. 2, oscillating at ωq itself rather than a
# constant σx as in H1) needs adding. Requires `cs`'s qubit built with its
# actual ωq (`build_system(...; ωq)`), unlike Hcs/H1 which never read it.
function Hlab(cs, ωq, ωm, g, A, ωd)
    σx, σy = op(cs, :qubit, :σx), op(cs, :qubit, :σy)
    Hstatic = quadratic_coupling(cs, :qubit, :osc, g)
    add_time_dependence(Hstatic,
        (t -> A * cos(ωd * t) * cos(ωq * t)) => σx,
        (t -> A * cos(ωd * t) * sin(ωq * t)) => σy,
    )
end

# Project the qubit onto |±⟩ = (|e⟩±|g⟩)/√2 and trace it out, renormalizing
# the remaining oscillator state — the measurement step common to Fig. 1 and
# Fig. 4. Thin, paper-specific wrapper (which local basis, which sign) around
# the framework's `condition_on`, promoted to src/ from this example's own
# hand-rolled embed/ptrace/renormalize logic once validated.
function conditional_oscillator_state(cs, state, sign::Symbol)
    qb = getsubsystem(cs, :qubit).basis
    pm = normalize(sign === :plus ? spinup(qb) + spindown(qb) : spinup(qb) - spindown(qb))
    ρ_cond, _ = condition_on(cs, state, :qubit, projector(pm))
    ρ_cond
end
