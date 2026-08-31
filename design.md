# QuantumDynamics Design

Detailed specs and design rationale for the QuantumDynamics framework. See `CLAUDE.md`
for a high-level project overview.

## Goal

A general framework for cavity QED / circuit QED simulations in Julia,
built as a thin organizational layer on top of `QuantumOptics.jl` (the
standard Julia package for Hilbert spaces, operators, and master-equation /
quantum-trajectory solvers — analogous to QuTiP). QuantumDynamics itself does not
reimplement any linear-algebra or solver machinery; it only adds:

- reusable, named building blocks for common subsystems (qubits, oscillators),
- a composite-system abstraction for combining them into a joint Hilbert space,
- a library of common Hamiltonian recipes (Jaynes-Cummings, Rabi,
  Tavis-Cummings, dispersive approximation),
- a consistent caching strategy so repeated Hamiltonian construction (e.g. in
  time-dependent problems or parameter sweeps) stays cheap,
- dissipation channels (`Decay`, `Dephasing`) that build jump operators
  the same way Hamiltonians are built — dispatched per subsystem type, from
  cached embedded operators,
- a single `evolve` entry point that dispatches to the right
  `QuantumOptics.jl` solver (`schroedinger`/`schroedinger_dynamic`/
  `master`/`master_dynamic`) based on whether the problem is closed or open
  and time-independent or time-dependent.

## Subsystems (`src/subsystems.jl`)

- `AbstractSubsystem` is the common supertype for all subsystem kinds.
- Every subsystem stores: a `name::Symbol` (used for lookup in composite
  systems), a bare frequency `ω::Float64`, its local `basis`, and a cached
  `ops::NamedTuple` of operators built once at construction time.
- Every subsystem's `ops` includes both `n` (the pure excitation-number
  operator, always integer-valued: `0, 1, 2, ...`) and `h0` (the subsystem's
  bare local Hamiltonian). For `Qubit`/`HarmonicOscillator`, `h0 = ω*n`; for
  `Transmon`, `h0` additionally includes the anharmonic correction (see
  below). The two are kept separate because `n` is what excitation-number
  *conservation* arguments (JC/Tavis-Cummings, tested via `[n_total, H]=0`)
  need, while `h0` is what `bare_hamiltonian` sums — a subsystem type is
  free to make its `h0` more than `ω*n` without disturbing that invariant.
- **Operators are precomputed and cached, not computed on demand.** Decision:
  storing them as fields avoids rebuilding sparse matrices on every access.
  This matters most for time-dependent Hamiltonians, which need the same
  local operators repeatedly — see "Time-dependent Hamiltonians" below.
- `ops` is a `NamedTuple` rather than individual struct fields so each
  subsystem type can expose a different operator set without bloating the
  struct definition, while staying type-stable (same keys/types per
  subsystem type).
- `op(s::AbstractSubsystem, key::Symbol) = s.ops[key]` is the uniform
  accessor, e.g. `op(qubit, :σz)`, `op(cavity, :a)`.

### `Qubit`

- Two-level system on a `SpinBasis(1//2)`.
- Cached ops: `σx, σy, σz, σp, σm, n, id`.
- `n = (id + σz) / 2` is the excitation-number operator (0 = ground,
  1 = excited).
- **Sign convention (load-bearing detail):** QuantumOptics.jl's `sigmam`
  lowers spin-**up** (`σz = +1`) to spin-down, i.e. spin-up is the
  higher-energy/excited state in their convention. `n` must therefore be
  built from `+σz`, not `-σz`, for `σm`/`σp` to behave as proper
  lowering/raising operators consistent with `n`. Getting this backwards
  silently produces a Hamiltonian that looks fine (Hermitian, right shape)
  but whose interaction terms (e.g. `ad*σm + a*σp`) do **not** conserve
  total excitation number — this was caught during development via exactly
  that commutator check, which is now a regression test
  (`test/runtests.jl`, "Excitation number conservation").

### `HarmonicOscillator`

- Bosonic mode on a `FockBasis(nmax)` (truncated Fock space, `nmax` states
  `0:nmax`).
- Cached ops: `a` (destroy), `ad` (create), `n` (number), `id`.
- `n` here is QuantumOptics' own `number(basis)` — the reference convention
  that the `Qubit`'s `n` was deliberately aligned with, so that bare
  energies for *any* subsystem type can be written uniformly as `ω * n`.

### `Transmon`

- Weakly anharmonic multi-level bosonic mode on a `FockBasis(nmax)`, same
  underlying basis type as `HarmonicOscillator`. Default `nmax=4` (rather
  than `HarmonicOscillator`'s `10`): anharmonicity increasingly detunes
  higher transitions, so only the first few levels are usually relevant —
  override for problems that need more.
- Cached ops: `a, ad, n, h0, id` — same as `HarmonicOscillator` plus `h0`.
- Bare energy ladder `E_k = k*ω + (α/2)*k*(k-1)`, cached as
  `h0 = ω*n + (α/2)*n*(n - id)`. `α` is the anharmonicity; **negative** for
  a physical transmon (level spacing shrinks with `k`, typically
  `α/2π ~ -200 MHz` against `ω/2π ~ 5 GHz`). `α = 0` recovers a plain
  harmonic oscillator's bare energy.
- `n` is still the plain `number(basis)` operator (unaffected by `α`), so
  `Transmon` participates in excitation-number conservation checks exactly
  like `HarmonicOscillator` — only `h0` carries the anharmonic correction.
- Couples to an oscillator the same way `HarmonicOscillator` would (via
  `a`/`ad`), *not* via `σm`/`σp` — see "qubit-like coupling dispatch" below.

## Composite systems (`src/composite.jl`)

- `CompositeSystem(subs::AbstractSubsystem...)` joins several subsystems
  into one joint Hilbert space:
  - `basis = tensor(s1.basis, s2.basis, ...)`.
  - `index::Dict{Symbol,Int}` maps each subsystem's name to its position in
    the tensor product (needed for `embed`).
  - **Names must be unique**, checked at construction (`ArgumentError`
    otherwise): every lookup (`getsubsystem`, `op(cs, name, key)`, the
    `_lower`/`_raise`/`_decay_op`/etc. dispatch helpers) resolves through
    `index`, a plain `Dict`, so a duplicate name would otherwise silently
    collide there — the later subsystem wins, the earlier one becomes
    permanently unreachable by name despite still occupying a factor of the
    joint Hilbert space, with no error to flag it.
  - `ops::Dict{Symbol,NamedTuple}` holds, for every subsystem, every one of
    its local operators **already embedded** into the joint basis via
    `embed(basis, index[name], local_op)`.
- **Embedding is precomputed eagerly at construction, for every operator in
  every subsystem's `ops`** — not lazily/memoized on first use. Decision:
  building the tensor-product matrix (`embed`) is the expensive step,
  strictly more so than the local operator construction. Doing it once per
  `CompositeSystem`, up front, means Hamiltonian-builder functions (and, in
  time-dependent problems, the solver's per-step callback) never pay that
  cost again — they only look up already-embedded operators and combine
  them with cheap sparse-matrix addition/multiplication.
- `getsubsystem(cs, name)` — returns the original subsystem object
  registered under `name`.
- `op(cs::CompositeSystem, name, key)` — returns the cached, embedded
  operator (e.g. `op(cs, :cavity, :a)`), mirroring the subsystem-level `op`
  accessor.
- `embed(cs::CompositeSystem, name, local_op)` — embeds an arbitrary operator
  on subsystem `name`'s local basis into `cs.basis`, for operators that
  aren't one of the subsystem's cached ones (e.g. a projector built from an
  arbitrary local state — see "Measurement" below). Adds a method to
  `QuantumOptics.jl`'s own `embed`, the same function `CompositeSystem`'s
  constructor uses internally for every cached op.
  **Load-bearing detail:** `embed` (and `ptrace`, used the same way in
  `measurement.jl`) are actually owned by `QuantumInterface`, only
  re-exported by `QuantumOptics`. A bare `using QuantumOptics` followed by
  `function embed(new_signature...) = ...` does **not** extend the existing
  generic function — Julia silently creates a new, separate `embed` local to
  `QuantumDynamics`, shadowing the real one (confirmed by `methods(embed)`
  showing "1 method for generic function `embed` from QuantumDynamics"
  instead of the expected multi-method QuantumOptics/QuantumInterface
  function; every `CompositeSystem` constructor call then breaks with
  `MethodError: no method matching embed(...)`, since it's now looking at an
  empty method table for the *real* `embed`). Fixed by an explicit
  `import QuantumOptics: embed, ptrace` in `QuantumDynamics.jl`, before the
  `include`s — required for **any** future method added to a function that
  was only brought into scope via `using`, not just these two.

## Hamiltonians (`src/hamiltonians.jl`)

All recipes take a `CompositeSystem` plus subsystem names and coupling
constants, and return either a plain `Operator` (time-independent) or are
composed further into a `TimeDependentSum` (see below). All are built purely
from the composite system's cached, embedded operators — no `embed()` calls
happen inside these functions.

- `bare_hamiltonian(cs)` — `Σ h0` over every subsystem in `cs`, using each
  subsystem's cached `h0` (see "Subsystems" above). Generic across
  subsystem types since every `AbstractSubsystem` caches its own `h0`.
- **Qubit-like coupling dispatch:** `jaynes_cummings`, `rabi`, and
  `tavis_cummings` all take a "qubit-like" subsystem name and couple it to
  an oscillator. That subsystem may be a `Qubit` (2-level, coupling via
  `σm`/`σp`) or a `Transmon` (multi-level, coupling via `a`/`ad`) — resolved
  internally by two small dispatch helpers, `_lower`/`_raise`, that pattern
  match on the subsystem's type (`getsubsystem(cs, name)`) and return the
  right operator pair. This works because both pairs satisfy the same
  algebraic relation the conservation arguments below depend on:
  `[n, lower] = -lower`, `[n, raise] = +raise`. Adding a further "qubit-like"
  subsystem type only requires adding two more `_lower`/`_raise` methods —
  `jaynes_cummings`/`rabi`/`tavis_cummings` themselves don't change.
- `jaynes_cummings(cs, qubit_name, osc_name, g)` — bare energies plus the
  rotating-wave-approximation coupling `g*(ad*lower + a*raise)`. Conserves
  total excitation number `nq + no` (verified in tests, for both `Qubit`
  and `Transmon`).
- `rabi(cs, qubit_name, osc_name, g)` — same bare energies, full
  (non-RWA) coupling `g*(a+ad)*(lower+raise)`. Deliberately does **not**
  conserve excitation number (counter-rotating terms) — needed outside the
  strong detuning / weak coupling regime where the RWA breaks down. Its
  failure to commute with total number is checked in tests as a sanity
  control (proves the JC conservation test isn't vacuous).
- `tavis_cummings(cs, qubit_names, osc_name, g)` — multi-qubit
  generalization of Jaynes-Cummings: every subsystem in `qubit_names`
  (each independently a `Qubit` or `Transmon`) couples to the same
  oscillator mode. `g` may be a scalar (applied to all) or an iterable
  matching `qubit_names` (per-subsystem coupling strengths). When `g` isn't
  a scalar, `length(g)` must equal `length(qubit_names)` (checked, throws
  `ArgumentError` otherwise) — `zip`ping mismatched-length iterables would
  otherwise silently truncate to the shorter one, dropping trailing qubits
  (or trailing `g`s) from the coupling with no error.
- `dispersive_hamiltonian(cs, qubit_name, osc_name, g)` — second-order
  dispersive approximation, valid when detuning `Δ = ωq - ωc` is large
  compared to `g`. Computes `χ = g²/Δ` automatically from the subsystems'
  own frequencies (rather than requiring the caller to pass `χ` directly),
  giving `H = ωq*nq + ωc*no + χ*σz*no`. Validated against the full
  Jaynes-Cummings spectrum at large detuning (matches to ~`g⁴/Δ³`, the
  expected size of neglected higher-order terms). **Qubit-only** — unlike
  the three functions above, this one is not generalized to `Transmon`: a
  transmon's dispersive shift depends on its anharmonicity too and needs a
  different formula (not yet implemented). **Rejects `Δ=0`** (checked,
  throws `ArgumentError`): on resonance the approximation isn't a degenerate
  corner of its own validity domain, it's the resonant Jaynes-Cummings
  regime the approximation doesn't apply to at all — left unguarded,
  `χ = g²/Δ` would silently blow up to `Inf`/`NaN` instead of signaling that.
- `quadratic_coupling(cs, qubit_name, osc_name, g)` — a second-order
  (optomechanical-type) coupling `H = h0_q + h0_o + g*(a+ad)²*σz`, physically
  distinct from the "qubit-like" ladder coupling `jaynes_cummings`/`rabi`/
  `tavis_cummings` use (`σm`/`σp` or `a`/`ad` raising/lowering): here the
  oscillator's squeezing is conditioned directly on the qubit's `σz`
  eigenvalue, rather than the qubit and oscillator exchanging excitations.
  Relevant for qubit-oscillator systems with strong quadratic coupling (e.g.
  superconducting qubits coupled to mechanical oscillators). Deliberately
  does **not** conserve total excitation number (checked in tests as a
  sanity control, the same role `rabi`'s non-conservation plays against the
  JC conservation test — `(a+ad)²` mixes `a²`/`ad²` terms that change
  oscillator excitation number by ±2). **Qubit-only**, like
  `dispersive_hamiltonian`: uses `σz` directly, and there is no natural `σz`
  analogue for a multi-level `Transmon`, so this does not generalize the way
  the ladder-coupling recipes do.

## Dissipation (`src/dissipation.jl`)

- `Dissipator` is the common supertype for dissipation channels; each stores
  a subsystem `name::Symbol` and a `rate::Float64`. Three concrete channels:
  `Decay(name, rate; nth=0.0)` (amplitude damping — cavity photon loss `κ`,
  qubit relaxation `γ1` — optionally thermal, see below), `Gain(name, rate;
  nth)` (amplitude gain/thermal excitation, `Decay`'s counterpart), and
  `Dephasing(name, rate)` (pure dephasing). `rate < 0` (and, where
  applicable, `nth < 0`) raises `ArgumentError` at construction. All structs
  implement this as their sole inner constructor (rather than a guard on the
  outer `Real`-typed convenience method) specifically because a bare
  `Float64` argument dispatches to a struct's auto-generated default inner
  constructor ahead of any outer method with a wider (`Real`) signature —
  an outer-only guard would silently not run for `Decay(:cavity, -0.3)`.
- `jump_operator(cs, d::Dissipator)` builds the single embedded jump
  operator for one channel; `jump_operators(cs, dissipators)` builds the
  list for an iterable of them, meant to be passed straight through as the
  `J` argument of [`evolve`](#solver-wrappers-srcevolutionjl). Named to match
  `QuantumOptics.jl`'s own vocabulary: `master`/`master_dynamic` take these
  as an argument literally named `J`, documented there as "jump operators"
  (not QuTiP's "collapse operator"/`c_ops` naming) — since QuantumDynamics is
  a thin layer over `QuantumOptics.jl` specifically (see "Goal"), its own API
  follows that library's terminology rather than a different one.
- **Jump operators are dispatched per subsystem type**, via three small
  internal helpers (`_decay_op`, `_gain_op`, `_dephasing_op`) that
  pattern-match on `getsubsystem(cs, name)` — the same dispatch shape as
  `_lower`/`_raise` in `hamiltonians.jl`. `_decay_op` returns `σm` for a
  `Qubit`, `a` for a `HarmonicOscillator`/`Transmon`. `_gain_op` (the "raise"
  counterpart used by `Gain`) returns `σp`/`ad` respectively. `_dephasing_op`
  returns `σz` for a `Qubit`, `n` for a `HarmonicOscillator`/`Transmon`.
  - **Decision: not reused from `_lower`/`_raise`.** Those are documented
    specifically as the "qubit-like" coupling dispatch (only ever called
    with a `Qubit`/`Transmon` name in `jaynes_cummings` etc.); decay (and
    gain) is physically meaningful for a bare `HarmonicOscillator` too
    (cavity photon loss/thermal absorption), so extending `_lower`/`_raise`
    to cover it would silently widen what `jaynes_cummings`'s `qubit_name`
    argument accepts. Kept as separate, parallel helpers instead — a few
    lines of duplication in exchange for not disturbing that invariant.
- `jump_operator(cs, Decay(name, rate; nth=0.0)) = sqrt(rate*(nth+1)) *
  _decay_op(cs, name)` — at the default `nth=0` (a zero-temperature bath),
  this is `sqrt(rate) * _decay_op(cs, name)` and `rate` is directly the
  population decay rate: for a cavity with only `Decay(:cavity, κ)` and no
  drive/coupling, `d⟨n⟩/dt = -κ⟨n⟩` exactly (checked in tests against the
  analytic `n₀*exp(-κt)`, likewise qubit `T1` against `exp(-γ1*t)`).
- **Finite-temperature baths: `Decay`'s `nth` + `Gain` (load-bearing
  detail).** A `Decay` alone only ever models the relaxation half of a
  thermal bath. A full finite-temperature bath (thermal occupation `nth`,
  coupling rate `rate`) needs *two* jump operators — relaxation
  `sqrt(rate*(nth+1))*lower` and thermal excitation `sqrt(rate*nth)*raise` —
  built by combining `Decay` and `Gain` on the same subsystem with the same
  `(rate, nth)`:
  ```julia
  J = jump_operators(cs, [Decay(:osc, γm; nth=nm_th), Gain(:osc, γm; nth=nm_th)])
  ```
  reproducing the standard thermal Lindblad terms `γm*(nth+1)*D[b] +
  γm*nth*D[b†]`. **Decision: `Gain` is a separate dissipator, not an `nth`
  branch inside `Decay`'s `jump_operator`.** An earlier design had `Decay`
  itself return a second jump operator when `nth>0`, but that would have
  broken `jump_operator`'s documented single-`Operator` contract (only
  `jump_operators`, the plural/list-building function, could then handle a
  thermal `Decay`) and made `Decay`'s behavior branch internally on a
  keyword the caller might not know was there. Splitting the gain half into
  its own `Gain` type keeps every dissipator single-purpose: one type, one
  jump operator, always. `Gain`'s `nth` has **no default** (unlike `Decay`'s
  `nth=0.0`) — a bare `Gain` in isolation, with no `nth` in mind, isn't a
  meaningful thing to construct — but `nth=0` **is** allowed (revised from
  an earlier, stricter version of this API that rejected it): it simply
  produces a zero jump operator, so a temperature sweep ranging down to
  `nth=0` can always construct both `Decay` and `Gain` without a
  `nth > 0 ? [...] : [...]` branch at the call site. Verified in
  `test/runtests.jl` ("evolve: thermal bath (Decay+Gain) matches known
  solution") against the analytic thermal relaxation
  `⟨n⟩(t) = nth + (n₀-nth)*exp(-rate*t)`.
- **`thermal_bath(name, rate, nth)` — matched `Decay`+`Gain` pair.**
  Equivalent to `[Decay(name, rate; nth), Gain(name, rate; nth)]` but
  constructed together so `rate`/`nth` can't drift apart between the two
  (building them separately risks a typo desyncing one from the other,
  which produces a silently-wrong steady-state occupation rather than an
  error — there is no crash to catch it). Prefer this over the manual
  two-object form whenever both channels share the same bath parameters,
  i.e. whenever building a physical finite-temperature bath rather than two
  independently-tuned channels.
- **Dephasing sign/factor convention (load-bearing detail):**
  `jump_operator(cs, Dephasing(name, rate)) = sqrt(rate/2) * _dephasing_op(cs, name)`
  — note the `/2`. For the Lindblad dissipator `D[c]ρ = cρc† - {c†c,ρ}/2`
  with `c = sqrt(rate/2)*σz` (so `c†c = (rate/2)*I` since `σz² = I`), the
  off-diagonal qubit coherence decays as `dρ₀₁/dt = -rate*ρ₀₁` — i.e. `rate`
  is directly `1/Tφ`. Using `c = sqrt(rate)*σz` instead (dropping the `/2`)
  decays coherence at `2*rate`, which would silently halve the effective
  dephasing time for anyone plugging in a literature `1/Tφ` value. The `/2`
  keeps `Dephasing`'s `rate` consistent with `Decay`'s `rate`: both are
  exactly the number you'd read out of an experimental `T1`/`Tφ`. Verified
  numerically in `test/runtests.jl` ("master-equation dissipation matches
  known solutions") by evolving a `(|0⟩+|1⟩)/√2` qubit state under pure
  dephasing and fitting the `⟨σm⟩` decay envelope.
  - The `HarmonicOscillator`/`Transmon` branch (`sqrt(rate/2)*n`) is a
    reasonable generalization (photon-number-dependent dephasing, standard
    in circuit QED for modeling frequency noise) but does **not** have the
    same clean "`rate` = single coherence decay rate" property that the
    qubit case does, since `n` (unlike `σz`) doesn't satisfy `n²∝I` — the
    decay rate of `ρ_{nm}` for `n≠m` scales with `(n-m)²`. Kept for API
    symmetry with `Decay`, not because the rate has one universal meaning
    there.

## Solver wrappers (`src/evolution.jl`)

- `evolve` is a single generic entry point over all four
  `QuantumOptics.jl` time-evolution solvers, so call sites don't need to
  remember which of `schroedinger`/`schroedinger_dynamic`/`master`/
  `master_dynamic` matches a given problem — swapping a time-independent `H`
  for a `TimeDependentSum`, or adding `J`, doesn't require touching the
  function name.
  - `evolve(tspan, psi0, H)` — closed system. `H::AbstractOperator` dispatches
    to `timeevolution.schroedinger`; `H::AbstractTimeDependentOperator`
    (e.g. a `TimeDependentSum`, see "Time-dependent Hamiltonians" below)
    dispatches to `.schroedinger_dynamic`.
  - `evolve(tspan, rho0, H, J)` — open system via the master equation, same
    dispatch on `H` between `.master` and `.master_dynamic`. `J` is a plain
    vector of embedded jump operators, typically built with
    `jump_operators` — named to match `QuantumOptics.jl`'s own `master`/
    `master_dynamic` argument name (see "Dissipation" above); `rho0` may be
    a density operator or a state vector (QuantumOptics.jl converts
    automatically). `rates`, `Jdagger`, and any other solver keyword
    arguments are forwarded through `kwargs...`.
  - Which of the four underlying functions runs is therefore determined
    structurally: presence of a `J` argument selects open vs. closed, and
    the type of `H` selects time-dependent vs. not — not by any
    QuantumDynamics-level flag or branch.
- **Thin, but not a bare forward.** Each method forwards to the matching
  `QuantumOptics.timeevolution.*` function; `evolve` does no integration of
  its own (QuantumDynamics organizes *which* solver to call and *what*
  operators to hand it — see "Goal"). It adds two things: different solver
  defaults and a post-solve normalization check, both described below.
  Verified in tests by comparing `evolve`'s output against the corresponding
  `timeevolution.*` function called with the same solver options.
- **Solver defaults.** `evolve` defaults to `alg = OrdinaryDiffEqVerner.Vern9()`,
  `reltol = 1e-8`, `abstol = 1e-10` (the `DEFAULT_ALG` / `DEFAULT_RELTOL` /
  `DEFAULT_ABSTOL` constants in `evolution.jl`), overriding `QuantumOptics.jl`'s
  own `DP5()` / `1e-6` / `1e-8`. `DP5()` is a 5th-order pair; on a
  time-dependent `H` with a *sustained* fast oscillation (the second failure
  mode in "Two ways the adaptive solver misses a feature" below) it
  accumulates per-cycle phase error into an O(1) bias, and tightening its
  tolerance alone drives the step count into `OrdinaryDiffEq`'s `maxiters`
  wall (returning silently-garbage state). `Vern9()` is 9th-order with a
  matching high-order dense-output interpolant — well suited to the
  thousands-of-points `tspan` this framework is usually called with — and
  converges on those cases at comparable wall-clock cost. `Vern7()` is the
  swap for very large Hilbert spaces: 10 internal stages vs 16, so less state
  storage, at the cost of a lower-order interpolant. Pass any of
  `alg`/`reltol`/`abstol` explicitly to override.
- **Post-solve normalization check.** Unless called with
  `check_normalization=false`, `evolve` warns (once) if the final state's
  norm² (closed) or `tr` (open) has drifted from 1 by more than
  `NORMALIZATION_ATOL` (`1e-4`). A correct solve at the default tolerances
  leaves ~`1e-7` of drift; an under-resolved solve (see "Two ways the adaptive
  solver misses a feature" below) typically drifts orders of magnitude more,
  so the check turns a near-invisible accuracy loss into a visible one.
  Disable it for a run that is non-norm-preserving by design (e.g. an
  effective non-Hermitian Hamiltonian). A custom `fout` that returns something
  other than a state is not checkable and passes through silently.
- **Two ways the adaptive solver misses a feature in a time-dependent `H`.**
  An adaptive ODE solver evaluates the right-hand side at a few fixed
  fractional points within each attempted step `[t_n, t_n+h]` and compares two
  embedded solutions built from those points to estimate the step's error.
  Two distinct things can defeat that:
  - **A narrow kick.** If `h` has grown large over a long, smooth stretch and
    then spans a much narrower feature in a coefficient (e.g. a short pulse),
    none of the sample points may fall inside it at all — both embedded orders
    agree (on data that already missed the feature), the error estimate comes
    out small, and the step is accepted with no warning. This is a
    sampling/aliasing failure, not an imprecision one: tightening
    `reltol`/`abstol` doesn't help (the comparison never saw the feature), and
    neither does a finer `tspan` (`tspan` only controls the output, not where
    the solver steps). Raising the method order doesn't help either. The
    `Vern9` default does **not** fix this mode; guarding is the caller's
    responsibility (see the fixes below).
  - **A sustained fast oscillation.** A carrier or rotating-frame term that
    oscillates continuously across an extended interval (rather than a
    localized kick) is seen by every step, so it isn't aliased away — but a
    low-order method resolves each cycle only roughly, and the small
    per-cycle phase error accumulates over many cycles into an O(1) bias in
    the final state, again with no warning. Here order and tolerance *do*
    help, which is exactly why `evolve` defaults to `Vern9()` at
    `1e-8`/`1e-10`.

  Fixes for the narrow-kick mode, all forwarded straight through to the solver:
  - `tstops=<times>`: forces a step boundary at each listed time, so the step
    that follows starts inside the feature and the normal error estimate can
    see it. The most targeted and cheapest fix — step size elsewhere is
    untouched — but needs the feature's timing known in advance.
  - `dtmax=<bound>`: caps the adaptive step size everywhere in the run, not
    just near the feature. Since the bound has to be smaller than the feature
    itself, this forces small steps over long, smooth stretches too. It does
    still keep the normal error-based accept/reject/shrink logic running
    underneath the cap, unlike a truly fixed step.
  - `dt=<value>` with `adaptive=false`: fixes the step size for the whole run.
    Since `dt` must be smaller than the sharpest feature, this is expensive
    precisely where the dynamics are smooth — and, unlike `dtmax`, there is no
    error-based accept/reject/shrink check running at all, so nothing in the
    run would flag a `dt` that turns out to be too coarse somewhere. (`dt`
    alone, without `adaptive=false`, is only an initial-step guess and does
    not help.)
- **Stiff `H`.** For a stiff problem — typically an open-system run with
  widely separated rates — pass an auto-switching algorithm explicitly.
  `QuantumOptics.jl`'s state vector is complex, which its default
  `ForwardDiff` Jacobian can't handle, so the implicit half needs a
  finite-difference Jacobian:

  ```julia
  using OrdinaryDiffEqVerner, OrdinaryDiffEqRosenbrock, ADTypes
  evolve(tspan, rho0, H, J;
         alg = AutoVern7(Rodas5P(autodiff = AutoFiniteDiff())))
  ```

  This isn't the default: it pulls in the whole implicit-solver stack, and on
  a non-stiff oscillatory problem the switcher just adds overhead.
- Composing dissipation with `evolve` is a two-step, explicit pipeline
  matching the rest of the framework's "produce plain operators, hand them
  to `QuantumOptics.jl`" philosophy — no hidden `cs`-threading inside
  `evolve` itself:
  ```julia
  J = jump_operators(cs, [Decay(:cavity, κ), Decay(:qubit, γ1), Dephasing(:qubit, γφ)])
  tout, rhos = evolve(tspan, psi0, H, J)
  ```
- A zero-dissipation open-system run (`evolve(tspan, psi0, H, Operator[])`)
  reproduces `dm(evolve(tspan, psi0, H)[2][end])` up to ODE-solver tolerance
  (not exact — the two use independent adaptive integrators; the test allows
  `1e-4`) — checked in tests as a closed/open consistency regression.

## Measurement (`src/measurement.jl`)

- `ptrace(cs::CompositeSystem, rho, name::Symbol)` — partial trace of a
  density operator over subsystem `name`, looked up via `cs.index` rather
  than requiring the caller to know Hilbert-space index order. Adds a
  method to `QuantumOptics.jl`'s own `ptrace`, same pattern as `embed`
  above. Single-subsystem only — no multi-name overload, since nothing in
  the framework needs to trace out more than one subsystem at a time yet;
  add one if/when that need shows up.
- `condition_on(cs::CompositeSystem, state, name, projector)` — project
  `state` (a `Ket` or density operator) onto a *local* `projector` on
  subsystem `name`, trace `name` out, and renormalize what's left, returning
  `(ρ_cond, prob)`. This is the "measure subsystem `name`, keep the rest of
  the system conditioned on the outcome" pattern (heralded state
  preparation, readout conditioning) — built from `embed` + `ptrace` + `tr`,
  nothing solver- or Hamiltonian-specific. `projector` stays entirely
  caller-supplied (e.g. `projector(spinup(qb) + spindown(qb) |> normalize)`
  for a specific measurement basis) — `condition_on` only owns the
  embed/trace/renormalize mechanics, not which local state or basis a given
  protocol measures in, keeping the same "framework owns the general
  recipe, caller owns the physics-specific choice" split used throughout
  `hamiltonians.jl` and `dissipation.jl`. `prob` (the measurement
  probability for `projector`'s outcome, `tr` of the unnormalized `P*ρ*P'`
  before renormalization) is returned alongside `ρ_cond` rather than left
  for the caller to recompute separately — deliberately, since silently
  dividing by a near-zero `prob` would otherwise produce a `NaN`/`Inf`
  `ρ_cond` with no signal that the outcome was near-impossible.
- `check_fock_cutoff(cs, name, states; tol=1e-5, levels=3)` — sums
  population in the top `levels` Fock level(s) of subsystem `name`'s basis
  (`HarmonicOscillator`/`Transmon` only) across one or more `Ket` states,
  and throws `FockCutoffTooSmall` once that exceeds `tol`. Exists because an
  under-sized Fock cutoff (`nmax`) fails silently: it biases results (e.g. a
  fidelity undershooting its converged value) rather than erroring, and the
  bias isn't visible from the state itself unless something specifically
  looks at the truncation boundary.
  - **`Ket`-only, not density operators, deliberately:** meant as a
    closed-system pre-flight check (run once on a `schroedinger` trajectory
    before adding dissipation), not a general open-system convergence
    diagnostic. Under a bath (`Decay`/`Gain`/`thermal_bath`), real
    steady-state population near the boundary of an otherwise-adequate
    cutoff is expected physics, not a truncation artifact, and this
    function can't distinguish the two — rather than risk that false
    positive (or a `tol` loosened to compensate that then also hides
    genuine truncation), it simply refuses density operators.
  - **`levels` defaults to 3, not 1** — for one structural reason and one
    arbitrary margin, kept distinct. Structurally: a single top level is
    provably blind to truncation whenever the dynamics are
    parity-restricted and `nmax` has the "wrong" parity for it (e.g. a
    coupling built from `a²`/`ad²`, as in `Hcs`/`H1` in
    `examples/conditionally_squeezed_states/`, only connects Fock levels
    two apart, so from an even initial level every odd level — including,
    possibly, the single top level of an odd `nmax` — stays at exactly zero
    population regardless of how truncated the simulation actually is); 2
    consecutive levels covers both parities and closes that blind spot for
    any Δn=2 coupling. Beyond that floor, 3 is arbitrary: it is *not*
    derived from any particular Hamiltonian's leakage pattern, just a small
    margin against silently missing a marginal cutoff. A coupling that
    leaks differently (connects levels more than 2 apart, so the 2-level
    floor doesn't even close its parity blind spot) is not guaranteed to be
    caught by this default — picking `levels`/`tol` deliberately for your
    own Hamiltonian is the caller's responsibility.
  - Bounded by `levels <= (nmax+1)÷2` (checked, throws `ArgumentError`
    otherwise) so the checked region can't grow to swallow the whole basis
    on a small `nmax` and make the check vacuous (always ≈1 regardless of
    convergence) — a default `Transmon` (`nmax=4`) doesn't leave room even
    for `levels=3` and needs an explicit smaller value.
  - `FockCutoffTooSmall` (a distinct `Exception` subtype) is thrown only
    for "the cutoff itself is too small"; caller misuse (wrong basis type,
    a `levels` that doesn't fit `nmax`, a non-`Ket` state) still throws
    `ArgumentError`. Kept distinct so calling code — e.g. future
    parameter-sweep infrastructure that wants to retry with a larger
    `nmax` — can catch the "too small" case specifically instead of
    matching an error string.
- **Boundary this deliberately crosses:** `CompositeSystem` itself manages
  only operators (see "Composite systems" above) — no state-level API.
  `ptrace`/`condition_on` are the framework's first state-level functions,
  kept in their own file rather than folded into `composite.jl`, so that
  file's "operators only" scope stays accurate. Promoted from
  `examples/conditionally_squeezed_states/system.jl`'s own hand-rolled
  embed/ptrace/renormalize logic (originally raw `QuantumOptics.jl`) once
  validated there against both Fig. 1's and Fig. 4's conditional-measurement
  steps — see that example's `README.md`.

## Time-dependent Hamiltonians

- Built on QuantumOptics.jl's `TimeDependentSum(coeff1 => op1, coeff2 =>
  op2, ...; init_time=0.0)`, which both `schroedinger_dynamic` and
  `master_dynamic` accept directly (as `AbstractTimeDependentOperator`).
- **Design rule: coefficients carry the time dependence, operators stay
  fixed.** Every `op_i` should be one of the framework's cached, embedded
  operators (or a fixed sum/product of them, computed once); only `coeff_i`
  should be a function of `t`. Concretely:
  ```julia
  a, ad = op(cs, :cavity, :a), op(cs, :cavity, :ad)
  Htd = TimeDependentSum(1.0 => jaynes_cummings(cs, :qubit, :cavity, g),
                          (t -> Ω*cos(ωd*t)) => (a + ad))
  ```
  Rationale: `set_time!(Htd, t)` (called internally by the solver at every
  step) only re-evaluates the scalar coefficient functions and re-forms the
  weighted sum — it never rebuilds `a + ad` or re-embeds anything. Verified
  directly: inspecting `static_operator(Htd)` before and after `set_time!`
  shows the underlying operators are untouched; only the linear combination
  changes.
- Practical implication for anyone adding new time-dependent terms: always
  factor out any composite operator you need (e.g. `a + ad`, or a coupling
  term) into a fixed expression built from cached ops, and keep the
  time-dependent part to a bare scalar closure.
- `add_time_dependence(H, terms::Pair...; init_time=0.0)` packages the
  pattern above: `add_time_dependence(H, coeff1 => op1, ...)` is exactly
  `TimeDependentSum(1.0 => H, coeff1 => op1, ...; init_time)`. `H` can be any
  static Hamiltonian — recipe output or hand-assembled from cached ops (e.g.
  an interaction-frame Hamiltonian with no bare-energy term at all) — and any
  number of terms may be given, so several simultaneous time-dependent
  pieces on top of one static part are a single call. Every term's operator
  must already be resolved (e.g. via `op(cs, name, key)`, or a fixed
  sum/product of cached ops) — never rebuilt inside the coefficient closure.
  **Named `add_time_dependence`, not `add_drive`:** a term added this way
  need not be an external drive — it may equally be a term whose time
  dependence is purely an artifact of a rotating/interaction-frame
  transformation (a fast counter-rotating piece left over after moving to
  that frame), with no classical field behind it. `H1` in
  `examples/conditionally_squeezed_states/system.jl` adds both kinds in one
  call: two counter-rotating `a²σz`/`ad²σz` frame-artifact terms alongside a
  genuine `σx` drive term.

## Result persistence (`src/io.jl`)

- `SimulationResult` bundles a completed trajectory (`times`, `states`) with
  everything needed to make sense of it independently later: the structural
  setup that produced it (`subsystems`, `dissipators`), the Hamiltonian
  actually evolved (`H`), and free-form caller-supplied context
  (`description`, `params`). Built for the "long-running simulation now,
  post-processing later" workflow — construct one from `evolve`'s output,
  `save_result` it, and `load_result` it back in a completely separate
  session, potentially much later.
- **Stores `subsystems`/`dissipators`, not `cs`/`J` themselves — the same
  "cache vs. spec" distinction `CompositeSystem` already makes internally.**
  `CompositeSystem` eagerly embeds every subsystem operator into the joint
  Hilbert space at construction (see "Composite systems" above); those
  embedded operators are 100% reconstructible from the subsystem list alone
  (`CompositeSystem(subsystems...)`), so persisting them too would just be
  redundant — and, for a multi-subsystem composite system, potentially large
  — disk usage for data with zero information content beyond what
  `subsystems` already carries. The same argument applies to jump operators:
  `Vector{Dissipator}` is small and self-describing, and
  `jump_operators(cs, dissipators)` rebuilds `J` from it after `cs` is
  rebuilt. `CompositeSystem(r::SimulationResult)` is provided as the
  reconstruction step for `cs`; `J` reconstruction is left as the existing
  two-argument `jump_operators` call (no new function introduced for it) to
  keep the "produce plain operators, hand them off explicitly" pipeline
  style used throughout this codebase, rather than hiding it behind a
  single magic call.
- **`H`: no new Hamiltonian-recipe type introduced.** Unlike `Dissipator`,
  which reifies a dissipation channel as data before `jump_operator` turns
  it into a matrix, `jaynes_cummings`/`rabi`/etc. go straight from arguments
  to a matrix with nothing reified — so, unlike `subsystems`/`dissipators`,
  there's no automatic way to recover "which recipe, which arguments" from a
  saved `H` alone. Adding a parallel reified-Hamiltonian type (mirroring
  `Dissipator`) was considered and deliberately deferred: it would mean
  rewiring every existing Hamiltonian recipe's public shape for a need this
  feature doesn't strictly require — `params` already exists as the
  caller's place to record `g`, `Δ`, etc. structured for later use, the same
  role a reified type would otherwise formalize.
  - **Time-independent `H` round-trips exactly** (JLD2 handles a plain
    `Operator`/sparse matrix natively).
  - **Time-dependent `H` (`TimeDependentSum`) round-trips on a best-effort
    basis only, verified empirically (not assumed):** JLD2 serializes a
    coefficient function by name. A *named* top-level function
    (`mydrive(t) = ...`) reloads as a fully working callable — confirmed by
    round-tripping a real `TimeDependentSum` through JLD2 across two
    separate Julia processes and running `schroedinger_dynamic` on the
    reloaded object directly, producing an identical evolved state. An
    *anonymous closure* (`t -> Ω*cos(ωd*t)` — the pattern
    `add_time_dependence`'s own docstring example uses, and every
    time-dependent example in this repo uses) does **not**: JLD2 warns at
    save time ("only stores functions by name, may not be useful for
    anonymous functions"), and reloading in a fresh process throws
    `MethodError: ... are not callable` — a loud failure, not a silent one,
    consistent with this codebase's general preference (`condition_on`'s
    `prob`, `FockCutoffTooSmall`) for surfacing a problem rather than
    producing a quietly-wrong result. What *does* survive a closure reload,
    for a human debugging rather than code relying on it: if the closure
    captures function-local variables (not globals — globals aren't closed
    over as fields, so those closures reload as a bare, valueless
    singleton), the reconstructed object's type carries the enclosing
    function's name and the captured variable names/values (e.g.
    `Reconstruct@#make_drive##0#make_drive##1{Float64,Float64}((2.0,
    3.0))` — readably `make_drive`'s closure with `Ω=2.0, ωd=3.0`), visible
    via `show`/`repr`/error messages even though the object is not callable.
  - Because of this, the recommended pattern for a time-dependent run is:
    put drive parameters in `params`, leave `H` as `nothing` or a fixed
    reference operator, and rebuild the actual `TimeDependentSum` with
    [`add_time_dependence`](#hamiltonians-srchamiltoniansjl) from the
    caller's own code after `load_result` — the same "framework owns the
    general recipe, caller owns the physics-specific reconstruction" split
    used throughout `hamiltonians.jl`/`dissipation.jl`.
- **`states`/`times`: whatever `evolve` returned, saved as-is.** `states`
  may hold `Ket`s (closed system) or density `Operator`s (open system) —
  `SimulationResult` doesn't care which, both serialize the same way.
  Coarse-graining in time (e.g. `times[1:10:end]`, `states[1:10:end]`) is
  deliberately left to the caller as a two-line slice *before* constructing
  a `SimulationResult`, rather than a feature `save_result` implements —
  symmetric with the `H`/`dissipators` policy above of storing minimal,
  caller-controlled inputs rather than every possible derived convenience.
  **Saving a *reduced* trajectory (`ptrace`ing out a subsystem first) is
  not currently supported**, even though it's a two-line change at the
  call site: `subsystems` is always the full list `cs` was built from, so
  `CompositeSystem(r)`/`jump_operators` on reload would reconstruct
  full-dimension operators that no longer match ptrace'd states, with
  nothing in the file recording that a reduction happened. Left for later
  (a `retained_subsystems` field or similar) rather than solved now.
- **`description` vs. `params`:** `description` is a free-text hint for a
  human (e.g. which Hamiltonian recipe/frame `H` came from) — not parsed by
  any code here. `params` is where the same kind of information goes
  *structured*, for anything a later analysis script might want to read
  back programmatically (`r.params[:g]`) rather than just display.
- **`save_result` writes fields as separate top-level JLD2 keys
  (`times`, `states`, `subsystems`, `dissipators`, `H`, `description`,
  `params`, plus a `format_version`), not one serialized `SimulationResult`
  blob.** This is the same lesson the `TimeDependentSum` closure testing
  surfaced applied to `SimulationResult` itself: JLD2 reconstructing a whole
  struct whose shape has changed since the file was written hits the same
  "type doesn't exist in workspace, reconstructing" fallback documented for
  `H` above. Storing plain named keys means `load_result` can instead
  fall back explicitly on a missing key (e.g. a not-yet-implemented
  `expectations` field — see "Not yet implemented" below) when reading an
  older file, without depending on JLD2 faithfully reconstructing this
  package's own struct across versions of it. `format_version` is written
  on every save for the same forward-compatibility reason, so a future
  `load_result` can branch on it explicitly if the schema ever changes
  shape rather than just grows — not yet used for anything beyond being
  recorded, since there is only one schema version so far.

## Testing (`test/runtests.jl`)

Run via `Pkg.test()` (uses `LinearAlgebra`, `Logging`, and `Test` as declared
in `Project.toml`'s `[extras]`/`[targets]`). Tests are organized as:

- `Qubit` / `HarmonicOscillator` / `Transmon` — basic construction and
  operator sanity (dimensions, `n` diagonal values; for `Transmon`, `h0`'s
  anharmonic ladder against a hand-computed `k*ω + (α/2)*k*(k-1)`, including
  the `α=0` harmonic-oscillator limit).
- `CompositeSystem` — joint basis dimension, `getsubsystem` identity,
  embedded operator shape; duplicate subsystem names rejected with
  `ArgumentError` (both a two-name and a three-subsystem case).
- `HarmonicOscillator` — `nmax` boundary: `nmax=0` rejected (by
  `FockBasis` itself, not `HarmonicOscillator`'s own code — pinned down
  since neither `HarmonicOscillator` nor `Transmon` validate `nmax`
  themselves), `nmax=1` (smallest valid basis) behaves correctly.
- `embed`/`ptrace` by subsystem name — cross-checked against the raw
  `QuantumOptics.jl` calls they wrap (`embed(cs.basis, cs.index[name], ...)`,
  `ptrace(rho, cs.index[name])`).
- `condition_on` — a product-state case (projecting the qubit half of a
  known separable state reproduces the exact expected reduced oscillator
  state, with the returned `prob` matching the qubit's own overlap with the
  projector) plus, on an entangled state, a cross-check of the returned
  `prob` against an independent manual project-trace computation, and a
  probability-normalization check (`P(+) + P(-) ≈ 1`); plus the documented
  near-zero-probability case (a projector orthogonal to the qubit's actual
  state gives `prob == 0.0` exactly and `ρ_cond` comes back `NaN` — confirms
  `condition_on` really does leave that to the caller rather than silently
  guarding it, as the docstring promises).
- **Excitation number conservation** — the key regression test for the
  `n`/`σ` sign convention: `[n_total, H_JC] == 0` exactly, while
  `[n_total, H_Rabi] ≠ 0` (sanity control). Repeated with a `Transmon` in
  place of the `Qubit`, exercising the `_lower`/`_raise` dispatch path.
- `jaynes_cummings` vs. manual construction from cached ops — cross-check.
- Hermiticity of every Hamiltonian builder's output, for both a `Qubit` +
  cavity and a `Transmon` + cavity composite system.
- `tavis_cummings` — excitation-number conservation for both scalar and
  per-qubit coupling; confirms different couplings produce different `H`.
  Also checked with a **mixed** `(Qubit, Transmon)` tuple, proving the
  coupling dispatch resolves correctly per-element; and a `g`/`qubit_names`
  length-mismatch case (both directions) rejected with `ArgumentError`,
  scalar `g` exempted from the check.
- `dispersive_hamiltonian` vs. full JC spectrum at large detuning —
  numerical agreement within expected higher-order correction size; `Δ=0`
  rejected with `ArgumentError`.
- `TimeDependentSum` — coefficient evaluation at two different times
  matches hand-computed expected matrices.
- `add_time_dependence` — produces output identical to the equivalent
  hand-built `TimeDependentSum`, checked at two different times, both for a
  single term and for multiple simultaneous terms in one call.
- **Dissipators** — `jump_operator`/`jump_operators` produce the
  expected `sqrt(rate)*<op>` (`Decay`) / `sqrt(rate/2)*<op>` (`Dephasing`)
  matrices, dispatched correctly per subsystem type (`Qubit`,
  `HarmonicOscillator`, `Transmon`), including `Gain`'s `ad` dispatch for
  `Transmon` (previously only exercised for `HarmonicOscillator`).
- **`evolve` dispatch** — each of the four `evolve` methods (closed/open ×
  time-independent/time-dependent) produces output identical to calling the
  matching `timeevolution.*` function directly with the same arguments.
- **`evolve` physics** — a closed/open consistency check (zero-dissipation
  `master` run matches `dm` of the `schroedinger` run, to solver tolerance),
  plus known-solution checks: cavity photon-number decay `n₀*exp(-κt)`,
  qubit `T1` decay `exp(-γ1*t)`, and pure-dephasing coherence decay
  `exp(-γφ*t)` with populations unchanged — the last one is also the
  regression test for the `Dephasing` `rate/2` convention (see
  "Dissipation" above).
- **`SimulationResult` save/load round trip** — closed system (`Ket`
  states, no dissipators): saved/reloaded `times`/`states`/`H` match
  exactly, defaults (`dissipators`/`description`/`params`) come back empty,
  and `CompositeSystem(r)` reconstructs a `cs` whose subsystem frequencies
  and operators match the original. Open system (density-operator states,
  non-empty `dissipators`, `description`/`params` set): saved/reloaded
  states and metadata match, and `jump_operators(CompositeSystem(r),
  r.dissipators)` reconstructs `J` matching the original. Plus a check that
  `format_version` is actually written to the file.

## Dependencies

- `QuantumOptics.jl` — provides all Hilbert space, operator, and solver
  machinery.
- `OrdinaryDiffEqVerner` — for `Vern9()`, `evolve`'s default ODE algorithm
  (see "Solver wrappers"). Lightweight: explicit Runge-Kutta only, no implicit
  machinery.
- `JLD2.jl` — serialization backend for [`save_result`](#result-persistence-srcio-jl)/[`load_result`](#result-persistence-srcio-jl).
- `LinearAlgebra`, `Logging`, `Test` — test-only dependencies (`[extras]`/`[targets]`).

## Not yet implemented

- A transmon-specific dispersive approximation (`dispersive_hamiltonian` is
  currently qubit-only — see "Hamiltonians" above).
- Additional subsystem types beyond `Qubit`, `HarmonicOscillator`, and
  `Transmon`.
- Parameter-sweep / batch-run infrastructure.
- Quantum-trajectory (`timeevolution.mcwf*`) solver wrappers — `evolve`
  currently covers `schroedinger`/`schroedinger_dynamic`/`master`/
  `master_dynamic` only.
- Correlated/multi-operator dissipators (e.g. a `rates` matrix for
  cross-coupled decay channels) — `Decay`/`Dephasing` each produce one jump
  operator; a caller who needs a `rates` matrix must still call
  `evolve(...; rates=...)` directly with hand-built operators rather than
  through the `Dissipator` API.
- Saving expectation-value trajectories as an alternative to full states in
  `SimulationResult` — deliberately deferred until there's a concrete design
  for it (which observables, how they're named/stored), rather than adding
  an unpopulated `expectations` field now. `save_result`/`load_result`
  storing fields as separate named JLD2 keys (see "Result persistence"
  above) means this can be added later as a purely additive field: older
  files without it simply load with that field absent/`nothing`.
- Saving a *reduced* (`ptrace`'d) trajectory in `SimulationResult` — see
  "Result persistence" above. `subsystems` is always the full list, so
  reconstruction on load silently assumes states weren't reduced; needs a
  field recording which subsystems the saved states actually span before
  this is safe to support.
