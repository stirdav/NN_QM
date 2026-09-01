# Sequential Fock-state preparation

Prepares a target Fock state `|N⟩` in a bosonic mode by climbing the
Jaynes-Cummings ladder one excitation at a time: a qubit coupled to the mode
is driven with a π-pulse while parked off resonance, then swapped resonantly
into the mode via a vacuum-Rabi exchange. Repeating this `N` times takes
`|g,0⟩ → |g,N⟩`.

This is a special case of the general protocol proposed by Law & Eberly,
*Arbitrary Control of a Quantum Electromagnetic Field*, PRL (1996), which
assumes a drive amplitude and a coupling strength that can each be turned on
and off independently. In the platforms this example is modeled on, the
qubit-mode coupling strength `g` is fixed by the hardware rather than
tunable, so "turning the coupling on and off" is instead done by tuning the
*qubit* in and out of resonance with the mode (the `Δ(t)` plateaus below).
This has been implemented for circuit QED (a photonic microwave resonator)
by Hofheinz et al., *Generation of Fock states in a superconducting quantum
circuit*, Nature (2008), and for circuit QAD (a phononic
bulk acoustic-wave resonator) by Chu et al., *Creation and control of
multi-phonon Fock states in a bulk acoustic-wave resonator*, Nature (2018) —
this example models the mode generically enough to stand in for either.

It was chosen as a `QuantumDynamics` example because it's a natural fit for
the framework's existing pieces (`jaynes_cummings`, `add_time_dependence`,
`evolve`) and exercises them in a new way: a long, multi-segment pulse train
built as one continuous `TimeDependentSum`, rather than a single drive term.

## The protocol

Work in the frame co-rotating with the mode at its own frequency `ωr`.
Because the JC coupling term conserves total excitation number (the tested
invariant behind `hamiltonians.jl`'s `jaynes_cummings`), it's *exactly*
invariant under this rotation, so building the qubit and oscillator with
`ω=0` and calling `jaynes_cummings(cs, :qubit, :osc, g)` gives the coupling
term alone, with no residual fast oscillation to resolve. The qubit's bare
energy becomes a single time-dependent detuning `Δ(t) = ωq(t) - ωr`, added
on top via `add_time_dependence` — the "tunable qubit frequency" a real
device implements with flux control.

Each step `n → n+1` is two plateaus of `Δ(t)`, joined by smooth `tanh`
ramps: a **dispersive** plateau (`Δ=Δbig`) where a Gaussian-enveloped
π-pulse drives `|g,n⟩ → |e,n⟩`, followed by a **resonant** plateau (`Δ=0`)
held for the vacuum-Rabi swap time `π/(2g√(n+1))`, exchanging
`|e,n⟩ → |g,n+1⟩`. `system.jl` has the shared machinery: `build_system`,
the pulse-schedule builder (`build_schedule`), `detuning_fn`/`envelope_fn`,
and `run_fock_prep(N, hamiltonian_fn)`, the common entry point every script
uses to build a system, evolve `|g,0⟩` under a given Hamiltonian builder,
and return the trajectory along with the Δ(t) and drive-envelope functions
used to build it.

`Δ(t)` and `Ω(t)` are shaped differently on purpose. `Δ(t)` needs to *sit at*
two precise values (`Δbig`, `0`) for a calibrated duration — a `tanh`
top-hat gives flat plateaus, converged to the target value away from the
transition. `Ω(t)` has no plateau to hit; a π-pulse only needs a correct
*integrated area* (`∫Ω dt = π`), so it's shaped as a Gaussian instead —
smooth, no sidelobes, with a Fourier transform that's also a Gaussian. That
matters most for `selective_pulses.jl`: a flat-top pulse (even with smooth
edges) has a sinc-like spectrum whose sidelobes decay only as `~1/f`, which
would leak into neighboring number-split manifolds regardless of how gentle
the edges are — exactly the selectivity that scheme depends on. The Gaussian
shape is kept for `fixed_frequency.jl` too, mainly for a uniform, shared
`envelope_fn` and area-calibration formula, even though that scheme doesn't
rely on spectral purity the way the selective one does.

### Two addressing schemes

The two plateaus are shared, but *which frequency addresses the π-pulse*
comes in two flavors, each in its own file:

- **`selective_pulses.jl`** (number-resolved): each step's pulse is long
  and narrowband, individually tuned to that step's dispersive
  number-split transition frequency (`ωd = Δbig + (2n+1)χ`, derived from
  exact 2×2 block-diagonalization of the `{|e,n⟩,|g,n+1⟩}` pairs the JC
  coupling splits the Hilbert space into). *Benefit*: genuinely
  spectroscopically selective — only the intended manifold responds, even
  if population has leaked elsewhere. *Drawback*: needs a deep dispersive
  regime and long pulses to resolve the splitting, making the protocol
  slow (`∝ N·σt`).
- **`fixed_frequency.jl`** (the scheme both Hofheinz et al. and Chu et al.
  actually demonstrate): every step uses the *same* short, broadband pulse
  frequency — no
  per-step retuning. This works without spectroscopic selectivity because
  the sequence is open-loop and deterministic: when step `n`'s pulse
  fires, population only exists in `|g,n⟩`, so there's nothing else nearby
  for an untuned pulse to address. *Benefit*: much faster (no long pulses
  needed). *Drawback*: slightly lower fidelity than the selective scheme
  even in this noiseless simulation — a gap that widens with `N` (see
  `pulse_scheme_comparison.jl`) — and the determinism it relies on is
  exactly what a real device can't fully guarantee: any population left
  behind by an imperfect earlier swap has no frequency protection and gets
  hit by the next pulse anyway.

- `population_ladder.jl` — evolves the `N=3` protocol under both schemes and
  plots, for each, three rows on a linked time axis: Fock-level populations,
  `Δ(t)` itself, and the drive envelope `|Ω(t)|` (shaded regions mark the
  resonant swap windows throughout) → `population_ladder.png`. Same
  step-by-step climb and plateau structure in both columns, but on very
  different time and amplitude scales — the fixed-frequency column's x-axis
  is ~100× shorter and its pulses ~150× taller, each one visibly landing
  between the swap windows rather than overlapping them. The `Δ(t)` row also
  makes the asymmetry between the two regimes visible directly: in the
  number-selective column the resonant (`Δ=0`) windows are so brief relative
  to the long dispersive pulses that they barely register as dips, while in
  the fixed-frequency column they're a substantial fraction of the timeline.
- `pulse_scheme_comparison.jl` — runs both schemes over `N=1..5` and plots
  fidelity and total protocol duration side by side →
  `pulse_scheme_comparison.png`. At this operating point the fixed-frequency
  scheme is ~100-114× faster, with fidelity within ~1 percentage point of the
  number-selective scheme through `N=3` (0.974 vs 0.982) and ~2 points behind
  by `N=5` (0.916 vs 0.938).

## A note on the tuned constants

Both schemes' defaults (`Δbig/g=30`, `τramp/g=0.001`) are numerically-tuned
operating points, not derived analytically — see the comments above
`fock_ladder_hamiltonian_selective`/`_fixed` for what each one trades off.
`τramp` in particular needs to be small for good fidelity: each step uses
the exact ideal swap time `π/(2g√(n+1))`, with no correction factor, so a
ramp that isn't fast enough shows up directly as lost fidelity.

## Running

```sh
julia --project=. population_ladder.jl
julia --project=. pulse_scheme_comparison.jl
```

`Project.toml` points `QuantumDynamics` at the package root via Julia's
`[sources]` table, so no separate install/dev step is needed.
