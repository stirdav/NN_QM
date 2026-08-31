# Resonant-coupling Fock-state preparation

Prepares a target Fock state `|N⟩` by climbing the Jaynes-Cummings ladder one
excitation at a time, like `examples/fock_state_preparation/` — but with the
qubit and oscillator **always resonantly coupled**, rather than tuned in and
out of resonance via a time-dependent detuning `Δ(t)`.

This is a variant, not a replacement: the sibling example implements the
*hardware-realistic* workaround for platforms where the coupling strength
`g` is fixed and only the qubit frequency is tunable (see its README for the
Hofheinz et al./Chu et al. references). This example instead keeps `g` fixed
*and* keeps the qubit on resonance throughout — closer to Law & Eberly's
original assumption of an independently gateable drive, except the coupling
here is never gated either. That's a step further from the general protocol
than the sibling example takes, so its practical performance has to be
checked rather than assumed — see "Validity" below.

## The protocol

Same co-rotating frame trick as the sibling example: build both subsystems
with `ω=0` so `jaynes_cummings(cs, :qubit, :osc, g)` is the coupling term
alone, with no residual fast oscillation. Because `Δ=0` always, that
coupling term is a genuinely time-independent operator this time — nothing
turns it on or off. The only time-dependent piece is a qubit drive:

```
H(t) = g·(ad·σm + a·σp)   +   Ω(t)·σx
```

Each step `n → n+1` is a **drive plateau** (`Ω(t)=Ω0`, flipping `|g,n⟩ →
|e,n⟩`) followed by a **swap window** of duration `π/(2g√(n+1))`, during
which `Ω=0` and the always-on coupling performs the vacuum-Rabi exchange
`|e,n⟩ → |g,n+1⟩` — the same swap-time formula as the sibling example, since
that piece of the physics doesn't change.

Two things are simpler here than in the sibling example, both because there
is no detuning at all:

- **The drive needs no carrier frequency.** A resonant drive on the qubit is
  a bare `σx` coefficient in this frame, not `cos(ωd·t)·σx` — there's no
  `Δ≠0` to compensate for.
- **The drive needs no per-step frequency tuning, and only one addressing
  scheme.** `Ω(t)·σx` flips the qubit the same way regardless of the
  oscillator's state, so unlike `selective_pulses.jl`/`fixed_frequency.jl`
  there's no transition frequency to resolve or protect — nothing for a
  pulse to accidentally address instead of the intended one.

## Pulse shape: tanh top-hat, not Gaussian

The sibling example shapes its drive as a Gaussian specifically for spectral
purity — `selective_pulses.jl` needs a pulse with no sidelobes so it doesn't
leak into neighboring number-split manifolds. That requirement doesn't exist
here (see above), so the drive is shaped as a smooth top-hat instead
(`tophat` in `system.jl`, the same tanh-edged shape convention used for
`Δ(t)` in the sibling example): flat at `Ω0` for a plateau width `w`, with
tanh edges of width `τ`.

This shape's calibration is exact and, unlike the Gaussian's, independent of
the edge width: `∫ tophat(t,t0,t1,τ) dt = t1-t0` for any `τ`, because
`smoothstep(t,t0,τ) - Θ(t-t0)` (its deviation from a hard step) is odd in
`t-t0` and integrates to zero.

The drive term is the bare `H = Ω(t)·σx`, which generates a Bloch-sphere
rotation angle of `2·∫Ω dt`, so a full `|g⟩→|e⟩` flip needs `∫Ω dt = π/2`:

```
Ω0 = π / (2w)
```

## Validity: why "always coupled" can still work well

Sitting in `|g,n⟩` during step `n`'s drive plateau, the always-on coupling
doesn't just wait quietly: `a·σp` connects `|g,n⟩` to `|e,n-1⟩` with matrix
element `g√n` — the reverse of the swap that produced `|g,n⟩` in the first
place. That's a stray coupling competing with the intended `Ω(t)·σx`
rotation for the whole duration of the drive plateau.

It stays negligible as long as the plateau is fast compared to the local
vacuum-Rabi period it's competing against, not just short compared to its
own `swap_time(n,g)`:

```
g·√N·w ≪ 1        (worst case at the last step, n = N-1)
```

`leakage_scan.jl` sweeps `w` at fixed `N` and plots final-state fidelity
against this exact combination, confirming fidelity stays high while
`g√N·w ≲ 1` and degrades once it isn't → `leakage_scan.png`.

- `population_ladder.jl` — evolves the `N=3` protocol and plots Fock-level
  populations and the drive envelope `Ω(t)` on a linked time axis (swap
  windows shaded) → `population_ladder.png`.

## Running

```sh
julia --project=. population_ladder.jl
julia --project=. leakage_scan.jl
```

`Project.toml` points `QuantumDynamics` at the package root via Julia's
`[sources]` table, so no separate install/dev step is needed.
