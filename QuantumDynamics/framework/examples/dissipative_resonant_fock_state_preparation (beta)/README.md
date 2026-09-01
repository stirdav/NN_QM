# Dissipative resonant Fock-state preparation

This self-contained example prepares `|g,N⟩` by alternating qubit-drive
plateaus with resonant Jaynes–Cummings swaps while evolving a Lindblad master
equation. The qubit and bosonic mode each have independent energy-relaxation
and pure-dephasing channels.

The Hamiltonian is

```
H(t) = g*(ad*σm + a*σp) + Ω(t)*σx.
```

The collapse operators are

```
sqrt(γ1)*σm       qubit energy relaxation
sqrt(γ2/2)*σz     qubit pure dephasing
sqrt(κ1)*a        oscillator energy relaxation
sqrt(κ2/2)*n      oscillator number dephasing
```

All rates are expressed in the same units as `g`. With `g=1`, they are the
dimensionless ratios `γ1/g`, `γ2/g`, `κ1/g`, and `κ2/g`. `γ2` is the
qubit's pure-coherence decay rate. For the oscillator, the `|m⟩⟨n|` matrix
element decays at `κ2*(m-n)^2/4` from number dephasing.

The defaults are derived from the Ramsey values in Table S2 of the
supplementary material for Science 380, adf7553 (2023): `g0/2π=258 kHz`,
`γ1/2π=19 kHz`, total `γ2,Ramsey/2π=24 kHz`, `κ1/2π=2.5 kHz`, and
total `κ2,Ramsey/2π=1.5 kHz`. The script documents the conversion from the
reported total decoherence rates to the pure-dephasing coefficients required
by `Dephasing`.

## Running

From this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. population_ladder.jl
```

The script prints the final `|g,N⟩` population and saves
`population_ladder.png`. Edit `N`, `g`, `γ1`, `γ2`, `κ1`, and `κ2` near the
top of `population_ladder.jl` to select a target and noise model.
