using Test
using QuantumOptics
using LinearAlgebra
using Logging
using JLD2
using QuantumDynamics

# Must be a genuine top-level (module-global) function, not one defined
# inside a @testset block — JLD2 serializes a TimeDependentSum coefficient
# function by qualified name (module + name) and looks it up the same way on
# load, so it only survives a round trip if that name is actually resolvable
# as a global binding. See the "SimulationResult" testset below.
_sr_test_drive_coeff(t) = 0.4 * cos(1.1 * t)

@testset "QuantumDynamics" begin

    @testset "Qubit" begin
        q = Qubit(:qubit, 5.0)
        @test q.ω == 5.0
        # spin-up (σz=+1) is the excited state under QuantumOptics.jl's sigmam/sigmap
        # convention, so n must align with +σz, not -σz.
        @test diag(dense(op(q, :n)).data) ≈ [1.0, 0.0]
        # h0 = ω*n, cached alongside n so bare_hamiltonian can sum it uniformly.
        @test diag(dense(op(q, :h0)).data) ≈ [5.0, 0.0]
    end

    @testset "HarmonicOscillator" begin
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        @test size(dense(op(c, :a)).data) == (5, 5)
        @test diag(dense(op(c, :n)).data) ≈ collect(0.0:4.0)
    end

    @testset "HarmonicOscillator: nmax boundary" begin
        # nmax=0 would mean a single-state Fock space, but QuantumOptics.jl's
        # FockBasis itself requires cutoff > offset (offset defaults to 0),
        # so it rejects this before any QuantumDynamics code runs — worth
        # pinning down since HarmonicOscillator/Transmon do no nmax
        # validation of their own and just trust the underlying FockBasis.
        @test_throws ArgumentError HarmonicOscillator(:cavity, 6.0; nmax=0)

        # nmax=1: the smallest valid (two-level, Fock states 0:1) basis —
        # a/ad/n still need to behave correctly at that boundary.
        c1 = HarmonicOscillator(:cavity, 6.0; nmax=1)
        @test size(dense(op(c1, :a)).data) == (2, 2)
        @test diag(dense(op(c1, :n)).data) ≈ [0.0, 1.0]

        q = Qubit(:qubit, 5.0)
        cs1 = CompositeSystem(q, c1)
        @test size(dense(op(cs1, :cavity, :n)).data) == (4, 4)
    end

    @testset "Transmon" begin
        t = Transmon(:transmon, 5.0, -0.2; nmax=4)
        @test t.ω == 5.0
        @test t.α == -0.2
        @test diag(dense(op(t, :n)).data) ≈ collect(0.0:4.0)

        # h0 = ω*n + (α/2)*n*(n-1): anharmonic ladder E_k = k*ω + (α/2)*k*(k-1)
        expected_h0 = [k * 5.0 + (-0.2 / 2) * k * (k - 1) for k in 0:4]
        @test diag(dense(op(t, :h0)).data) ≈ expected_h0

        # α=0 recovers a plain harmonic oscillator's bare energy ladder
        t0 = Transmon(:t0, 5.0, 0.0; nmax=4)
        @test diag(dense(op(t0, :h0)).data) ≈ collect(0.0:4.0) .* 5.0
    end

    @testset "CompositeSystem" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)

        @test getsubsystem(cs, :qubit) === q
        @test getsubsystem(cs, :cavity) === c
        @test size(dense(op(cs, :qubit, :n)).data) == (12, 12)
    end

    @testset "CompositeSystem: duplicate subsystem names rejected" begin
        # Two subsystems sharing a name would otherwise silently collide in
        # `index`/`ops` (last one wins), leaving the first permanently
        # unreachable by name despite still occupying a factor of the joint
        # Hilbert space — must throw instead of silently shadowing.
        q1 = Qubit(:qubit, 5.0)
        q2 = Qubit(:qubit, 6.0)
        @test_throws ArgumentError CompositeSystem(q1, q2)

        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        @test_throws ArgumentError CompositeSystem(q1, c, q2)
    end

    @testset "embed/ptrace by subsystem name" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)

        # embed(cs, name, local_op) matches the pre-embedded cached op for the
        # same local operator (composite.jl's constructor embeds identically).
        @test dense(embed(cs, :qubit, op(q, :σx))).data ≈ dense(op(cs, :qubit, :σx)).data

        ψ = normalize(spinup(q.basis) + spindown(q.basis)) ⊗ fockstate(c.basis, 0)
        ρ = dm(ψ)
        @test ptrace(cs, ρ, :cavity).data ≈ ptrace(ρ, cs.index[:cavity]).data
    end

    @testset "condition_on: project + trace + renormalize" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)

        # (|↑⟩+|↓⟩)/√2 ⊗ |0⟩: the qubit is an equal superposition, so
        # projecting onto |↑⟩ has probability 0.5 — but since *both* branches
        # leave the cavity in |0⟩⟨0|, the *reduced* cavity state is
        # deterministic regardless of the (probabilistic) qubit outcome.
        ψ = normalize(spinup(q.basis) + spindown(q.basis)) ⊗ fockstate(c.basis, 0)
        ρ_cond, prob = condition_on(cs, ψ, :qubit, projector(spinup(q.basis)))
        @test dense(ρ_cond).data ≈ dense(dm(fockstate(c.basis, 0))).data
        @test prob ≈ 0.5

        # Works on a density operator input too; returned probabilities match
        # a manual, unnormalized project-trace computation, and the ± outcome
        # probabilities of an entangled state sum to 1.
        ψ_entangled = normalize(spinup(q.basis) ⊗ fockstate(c.basis, 0) + spindown(q.basis) ⊗ fockstate(c.basis, 1))
        ρ_entangled = dm(ψ_entangled)
        P_up = embed(cs, :qubit, projector(spinup(q.basis)))
        P_down = embed(cs, :qubit, projector(spindown(q.basis)))
        p_up_manual = real(tr(P_up * ρ_entangled * P_up'))
        p_down_manual = real(tr(P_down * ρ_entangled * P_down'))
        _, p_up = condition_on(cs, ρ_entangled, :qubit, projector(spinup(q.basis)))
        _, p_down = condition_on(cs, ρ_entangled, :qubit, projector(spindown(q.basis)))
        @test p_up ≈ p_up_manual
        @test p_down ≈ p_down_manual
        @test p_up + p_down ≈ 1.0

        # Documented near-zero-probability behavior: condition_on does *not*
        # guard against dividing by prob≈0 itself (that's why prob is
        # returned — so the caller can check it before trusting ρ_cond). A
        # projector orthogonal to the qubit's actual (pure, not superposed)
        # state gives prob exactly 0, and ρ_cond becomes NaN (0/0) rather
        # than an error.
        ψ_pure = spinup(q.basis) ⊗ fockstate(c.basis, 0)
        ρ_cond_zero, prob_zero = condition_on(cs, ψ_pure, :qubit, projector(spindown(q.basis)))
        @test prob_zero == 0.0
        @test all(isnan, dense(ρ_cond_zero).data)
    end

    @testset "check_fock_cutoff: catches truncated Fock population" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=20)
        cs = CompositeSystem(q, c)

        # All population at the cutoff: must throw regardless of trajectory
        # position. Density operators are rejected outright (Ket-only).
        ψ_top = spinup(q.basis) ⊗ fockstate(c.basis, 20)
        @test_throws FockCutoffTooSmall check_fock_cutoff(cs, :cavity, ψ_top)
        @test_throws ArgumentError check_fock_cutoff(cs, :cavity, dm(ψ_top))
        ψ_ok = spinup(q.basis) ⊗ fockstate(c.basis, 0)
        @test_throws FockCutoffTooSmall check_fock_cutoff(cs, :cavity, [ψ_ok, ψ_ok, ψ_top])

        # All population away from the cutoff: must pass silently.
        @test check_fock_cutoff(cs, :cavity, ψ_ok) === nothing
        @test check_fock_cutoff(cs, :cavity, [ψ_ok, ψ_ok, ψ_ok]) === nothing

        # Not a FockBasis subsystem, or a nonsensical `levels`: argument errors.
        @test_throws ArgumentError check_fock_cutoff(cs, :qubit, ψ_ok)
        @test_throws ArgumentError check_fock_cutoff(cs, :cavity, ψ_ok; levels=0)
        @test_throws ArgumentError check_fock_cutoff(cs, :cavity, ψ_ok; levels=15)

        # Parity blind spot: a ±2-step coupling (e.g. Hcs/H1 in
        # examples/conditionally_squeezed_states/) starting from an even Fock
        # level leaves every odd level at exactly zero population — so a
        # levels=1 check on an odd nmax is blind to real truncation, however
        # bad. The >=2 default must still catch it.
        c_odd = HarmonicOscillator(:cavity, 6.0; nmax=21)
        cs_odd = CompositeSystem(q, c_odd)
        ψ_evenmax = spinup(q.basis) ⊗ fockstate(c_odd.basis, 20)  # top even level populated; n=21 always 0
        @test real(expect(embed(cs_odd, :cavity, projector(fockstate(c_odd.basis, 21))), ψ_evenmax)) == 0.0
        @test_throws FockCutoffTooSmall check_fock_cutoff(cs_odd, :cavity, ψ_evenmax)      # default levels=3 catches it
        @test check_fock_cutoff(cs_odd, :cavity, ψ_evenmax; levels=1) === nothing          # levels=1 alone is blind here

        # A basis too small for the default `levels` makes "population near
        # the boundary" meaningless (the checked region would be most/all of
        # the Hilbert space, so it's always ≈1 regardless of convergence) —
        # must fail loudly with ArgumentError rather than silently
        # always-failing. Mirrors a default Transmon (nmax=4).
        c_small = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs_small = CompositeSystem(q, c_small)
        @test_throws ArgumentError check_fock_cutoff(cs_small, :cavity, spinup(q.basis) ⊗ fockstate(c_small.basis, 0))
    end

    @testset "Excitation number conservation (regression: n/σ sign convention)" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)
        g = 0.1

        Hjc = jaynes_cummings(cs, :qubit, :cavity, g)
        Ntot = op(cs, :cavity, :n) + op(cs, :qubit, :n)
        comm_jc = dense(Ntot * Hjc - Hjc * Ntot).data
        @test maximum(abs.(comm_jc)) < 1e-10

        # Sanity check: the full (non-RWA) Rabi model should *not* conserve
        # excitation number, otherwise the test above would be vacuous.
        Hrabi = rabi(cs, :qubit, :cavity, g)
        comm_rabi = dense(Ntot * Hrabi - Hrabi * Ntot).data
        @test maximum(abs.(comm_rabi)) > 1e-3
    end

    @testset "Excitation number conservation (Transmon + cavity)" begin
        t = Transmon(:transmon, 5.0, -0.2; nmax=4)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(t, c)
        g = 0.1

        Hjc = jaynes_cummings(cs, :transmon, :cavity, g)
        Ntot = op(cs, :cavity, :n) + op(cs, :transmon, :n)
        comm_jc = dense(Ntot * Hjc - Hjc * Ntot).data
        @test maximum(abs.(comm_jc)) < 1e-10

        Hrabi = rabi(cs, :transmon, :cavity, g)
        comm_rabi = dense(Ntot * Hrabi - Hrabi * Ntot).data
        @test maximum(abs.(comm_rabi)) > 1e-3
    end

    @testset "jaynes_cummings matches manual construction" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)
        g = 0.1

        Hjc = jaynes_cummings(cs, :qubit, :cavity, g)
        H_manual = bare_hamiltonian(cs) +
                   g * (op(cs, :cavity, :ad) * op(cs, :qubit, :σm) +
                        op(cs, :cavity, :a) * op(cs, :qubit, :σp))
        @test dense(Hjc).data ≈ dense(H_manual).data
    end

    @testset "Hamiltonians are Hermitian" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)
        g = 0.1

        for H in (bare_hamiltonian(cs),
                  jaynes_cummings(cs, :qubit, :cavity, g),
                  rabi(cs, :qubit, :cavity, g),
                  dispersive_hamiltonian(cs, :qubit, :cavity, g),
                  quadratic_coupling(cs, :qubit, :cavity, g))
            @test ishermitian(dense(H).data)
        end
    end

    @testset "Hamiltonians are Hermitian (Transmon + cavity)" begin
        t = Transmon(:transmon, 5.0, -0.2; nmax=4)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(t, c)
        g = 0.1

        for H in (bare_hamiltonian(cs),
                  jaynes_cummings(cs, :transmon, :cavity, g),
                  rabi(cs, :transmon, :cavity, g))
            @test ishermitian(dense(H).data)
        end
    end

    @testset "tavis_cummings multi-qubit" begin
        q1 = Qubit(:q1, 5.0)
        q2 = Qubit(:q2, 5.2)
        cav = HarmonicOscillator(:cav, 6.0; nmax=4)
        cs = CompositeSystem(q1, q2, cav)

        Htc = tavis_cummings(cs, (:q1, :q2), :cav, 0.1)
        Ntot = op(cs, :cav, :n) + op(cs, :q1, :n) + op(cs, :q2, :n)
        @test maximum(abs.(dense(Ntot * Htc - Htc * Ntot).data)) < 1e-10
        @test ishermitian(dense(Htc).data)

        Htc_indiv = tavis_cummings(cs, (:q1, :q2), :cav, (0.1, 0.15))
        @test !isapprox(dense(Htc).data, dense(Htc_indiv).data)
    end

    @testset "tavis_cummings: g/qubit_names length mismatch rejected" begin
        # zip(qubit_names, g) would otherwise silently truncate to the
        # shorter of the two, dropping trailing qubits (or g's) from the
        # coupling with no error.
        q1 = Qubit(:q1, 5.0)
        q2 = Qubit(:q2, 5.2)
        q3 = Qubit(:q3, 5.4)
        cav = HarmonicOscillator(:cav, 6.0; nmax=4)
        cs = CompositeSystem(q1, q2, q3, cav)

        @test_throws ArgumentError tavis_cummings(cs, (:q1, :q2, :q3), :cav, (0.1, 0.15))
        @test_throws ArgumentError tavis_cummings(cs, (:q1, :q2), :cav, (0.1, 0.15, 0.2))
        # A scalar g is exempt (broadcast to every qubit, no length to check).
        @test tavis_cummings(cs, (:q1, :q2, :q3), :cav, 0.1) isa QuantumOptics.Operator
    end

    @testset "tavis_cummings mixed Qubit/Transmon" begin
        q = Qubit(:q1, 5.0)
        t = Transmon(:q2, 5.2, -0.2; nmax=4)
        cav = HarmonicOscillator(:cav, 6.0; nmax=4)
        cs = CompositeSystem(q, t, cav)

        Htc = tavis_cummings(cs, (:q1, :q2), :cav, 0.1)
        Ntot = op(cs, :cav, :n) + op(cs, :q1, :n) + op(cs, :q2, :n)
        @test maximum(abs.(dense(Ntot * Htc - Htc * Ntot).data)) < 1e-10
        @test ishermitian(dense(Htc).data)
    end

    @testset "quadratic_coupling" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)
        g = 0.1

        Hqc = quadratic_coupling(cs, :qubit, :cavity, g)
        a, ad = op(cs, :cavity, :a), op(cs, :cavity, :ad)
        σz = op(cs, :qubit, :σz)
        H_manual = bare_hamiltonian(cs) + g * (a + ad)^2 * σz
        @test dense(Hqc).data ≈ dense(H_manual).data

        # Quadratic coupling is not qubit-like ladder coupling and does not
        # conserve total excitation number (sanity control, same role as
        # rabi's non-conservation against the JC conservation test).
        Ntot = op(cs, :cavity, :n) + op(cs, :qubit, :n)
        comm = dense(Ntot * Hqc - Hqc * Ntot).data
        @test maximum(abs.(comm)) > 1e-3
    end

    @testset "dispersive_hamiltonian matches full JC at large detuning" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 8.0; nmax=5)   # Δ=3, g=0.05
        cs = CompositeSystem(q, c)
        g = 0.05

        Hfull = jaynes_cummings(cs, :qubit, :cavity, g)
        Hdisp = dispersive_hamiltonian(cs, :qubit, :cavity, g)

        evals_full = sort(real(eigvals(dense(Hfull).data)))
        evals_disp = sort(real(eigvals(dense(Hdisp).data)))
        @test maximum(abs.(evals_full .- evals_disp)) < 0.01
    end

    @testset "dispersive_hamiltonian: Δ=0 rejected" begin
        # On resonance the dispersive approximation is invalid (that's the
        # resonant JC regime); χ = g²/Δ would otherwise silently blow up to
        # Inf/NaN instead of signaling the approximation doesn't apply.
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 5.0; nmax=5)  # Δ=0
        cs = CompositeSystem(q, c)
        @test_throws ArgumentError dispersive_hamiltonian(cs, :qubit, :cavity, 0.05)
    end

    @testset "Time-dependent Hamiltonian (TimeDependentSum)" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)
        g = 0.1

        H = jaynes_cummings(cs, :qubit, :cavity, g)
        a, ad = op(cs, :cavity, :a), op(cs, :cavity, :ad)
        drive = a + ad
        Htd = TimeDependentSum(1.0 => H, (t -> 0.3 * cos(2.0 * t)) => drive)

        set_time!(Htd, 0.0)
        expected0 = dense(H).data + 0.3 * cos(0.0) * dense(drive).data
        @test dense(static_operator(Htd)).data ≈ expected0

        set_time!(Htd, pi / 4)
        expected1 = dense(H).data + 0.3 * cos(2.0 * pi / 4) * dense(drive).data
        @test dense(static_operator(Htd)).data ≈ expected1
    end

    @testset "add_time_dependence" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=5)
        cs = CompositeSystem(q, c)
        g = 0.1

        H = jaynes_cummings(cs, :qubit, :cavity, g)
        a, ad = op(cs, :cavity, :a), op(cs, :cavity, :ad)
        drive = a + ad
        coeff = t -> 0.3 * cos(2.0 * t)

        Htd_manual = TimeDependentSum(1.0 => H, coeff => drive)
        Htd_operator = add_time_dependence(H, coeff => drive)

        # matches the hand-built TimeDependentSum
        for t in (0.0, pi / 4)
            set_time!(Htd_manual, t)
            set_time!(Htd_operator, t)
            @test dense(static_operator(Htd_operator)).data ≈ dense(static_operator(Htd_manual)).data
        end

        # Multiple simultaneous time-dependent terms in one call (design.md:
        # several terms on top of one static part, not just a single drive).
        σx = op(cs, :qubit, :σx)
        coeff2 = t -> 0.2 * sin(0.5 * t)
        Htd_multi_manual = TimeDependentSum(1.0 => H, coeff => drive, coeff2 => σx)
        Htd_multi = add_time_dependence(H, coeff => drive, coeff2 => σx)
        for t in (0.0, 1.3)
            set_time!(Htd_multi_manual, t)
            set_time!(Htd_multi, t)
            @test dense(static_operator(Htd_multi)).data ≈ dense(static_operator(Htd_multi_manual)).data
        end
    end

    @testset "Dissipators reject negative rates at construction" begin
        @test_throws ArgumentError Decay(:cavity, -0.1)
        @test_throws ArgumentError Dephasing(:qubit, -0.1)
        # Float64 literals must hit the same validating constructor as
        # generic Real args, not Julia's auto-generated default inner
        # constructor for the (Symbol, Float64) field types.
        @test_throws ArgumentError Decay(:cavity, -0.1::Float64)
        @test_throws ArgumentError Dephasing(:qubit, -0.1::Float64)

        @test_throws ArgumentError Decay(:cavity, -0.1; nth=1.0)
        @test_throws ArgumentError Decay(:cavity, 0.1; nth=-1.0)
        @test_throws ArgumentError Gain(:cavity, -0.1; nth=1.0)
        @test_throws ArgumentError Gain(:cavity, 0.1; nth=-1.0)
        # nth has no default for Gain (unlike Decay), but nth=0 is allowed
        # (symmetric with Decay) — it just gives a zero jump operator, so a
        # temperature sweep down to nth=0 doesn't need to special-case Gain.
        @test_throws UndefKeywordError Gain(:cavity, 0.1)
    end

    @testset "Dissipators build correctly scaled, dispatched jump operators" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs = CompositeSystem(q, c)

        # Decay: σm for Qubit, a for HarmonicOscillator/Transmon
        @test dense(jump_operator(cs, Decay(:cavity, 0.3))).data ≈
              sqrt(0.3) * dense(op(cs, :cavity, :a)).data
        @test dense(jump_operator(cs, Decay(:qubit, 0.4))).data ≈
              sqrt(0.4) * dense(op(cs, :qubit, :σm)).data

        # Dephasing: σz for Qubit, n for HarmonicOscillator/Transmon; note the
        # rate/2 factor (see Dephasing docstring) so `rate` is directly the
        # coherence decay rate.
        @test dense(jump_operator(cs, Dephasing(:qubit, 0.05))).data ≈
              sqrt(0.05 / 2) * dense(op(cs, :qubit, :σz)).data
        @test dense(jump_operator(cs, Dephasing(:cavity, 0.05))).data ≈
              sqrt(0.05 / 2) * dense(op(cs, :cavity, :n)).data

        t = Transmon(:transmon, 5.0, -0.2; nmax=4)
        cs2 = CompositeSystem(t, c)
        @test dense(jump_operator(cs2, Decay(:transmon, 0.1))).data ≈
              sqrt(0.1) * dense(op(cs2, :transmon, :a)).data
        @test dense(jump_operator(cs2, Dephasing(:transmon, 0.1))).data ≈
              sqrt(0.1 / 2) * dense(op(cs2, :transmon, :n)).data

        J = jump_operators(cs, [Decay(:cavity, 0.3), Dephasing(:qubit, 0.05)])
        @test length(J) == 2
    end

    @testset "Decay nth and Gain thermal-bath scaling" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs = CompositeSystem(q, c)

        # Decay(nth=0) (default) reproduces exactly the original single-op
        # behavior: sqrt(rate*(0+1)) == sqrt(rate).
        @test dense(jump_operator(cs, Decay(:cavity, 0.3))).data ≈
              sqrt(0.3) * dense(op(cs, :cavity, :a)).data

        # Decay(nth>0): sqrt(rate*(nth+1))
        @test dense(jump_operator(cs, Decay(:cavity, 0.3; nth=0.5))).data ≈
              sqrt(0.3 * 1.5) * dense(op(cs, :cavity, :a)).data

        # Gain: sqrt(rate*nth) * raise (ad for oscillator, σp for qubit)
        @test dense(jump_operator(cs, Gain(:cavity, 0.3; nth=0.5))).data ≈
              sqrt(0.3 * 0.5) * dense(op(cs, :cavity, :ad)).data
        @test dense(jump_operator(cs, Gain(:qubit, 0.4; nth=2.0))).data ≈
              sqrt(0.4 * 2.0) * dense(op(cs, :qubit, :σp)).data

        J = jump_operators(cs, [Decay(:cavity, 0.3; nth=0.5), Gain(:cavity, 0.3; nth=0.5)])
        @test length(J) == 2

        # Gain(nth=0) is allowed (symmetric with Decay's default) and gives
        # a zero jump operator, so temperature sweeps down to nth=0 don't
        # need to special-case omitting Gain.
        @test all(iszero, dense(jump_operator(cs, Gain(:cavity, 0.3; nth=0.0))).data)

        # Gain dispatches to `ad` for a Transmon too (the _gain_op union
        # branch covers HarmonicOscillator *and* Transmon, but only the
        # HarmonicOscillator side was exercised above).
        t = Transmon(:transmon, 5.0, -0.2; nmax=4)
        cs_t = CompositeSystem(q, t)
        @test dense(jump_operator(cs_t, Gain(:transmon, 0.3; nth=0.5))).data ≈
              sqrt(0.3 * 0.5) * dense(op(cs_t, :transmon, :ad)).data
    end

    @testset "thermal_bath builds a matched Decay/Gain pair" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs = CompositeSystem(q, c)

        pair = thermal_bath(:cavity, 0.3, 0.5)
        @test length(pair) == 2
        @test pair[1] isa Decay && pair[2] isa Gain
        @test pair[1].rate == pair[2].rate == 0.3
        @test pair[1].nth == pair[2].nth == 0.5

        J1 = jump_operators(cs, thermal_bath(:cavity, 0.3, 0.5))
        J2 = jump_operators(cs, [Decay(:cavity, 0.3; nth=0.5), Gain(:cavity, 0.3; nth=0.5)])
        @test all(dense(a).data ≈ dense(b).data for (a, b) in zip(J1, J2))

        # nth=0 still returns a two-element list (Gain's op is just zero),
        # so callers don't need to special-case the zero-temperature limit.
        @test length(thermal_bath(:cavity, 0.3, 0.0)) == 2
    end

    @testset "evolve dispatches to the matching QuantumOptics.jl solver" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs = CompositeSystem(q, c)
        g = 0.1
        H = jaynes_cummings(cs, :qubit, :cavity, g)
        psi0 = spinup(SpinBasis(1//2)) ⊗ fockstate(FockBasis(4), 0)
        tspan = 0:0.1:2.0

        # `evolve` overrides QuantumOptics.jl's own solver defaults (see "Solver
        # wrappers" in design.md), so the reference calls have to match for an
        # exact comparison — that forwarding is itself part of what's tested.
        solver_opts = (; alg=QuantumDynamics.DEFAULT_ALG,
                         reltol=QuantumDynamics.DEFAULT_RELTOL,
                         abstol=QuantumDynamics.DEFAULT_ABSTOL)

        # closed, time-independent -> schroedinger
        _, states = evolve(tspan, psi0, H)
        _, states_ref = timeevolution.schroedinger(tspan, psi0, H; solver_opts...)
        @test all(states[i].data ≈ states_ref[i].data for i in eachindex(tspan))

        # closed, time-dependent -> schroedinger_dynamic
        a, ad = op(cs, :cavity, :a), op(cs, :cavity, :ad)
        Htd = TimeDependentSum(1.0 => H, (t -> 0.05 * cos(t)) => (a + ad))
        _, states_td = evolve(tspan, psi0, Htd)
        _, states_td_ref = timeevolution.schroedinger_dynamic(tspan, psi0, Htd; solver_opts...)
        @test all(states_td[i].data ≈ states_td_ref[i].data for i in eachindex(tspan))

        # open, time-independent -> master
        J = jump_operators(cs, [Decay(:cavity, 0.2)])
        _, rhos = evolve(tspan, psi0, H, J)
        _, rhos_ref = timeevolution.master(tspan, psi0, H, J; solver_opts...)
        @test all(dense(rhos[i]).data ≈ dense(rhos_ref[i]).data for i in eachindex(tspan))

        # open, time-dependent -> master_dynamic
        _, rhos_td = evolve(tspan, psi0, Htd, J)
        _, rhos_td_ref = timeevolution.master_dynamic(tspan, psi0, Htd, J; solver_opts...)
        @test all(dense(rhos_td[i]).data ≈ dense(rhos_td_ref[i]).data for i in eachindex(tspan))
    end

    @testset "evolve: solver defaults and normalization guard" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs = CompositeSystem(q, c)
        H = jaynes_cummings(cs, :qubit, :cavity, 0.1)
        psi0 = spinup(SpinBasis(1//2)) ⊗ fockstate(FockBasis(4), 0)
        tspan = 0:0.1:2.0

        # evolve overrides QuantumOptics.jl's DP5 default with a higher-order method
        @test nameof(typeof(QuantumDynamics.DEFAULT_ALG)) == :Vern9

        # a well-resolved run stays normalized and emits no warning
        result = @test_logs min_level = Logging.Warn evolve(tspan, psi0, H)
        _, states = result
        @test abs(norm(states[end])^2 - 1) < QuantumDynamics.NORMALIZATION_ATOL
        @test_logs min_level = Logging.Warn evolve(tspan, psi0, H; check_normalization=false)

        # the guard passes the trajectory through untouched...
        @test QuantumDynamics._check_normalization(result) === result
        # ...but warns when norm² (Ket) or trace (density operator) has drifted
        @test_logs (:warn, r"norm² deviates from 1") QuantumDynamics._check_normalization((tspan, [1.5 * states[end]]))
        @test_logs (:warn, r"trace deviates from 1") QuantumDynamics._check_normalization((tspan, [1.5 * dm(states[end])]))

        # a custom fout whose output isn't a state is not checkable -> silent passthrough
        expectation_out = (tspan, [0.1, 0.2, 0.3])
        val = @test_logs min_level = Logging.Warn QuantumDynamics._check_normalization(expectation_out)
        @test val === expectation_out
    end

    @testset "evolve: closed/open consistency (no dissipation)" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=4)
        cs = CompositeSystem(q, c)
        g = 0.1
        H = jaynes_cummings(cs, :qubit, :cavity, g)
        psi0 = spinup(SpinBasis(1//2)) ⊗ fockstate(FockBasis(4), 0)
        tspan = 0:0.1:2.0

        _, states = evolve(tspan, psi0, H)
        _, rhos = evolve(tspan, psi0, H, Operator[])
        @test isapprox(dense(rhos[end]).data, dense(dm(states[end])).data; atol=1e-4)
    end

    @testset "evolve: master-equation dissipation matches known solutions" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=6)
        cs = CompositeSystem(q, c)
        Hbare = bare_hamiltonian(cs)  # no coupling: qubit and cavity decay independently
        tspan = 0:0.05:4.0

        # Cavity photon loss: ⟨n⟩(t) = n0 * exp(-κt)
        κ = 0.5
        J = jump_operators(cs, [Decay(:cavity, κ)])
        psi0 = spindown(SpinBasis(1//2)) ⊗ fockstate(FockBasis(6), 2)
        _, rhos = evolve(tspan, psi0, Hbare, J)
        n_c = op(cs, :cavity, :n)
        pops = [real(expect(n_c, r)) for r in rhos]
        @test maximum(abs.(pops .- 2 .* exp.(-κ .* tspan))) < 1e-3

        # Qubit T1 relaxation: ⟨n⟩(t) = exp(-γ1*t)
        γ1 = 0.4
        J_q = jump_operators(cs, [Decay(:qubit, γ1)])
        psi0_q = spinup(SpinBasis(1//2)) ⊗ fockstate(FockBasis(6), 0)
        _, rhos_q = evolve(tspan, psi0_q, Hbare, J_q)
        n_q = op(cs, :qubit, :n)
        pops_q = [real(expect(n_q, r)) for r in rhos_q]
        @test maximum(abs.(pops_q .- exp.(-γ1 .* tspan))) < 1e-3

        # Pure dephasing: jump operator sqrt(γφ/2)*σz leaves populations
        # untouched and decays coherence (off-diagonal, tracked via ⟨σm⟩) at
        # rate γφ directly (see Dephasing docstring for the /2 convention).
        γφ = 0.3
        J_deph = jump_operators(cs, [Dephasing(:qubit, γφ)])
        sb = SpinBasis(1//2)
        psi0_coh = normalize(spinup(sb) + spindown(sb)) ⊗ fockstate(FockBasis(6), 0)
        _, rhos_deph = evolve(tspan, psi0_coh, Hbare, J_deph)
        σm = op(cs, :qubit, :σm)
        coh = abs.([expect(σm, r) for r in rhos_deph])
        @test maximum(abs.(coh .- coh[1] .* exp.(-γφ .* tspan))) < 1e-3
        @test maximum(abs.([real(expect(n_q, r)) for r in rhos_deph] .- 0.5)) < 1e-3
    end

    @testset "evolve: thermal bath (Decay+Gain) matches known solution" begin
        q = Qubit(:qubit, 5.0)
        # nmax=15 keeps Fock-space truncation error well under the test
        # tolerance for nth=1.5 (nmax=10 leaves ~2.5% truncation bias here).
        c = HarmonicOscillator(:cavity, 6.0; nmax=15)
        cs = CompositeSystem(q, c)
        Hbare = bare_hamiltonian(cs)
        tspan = 0:0.05:6.0

        # A thermal bath (Decay+Gain sharing the same rate/nth) relaxes
        # ⟨n⟩(t) toward nth: ⟨n⟩(t) = nth + (n0-nth)*exp(-rate*t).
        γ = 0.5
        nth = 1.5
        J = jump_operators(cs, [Decay(:cavity, γ; nth=nth), Gain(:cavity, γ; nth=nth)])
        psi0 = spindown(SpinBasis(1//2)) ⊗ fockstate(FockBasis(15), 0)
        _, rhos = evolve(tspan, psi0, Hbare, J)
        n_c = op(cs, :cavity, :n)
        pops = [real(expect(n_c, r)) for r in rhos]
        expected = nth .+ (0.0 - nth) .* exp.(-γ .* tspan)
        @test maximum(abs.(pops .- expected)) < 5e-3
    end

    @testset "SimulationResult: save_result/load_result round trip" begin
        q = Qubit(:qubit, 5.0)
        c = HarmonicOscillator(:cavity, 6.0; nmax=6)
        cs = CompositeSystem(q, c)
        H = jaynes_cummings(cs, :qubit, :cavity, 0.3)
        psi0 = spinup(SpinBasis(1//2)) ⊗ fockstate(FockBasis(6), 0)
        tspan = 0:0.1:1.0

        @testset "closed system (Kets, no dissipators)" begin
            tout, states = evolve(tspan, psi0, H)
            r = SimulationResult(cs, tout, states; H=H)
            path = tempname() * ".jld2"
            save_result(path, r)
            r2 = load_result(path)
            rm(path)

            @test r2.times == tout
            @test eltype(r2.states) <: Ket
            @test all(r2.states[i].data ≈ states[i].data for i in eachindex(states))
            @test dense(r2.H).data ≈ dense(H).data
            @test isempty(r2.dissipators)
            @test r2.description == ""
            @test isempty(r2.params)

            # subsystems (not the CompositeSystem itself) are what's saved —
            # CompositeSystem(r2) rebuilds cs from them.
            cs2 = CompositeSystem(r2)
            @test getsubsystem(cs2, :qubit).ω == 5.0
            @test getsubsystem(cs2, :cavity).ω == 6.0
            @test dense(op(cs2, :qubit, :σz)).data ≈ dense(op(cs, :qubit, :σz)).data
        end

        @testset "open system (density operators, dissipators, description/params)" begin
            dissipators = [Decay(:cavity, 0.1)]
            J = jump_operators(cs, dissipators)
            tout, rhos = evolve(tspan, psi0, H, J)
            r = SimulationResult(cs, tout, rhos; dissipators=dissipators, H=H,
                description="JC + cavity decay", params=Dict(:g => 0.3, :κ => 0.1))
            path = tempname() * ".jld2"
            save_result(path, r)
            r2 = load_result(path)
            rm(path)

            @test all(dense(r2.states[i]).data ≈ dense(rhos[i]).data for i in eachindex(rhos))
            @test r2.description == "JC + cavity decay"
            @test r2.params == Dict(:g => 0.3, :κ => 0.1)

            # dissipators (not J itself) are what's saved — jump_operators
            # rebuilds J after reconstructing cs.
            cs2 = CompositeSystem(r2)
            J2 = jump_operators(cs2, r2.dissipators)
            @test length(J2) == 1
            @test dense(J2[1]).data ≈ dense(J[1]).data
        end

        @testset "format_version key is written" begin
            tout, states = evolve(tspan, psi0, H)
            r = SimulationResult(cs, tout, states)
            path = tempname() * ".jld2"
            save_result(path, r)
            fv = jldopen(f -> f["format_version"], path, "r")
            rm(path)
            @test fv == QuantumDynamics.RESULT_FORMAT_VERSION
        end

        @testset "time-dependent H (named coefficient function) round trip" begin
            # Regression test for the one TimeDependentSum round-trip path
            # design.md documents as actually working: a *named*, top-level
            # coefficient function (unlike an anonymous closure, which JLD2
            # cannot reload as callable — see "Result persistence" in
            # design.md). Uses `_sr_test_drive_coeff`, defined at the top of
            # this file for that reason.
            Ht = add_time_dependence(H, _sr_test_drive_coeff => op(cs, :qubit, :σx))
            tout, states = evolve(tspan, psi0, Ht)
            r = SimulationResult(cs, tout, states; H=Ht)
            path = tempname() * ".jld2"
            save_result(path, r)
            r2 = load_result(path)
            rm(path)

            @test r2.H isa TimeDependentSum
            tout2, states2 = evolve(tspan, psi0, r2.H)
            @test all(states2[i].data ≈ states[i].data for i in eachindex(states))
        end
    end

end
