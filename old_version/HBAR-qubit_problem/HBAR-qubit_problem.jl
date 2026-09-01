#####################################################################################
#= Physical parameters and variables of a HBAR coupled to a qubit (from Chu et al.)=#
#####################################################################################
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
n = 0; #average number of phonons at first step
Δ0 = ωq - ωm; #System detuning
#g = 258 ; #JC coupling rate


#Coupled System -> operators and basis definitions using QuantumOptics.jl
basis = tensor(SpinBasis(1//2), FockBasis(N_mech))
qub, mech_res, qubit_mech = Qubit_HO(N_mech, :FockBasis, 1//2)



########################################################################################################################################
#################################################################################
#= FUNCTIONS TO GENERATE THE DATASET FOR THE FL_1step PROBLEM (Chu's protocol) =#
#################################################################################



######################### inputs outputs functions ##################################
function FL_1step_NN_outputs(p, parameters_range, dim_parameters_space, n_samples)  #ok
    parameters_space, prob = twoD_parameter_space(p, parameters_range, dim_parameters_space)

    return sample(parameters_space, Weights(prob), n_samples)
end

function FL_1step_NN_inputs(t0, initial_state, dataset_features, pulse_parameters, n_samples, typeofdynamics) #Typeofdynamics contains [:corrected]
    inputs = Vector{Vector{Float64}}(undef, n_samples) #exit vector

    final_states = [] #vector with target states
    
    infidelity_spin_flip = Vector{Float64}(undef, n_samples)

    #G = gellmann_operators(dimension) ./ 2 #Gell-mann functions

    #=The dynamical parameters are [τ_exc, Ω_R = π / τ_exc, τ_SWAP] =#
    pulse_parameters = [[pulse_parameters[i][1], π / pulse_parameters[i][1], pulse_parameters[i][2]] for i in 1:length(pulse_parameters)]

    #Generation of n_samples number of quantum trajectories
    @showprogress for i in 1:n_samples
        ρ_at_flip,  ρ_at_t_end = FLstep_dynamics(t0, initial_state, pulse_parameters[i], typeofdynamics, dataset_features, :dynamics) #output -> ψ_at_flip,  ψ_at_t_end (kets or density matrices) 

        push!(final_states,  ρ_at_t_end)  #⊗ dagger(ρ_at_t_end)
        infidelity_spin_flip[i] = qo_infidelity(ρ_at_flip, dataset_features.state_target_spinflip) #need to be fixed, ⊗ dagger(ρ_at_flip)
        inputs[i] = [expectation_value(matrix, final_states[end]) for matrix in dataset_features.decom_basis] #Expectation values of the states on the Gell-Mann matrices

    end

    #The following computes the ms infidelity of the final state on the target
    swap_infidelity = in_qo_infidelity(final_states, dataset_features.state_target_1step) #need to be fixed
    input_infidelity = [[infidelity_spin_flip[i], swap_infidelity[i]] for i in 1:length(swap_infidelity)]

    inputs = [vcat(inputs[t], input_infidelity[t]) for t in 1:n_samples]

    return inputs#, final_states

end


function FLstep_dynamics(t0, initial_state, pulse_parameters, typeofdynamics, problem_features, modeofdynamics) #mode_vector -> [typeofcorrection, n_phonon, :dynamics]
    τ_exc, Ω_R, τ_SWAP = pulse_parameters

    dt = 1e-6; #dt of time integration

    typeofcorrection, n_phonon = problem_features.correction, problem_features.phonon_n

    # Pre-define the function, avoid scoping issues
    Hamiltonian, dissipation, dissipation_d = create_FLstep_dynamics(t0, pulse_parameters, typeofcorrection, n_phonon)
    dynamics_input = nothing

    #Here, we discriminate between dissipative and unitary dynamics
    if typeofdynamics == :master || typeofdynamics == :master_dynamic #to be fixed
        dynamics_input = (t, ψ) -> (Hamiltonian(t, ψ), dissipation, dissipation_d)

        #evolution from t0 to τ_exc
        tspan1 = [t0, dt, t0 + τ_exc]
        tspan_1, ρ_out1 = dynamic_evolution(tspan1, initial_state, dynamics_input,  typeofdynamics)
        ρ_at_τ_exc = ρ_out1[end] 

        #evolution from τ_exc to t_end = τ_exc + τ_SWAP
        tspan2 = [t0 + τ_exc, dt, t0 + τ_exc + τ_SWAP]
        tspan_2, ρ_out2 = dynamic_evolution(tspan2, ρ_at_τ_exc, dynamics_input,  typeofdynamics)
        ρ_at_t_end = ρ_out2[end]

        if modeofdynamics == :dynamics
            return ρ_at_τ_exc,  ρ_at_t_end
        elseif modeofdynamics == :final_state
            return tspan_2[end], ρ_at_t_end
        elseif modeofdynamics == :all_dynamics
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end])
        else
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end]), τ_exc, τ_SWAP
        end

    else #outputs are kets
        dynamics_input = (t, ψ) -> Hamiltonian(t, ψ)

        #evolution from t0 to τ_exc
        tspan1 = [t0, dt, t0 + τ_exc]
        tspan_1, ρ_out1 = dynamic_evolution(tspan1, initial_state, dynamics_input,  typeofdynamics)
        ρ_at_τ_exc = ρ_out1[end] 

        #evolution from τ_exc to t_end = τ_exc + τ_SWAP
        tspan2 = [t0 + τ_exc, dt, t0 + τ_exc + τ_SWAP]
        tspan_2, ρ_out2 = dynamic_evolution(tspan2, ρ_at_τ_exc, dynamics_input,  typeofdynamics)
        ρ_at_t_end = ρ_out2[end]
        
        if modeofdynamics == :dynamics
            return ρ_at_τ_exc, ρ_at_t_end
        elseif modeofdynamics == :final_state
            return tspan_2[end], ρ_at_t_end
        elseif modeofdynamics == :all_dynamics
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end])
        else
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end]), τ_exc, τ_SWAP
        end

    end
    
end

#= [Hamiltonian, Lindblad operator and its hermitian conjugate]  =#
function create_FLstep_dynamics(t0, pulse_parameters, typeofcorrection, n_phonon)
    τ_exc, Ω_R, τ_SWAP = pulse_parameters

    if typeofcorrection == :correction_on
        χ = g^2 / Δ0

        Δ0_tilde = Δ0 + (2*χ*(n_phonon + 0.5))
    else
        Δ0_tilde = Δ0
    end

    #Time indipendent Hamiltonian
    H_JC = g * (qubit_mech.pI*qubit_mech.Ia + qubit_mech.mI*qubit_mech.Iad)
    H0 = 0.5 * Δ0_tilde * qubit_mech.zI + H_JC

    #Function defining the Rabi oscillation and detuning
    Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(Δ0_tilde * t)
    Δ(t) = - Δ0_tilde * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    Ht = LazySum([Ω(t0), Δ(t0)/2], [qubit_mech.xI, qubit_mech.zI])
    function Hamiltonian(t, ψ)
        Ht.factors[1] = Ω(t)
        Ht.factors[2] = Δ(t)/2
        return H0 + Ht
    end
    
    return [
    Hamiltonian, 
    [sqrt(γm * (nthm+1))*qubit_mech.Ia, sqrt(γm*(nthm))*qubit_mech.Iad, sqrt(κϕ/2)  * qubit_mech.zI, sqrt(κ) * qubit_mech.mI],
    dagger.([sqrt(γm * (nthm+1))*qubit_mech.Ia, sqrt(γm*(nthm))*qubit_mech.Iad, sqrt(κϕ/2)  * qubit_mech.zI, sqrt(κ) * qubit_mech.mI])
    ]
end


# fix it 
function dynamics_n_steps_FL(N_steps, initial_state, pulse_parameters, typeofdynamics, problem_features)
    target_states = problem_features.state_target_1step
    type_of_correction = problem_features.correction

    time_final = Float64[]
    solution_final = []
    infidelities = zeros(Float64, N_steps)

    t0 = 0.0

    @showprogress for step in 1:N_steps

        params = pulse_parameters[step]
        params = [params[1], π / params[1], params[2]]

        tspan, solution = FLstep_dynamics(t0, initial_state, params, typeofdynamics, problem_features, :all_dynamics) 

        time_final = vcat(time_final, tspan)
        solution_final = vcat(solution_final, solution)
        infidelities[step] = qo_infidelity(solution[end], problem_features.state_target_1step[step])

        #updating
        t0 = time_final[end]
        initial_state = solution_final[end]
        problem_features =  fl_1step_features(
                            [0.0],
                            target_states, 
                            [0], 
                            step, 
                            type_of_correction
        )

    end

    return time_final, solution_final, infidelities

end

function n_steps_FL_plot(initial_state, pulse_parameters, typeofdynamics, problem_features)
    N_steps = length(pulse_parameters)

    tspan, trajectory, infidelities = dynamics_n_steps_FL(N_steps, initial_state, pulse_parameters, typeofdynamics, problem_features)
    data1, data2, plt = expectation_and_plot_comparison(tspan, qubit_mech.n_qubit, "<σz>", qubit_mech.n_mech, "<n>", trajectory)

    return data1, data2, plt, infidelities

end


function creation_step_states(ψ0::Ket{B, T}, step_number) where {B,T}
    sn = step_number

    return step_states(
    ψ0,
    ψ0 ⊗ dagger(ψ0),
    tensor(spinup(qub.basis),fockstate(mech_res.basis, sn-1)),
    tensor(spindown(qub.basis),fockstate(mech_res.basis, sn)),
    tensor(spinup(qub.basis),fockstate(mech_res.basis, sn-1)) ⊗ dagger(tensor(spinup(qub.basis),fockstate(mech_res.basis, sn-1))),
    tensor(spindown(qub.basis),fockstate(mech_res.basis, sn)) ⊗ dagger(tensor(spindown(qub.basis),fockstate(mech_res.basis, sn)))
    )


end


function creation_step_states(ρ0::Operator{B1, B2, T}, step_number) where {B1, B2, T}
    sn = step_number
    ψ0 = tensor(spindown(qub.basis),fockstate(mech_res.basis, 0))

    return step_states(
    ψ0,
    ρ0,
    tensor(spinup(qub.basis),fockstate(mech_res.basis, sn-1)),
    tensor(spindown(qub.basis),fockstate(mech_res.basis, sn)),
    tensor(spinup(qub.basis),fockstate(mech_res.basis, sn-1)) ⊗ dagger(tensor(spinup(qub.basis),fockstate(mech_res.basis, sn-1))),
    tensor(spindown(qub.basis),fockstate(mech_res.basis, sn)) ⊗ dagger(tensor(spindown(qub.basis),fockstate(mech_res.basis, sn)))
    )


end




##################################################################################################################################################

#=
####################
#= generic drives =#
####################
function FT_FL_NN_outputs(p, parameters_range, dim_parameters_space, n_samples)  #ok
    # Generate the space for the problem

    return sample(parameters_space, Weights(prob), n_samples)
end

function FL_NN_inputs(t0, initial_state, dataset_features, pulse_parameters, n_samples, typeofdynamics) #ok
    ψ_target_1step, ψ_target_spinflip, basis_decomposition = dataset_features

    inputs = Vector{Vector{Float64}}(undef, n_samples) #exit vector

    final_states = Vector{Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, FockBasis{Int64}}},
                          CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, FockBasis{Int64}}}, 
                          Matrix{ComplexF64}}}(undef, n_samples) #vector with target states
    
    infidelity_spin_flip = Vector{Float64}(undef, n_samples)

    #G = gellmann_operators(dimension) ./ 2 #Gell-mann functions

    #=The dynamical parameters are [τ_exc, Ω_R = π / τ_exc, τ_SWAP] =#
    pulse_parameters = [[pulse_parameters[i][1], π / pulse_parameters[i][1], pulse_parameters[i][2]] for i in 1:length(pulse_parameters)]

    #Generation of n_samples number of quantum trajectories
    @showprogress for i in 1:n_samples
        ρ_at_flip,  ρ_at_t_end = FLstep_dynamics(t0, initial_state, pulse_parameters[i], typeofdynamics, :final_state) #output -> ρ_at_flip,  ρ_at_t_end 
        
        final_states[i] = ρ_at_t_end
        infidelity_spin_flip[i] = qo_infidelity(ρ_at_flip, ψ_target_spinflip)
        inputs[i] = [expectation_value(matrix, final_states[i]) for matrix in basis_decomposition] #Expectation values of the states on the Gell-Mann matrices

    end
    #The following computes the ms infidelity of the final state on the target
    swap_infidelity = in_qo_infidelity(final_states, ψ_target_1step)
    input_infidelity = [[infidelity_spin_flip[i], swap_infidelity[i]] for i in 1:length(swap_infidelity)]

    inputs = [vcat(inputs[t], input_infidelity[t]) for t in 1:n_samples]

    return inputs#, final_states

end



function Fourier_basis(N, T)
    ω0 = 2π / T
    ωs = [ω0 * n for n in 1:N]

    return ωs, [t -> cos(ω * t) for ω in ωs], [t -> sin(ω * t) for ω in ωs]
end

# drive = Fourier_composition()
function Fourier_composition(coeffs, Fbasis)
    return x -> sum(coeffs[i] * Fbasis[i](t) for i in eachindex(c))
end
=#

##########################################################################################
#= FUNCTIONS TO GENERATE THE DATASET FOR THE FL_1step PROBLEM WITH 3 DEGREES OF FREEDOM=#
##########################################################################################
function FL_1step_3p_NN_outputs(p, parameters_range, dim_parameters_space, n_samples)  #Ok
    parameters_space, prob = threeD_parameter_space(p, parameters_range, dim_parameters_space)

    return sample(parameters_space, Weights(prob), n_samples)
end

function FL_1step_3p_NN_inputs(t0, initial_state, dataset_features, pulse_parameters, n_samples, typeofdynamics) #Typeofdynamics contains [:corrected]
    inputs = Vector{Vector{Float64}}(undef, n_samples) #exit vector

    final_states = [] #vector with target states
    
    infidelity_spin_flip = Vector{Float64}(undef, n_samples)

    #=The dynamical parameters are [τ_exc, Ω_R = π / τ_exc, τ_SWAP] =#
    pulse_parameters = [[pulse_parameters[i][1], π /pulse_parameters[i][1] , pulse_parameters[i][2], pulse_parameters[i][3]] for i in 1:length(pulse_parameters)]

    #Generation of n_samples number of quantum trajectories
    @showprogress for i in 1:n_samples
        ρ_at_flip,  ρ_at_t_end = FLstep_dynamics_3p(t0, initial_state, pulse_parameters[i], typeofdynamics, dataset_features, :dynamics) #output -> ψ_at_flip,  ψ_at_t_end (kets or density matrices) 

        push!(final_states,  ρ_at_t_end)  #⊗ dagger(ρ_at_t_end)
        infidelity_spin_flip[i] = qo_infidelity(ρ_at_flip, dataset_features.state_target_spinflip) #need to be fixed, ⊗ dagger(ρ_at_flip)
        inputs[i] = [expectation_value(matrix, final_states[end]) for matrix in dataset_features.decom_basis] #Expectation values of the states on the Gell-Mann matrices

    end

    #The following computes the ms infidelity of the final state on the target
    swap_infidelity = in_qo_infidelity(final_states, dataset_features.state_target_1step) #need to be fixed
    input_infidelity = [[infidelity_spin_flip[i], swap_infidelity[i]] for i in 1:length(swap_infidelity)]

    inputs = [vcat(inputs[t], input_infidelity[t]) for t in 1:n_samples]

    return inputs#, final_states
end


function FLstep_dynamics_3p(t0, initial_state, pulse_parameters, typeofdynamics, problem_features, modeofdynamics) #mode_vector -> [typeofcorrection, n_phonon, :dynamics]
    τ_exc, Ω_R, ωd, τ_SWAP = pulse_parameters

    dt = 1e-6; #dt of time integration

    typeofcorrection, n_phonon = problem_features.correction, problem_features.phonon_n

    # Pre-define the function, avoid scoping issues
    Hamiltonian, dissipation, dissipation_d = create_FLstep_dynamics_3p(t0, pulse_parameters, typeofcorrection, n_phonon)
    dynamics_input = nothing

    #Here, we discriminate between dissipative and unitary dynamics
    if typeofdynamics == :master || typeofdynamics == :master_dynamic #to be fixed
        dynamics_input = (t, ψ) -> (Hamiltonian(t, ψ), dissipation, dissipation_d)

        #evolution from t0 to τ_exc
        tspan1 = [t0, dt, t0 + τ_exc]
        tspan_1, ρ_out1 = dynamic_evolution(tspan1, initial_state, dynamics_input,  typeofdynamics)
        ρ_at_τ_exc = ρ_out1[end] 

        #evolution from τ_exc to t_end = τ_exc + τ_SWAP
        tspan2 = [t0 + τ_exc, dt, t0 + τ_exc + τ_SWAP]
        tspan_2, ρ_out2 = dynamic_evolution(tspan2, ρ_at_τ_exc, dynamics_input,  typeofdynamics)
        ρ_at_t_end = ρ_out2[end]

        if modeofdynamics == :dynamics
            return ρ_at_τ_exc,  ρ_at_t_end
        elseif modeofdynamics == :final_state
            return tspan_2[end], ρ_at_t_end
        elseif modeofdynamics == :all_dynamics
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end])
        else
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end]), τ_exc, τ_SWAP
        end

    else #outputs are kets
        dynamics_input = (t, ψ) -> Hamiltonian(t, ψ)

        #evolution from t0 to τ_exc
        tspan1 = [t0, dt, t0 + τ_exc]
        tspan_1, ρ_out1 = dynamic_evolution(tspan1, initial_state, dynamics_input,  typeofdynamics)
        ρ_at_τ_exc = ρ_out1[end] 

        #evolution from τ_exc to t_end = τ_exc + τ_SWAP
        tspan2 = [t0 + τ_exc, dt, t0 + τ_exc + τ_SWAP]
        tspan_2, ρ_out2 = dynamic_evolution(tspan2, ρ_at_τ_exc, dynamics_input,  typeofdynamics)
        ρ_at_t_end = ρ_out2[end]
        
        if modeofdynamics == :dynamics
            return ρ_at_τ_exc, ρ_at_t_end
        elseif modeofdynamics == :final_state
            return tspan_2[end], ρ_at_t_end
        elseif modeofdynamics == :all_dynamics
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end])
        else
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end]), τ_exc, τ_SWAP
        end

    end
    
end

#= [Hamiltonian, Lindblad operator and its hermitian conjugate]  =#
function create_FLstep_dynamics_3p(t0, pulse_parameters, typeofcorrection, n_phonon)
    τ_exc, Ω_R, ωd, τ_SWAP = pulse_parameters

    #=
    if typeofcorrection == :correction_on
        χ = g^2 / Δ0

        Δ0_tilde = Δ0 + (2*χ*(n_phonon + 0.5))
    else
        Δ0_tilde = Δ0
    end
    =#

    #Time indipendent Hamiltonian
    H_JC = g * (qubit_mech.pI*qubit_mech.Ia + qubit_mech.mI*qubit_mech.Iad)
    H0 = 0.5 * Δ0 * qubit_mech.zI + H_JC

    #Function defining the Rabi oscillation and detuning
    Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(ωd * t)
    Δ(t) = - Δ0 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    Ht = LazySum([Ω(t0), Δ(t0)/2], [qubit_mech.xI, qubit_mech.zI])
    function Hamiltonian(t, ψ)
        Ht.factors[1] = Ω(t)
        Ht.factors[2] = Δ(t)/2
        return H0 + Ht
    end
    
    return [
    Hamiltonian, 
    [sqrt(γm * (nthm+1))*qubit_mech.Ia, sqrt(γm*(nthm))*qubit_mech.Iad, sqrt(κϕ/2)  * qubit_mech.zI, sqrt(κ) * qubit_mech.mI],
    dagger.([sqrt(γm * (nthm+1))*qubit_mech.Ia, sqrt(γm*(nthm))*qubit_mech.Iad, sqrt(κϕ/2)  * qubit_mech.zI, sqrt(κ) * qubit_mech.mI])
    ]
end






#########################################################################################
#= FUNCTIONS TO GENERATE THE DATASET FOR THE 2-drives FL=#
#########################################################################################
# The purpose is to generate the FL with two drives: '
#    1. spin flip of the qubit with drive [Ω_R1(t), τ_flip];
#    2. SWAP operation with qubit drive [Ω_R2(t), τ_swap].
# The ouput are decomposed on BSpline
function FL_1step_2drives_NN_outputs(prs, parameters_range, dim_parameters_space, n_samples; log_sampling=true)  #ok
    dim_dataset = length(parameters_range)
    samples = []

    #=
    #BSpline definition
    order_spline = 4            # spline order (cubic spline)
    n_basis_spline = 10         # number of basis functions
    domain_spline = (0.0, 1.0)  # interval for the spline

    BS_basis = generate_Bspline_basis(order, n_basis, domain)
    =#

    index = 1
    while length(samples) < n_samples
        single_sample = Vector{Float64}(undef, dim_dataset)
        probability = 1.0

        for k in 1:dim_dataset
            a, b = parameters_range[k] 
            if a > 0 && log_sampling
                extracted = exp(rand(Uniform(log(a), log(b))))
            else
                extracted = rand(Uniform(a, b))
            end

            single_sample[k] =  extracted
            probability *= prs[k](extracted)
        end

        u = rand()
        if u < probability
            push!(samples, single_sample)
            index += 1
        end

    end
    return samples
end




function FL_1step_2drives_NN_inputs(t0, initial_state, dataset_features, pulse_parameters, n_samples, typeofdynamics) #Typeofdynamics contains [:corrected]
    inputs = Vector{Vector{Float64}}(undef, n_samples) #exit vector

    final_states = [] #vector with target states
    
    infidelity_spin_flip = Vector{Float64}(undef, n_samples)

    #Generation of n_samples number of quantum trajectories
    @showprogress for i in 1:n_samples
        ρ_at_flip,  ρ_at_t_end = FLstep_2drives_dynamics(t0, initial_state, pulse_parameters[i], typeofdynamics, dataset_features, :dynamics) #output -> ψ_at_flip,  ψ_at_t_end (kets or density matrices) 

        push!(final_states,  ρ_at_t_end)  #⊗ dagger(ρ_at_t_end)
        infidelity_spin_flip[i] = qo_infidelity(ρ_at_flip, dataset_features.state_target_spinflip) #need to be fixed, ⊗ dagger(ρ_at_flip)
        inputs[i] = [expectation_value(matrix, final_states[end]) for matrix in dataset_features.decom_basis] #Expectation values of the states on the Gell-Mann matrices

    end

    #The following computes the ms infidelity of the final state on the target
    swap_infidelity = in_qo_infidelity(final_states, dataset_features.state_target_1step) #need to be fixed
    input_infidelity = [[infidelity_spin_flip[i], swap_infidelity[i]] for i in 1:length(swap_infidelity)]

    inputs = [vcat(inputs[t], input_infidelity[t]) for t in 1:n_samples]

    return inputs#, final_states
end


function FLstep_2drives_dynamics(t0, initial_state, pulse_parameters, typeofdynamics, problem_features, modeofdynamics) #mode_vector -> [typeofcorrection, n_phonon, :dynamics]
    τ_exc, τ_SWAP = pulse_parameters[end-1], pulse_parameters[end]
    dt = 1e-6; #dt of time integration

    typeofcorrection, n_phonon = problem_features.correction, problem_features.phonon_n

    # Pre-define the function, avoid scoping issues
    Hamiltonian, dissipation, dissipation_d = create_FLstep_2drives_dynamics(t0, pulse_parameters, typeofcorrection, n_phonon)
    dynamics_input = nothing

    #Here, we discriminate between dissipative and unitary dynamics
    if typeofdynamics == :master || typeofdynamics == :master_dynamic #to be fixed
        dynamics_input = (t, ψ) -> (Hamiltonian(t, ψ), dissipation, dissipation_d)

        #evolution from t0 to τ_exc
        tspan1 = [t0, dt, t0 + τ_exc]
        tspan_1, ρ_out1 = dynamic_evolution(tspan1, initial_state, dynamics_input,  typeofdynamics)
        ρ_at_τ_exc = ρ_out1[end] 

        #evolution from τ_exc to t_end = τ_exc + τ_SWAP
        tspan2 = [t0 + τ_exc, dt, t0 + τ_exc + τ_SWAP]
        tspan_2, ρ_out2 = dynamic_evolution(tspan2, ρ_at_τ_exc, dynamics_input,  typeofdynamics)
        ρ_at_t_end = ρ_out2[end]

        if modeofdynamics == :dynamics
            return ρ_at_τ_exc,  ρ_at_t_end
        elseif modeofdynamics == :final_state
            return tspan_2[end], ρ_at_t_end
        elseif modeofdynamics == :all_dynamics
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end])
        else
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end]), τ_exc, τ_SWAP
        end

    else #outputs are kets
        dynamics_input = (t, ψ) -> Hamiltonian(t, ψ)

        #evolution from t0 to τ_exc
        tspan1 = [t0, dt, t0 + τ_exc]
        tspan_1, ρ_out1 = dynamic_evolution(tspan1, initial_state, dynamics_input,  typeofdynamics)
        ρ_at_τ_exc = ρ_out1[end] 

        #evolution from τ_exc to t_end = τ_exc + τ_SWAP
        tspan2 = [t0 + τ_exc, dt, t0 + τ_exc + τ_SWAP]
        tspan_2, ρ_out2 = dynamic_evolution(tspan2, ρ_at_τ_exc, dynamics_input,  typeofdynamics)
        ρ_at_t_end = ρ_out2[end]
        
        if modeofdynamics == :dynamics
            return ρ_at_τ_exc, ρ_at_t_end
        elseif modeofdynamics == :final_state
            return tspan_2[end], ρ_at_t_end
        elseif modeofdynamics == :all_dynamics
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end])
        else
            return vcat(tspan_1, tspan_2[2:end]), vcat(ρ_out1, ρ_out2[2:end]), τ_exc, τ_SWAP
        end

    end
    
end


function create_FLstep_2drives_dynamics(t0, pulse_parameters, typeofcorrection, n_phonon)
    τ_exc, τ_SWAP = pulse_parameters[end-1], pulse_parameters[end]
    Ω_R1, Ω_R2 = pulse_parameters[end-3], pulse_parameters[end-2]
    T = τ_exc + τ_SWAP

    if typeofcorrection == :correction_on
        χ = g^2 / Δ0

        Δ0_tilde = Δ0 + (2*χ*(n_phonon + 0.5))
    else
        Δ0_tilde = Δ0
    end

    #Time indipendent Hamiltonian
    H_JC = +g * (qubit_mech.pI*qubit_mech.Ia + qubit_mech.mI*qubit_mech.Iad)
    H0 = 0.5 * Δ0_tilde * qubit_mech.zI + H_JC

    #Reassembling the Splines (physical splines)
    func1 = drive_from_normalized_spline(
        Bspline_composition(pulse_parameters[1:n_basis_spline], basis_spline),
        T
    )
    f1 = t -> func1(t)

    func2 = drive_from_normalized_spline(
        Bspline_composition(pulse_parameters[n_basis_spline+1:(2*n_basis_spline)], basis_spline),
        T
    )
    f2 = t -> func2(t)


    Ω1(t) = Ω_R1 * π_pulse_shape(t, t0, τ_exc) * f1(t)
    Ω2(t) = Ω_R2 * π_pulse_shape(t, t0 + τ_exc, τ_SWAP) * f2(t)
    Δ(t) = - Δ0_tilde * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    #println(Ω1.(LinRange(0.0, T, 1000)))
    #Function defining the Rabi oscillation and detuning
    #Ω(t) = Ω_R * π_pulse_shape(t, t0, τ_exc) * cos(Δ0_tilde * t)
    #Δ(t) = - Δ0_tilde * π_pulse_shape(t, t0 + τ_exc, τ_SWAP)

    Ht = LazySum([Ω1(t0), Ω2(t0), Δ(t0)/2], [qubit_mech.xI, qubit_mech.xI, qubit_mech.zI])
    function Hamiltonian(t, ψ)
        Ht.factors[1] = Ω1(t)
        Ht.factors[2] = Ω2(t)
        Ht.factors[3] = Δ(t)/2
        return H0 + Ht
    end
    
    return [
    Hamiltonian, 
    [sqrt(γm * (nthm+1))*qubit_mech.Ia, sqrt(γm*(nthm))*qubit_mech.Iad, sqrt(κϕ/2)  * qubit_mech.zI, sqrt(κ) * qubit_mech.mI],
    dagger.([sqrt(γm * (nthm+1))*qubit_mech.Ia, sqrt(γm*(nthm))*qubit_mech.Iad, sqrt(κϕ/2)  * qubit_mech.zI, sqrt(κ) * qubit_mech.mI])
    ]
end







###################################################################################################################################################
#######################
#= GENERIC FUNCTIONS =#
#######################
function filtering_target(target_state)
    result = deepcopy(target_state)

    for i in 1:length(result)
        if result[i] > 1
            result[i] = 1

        elseif result[i] < 0
            result[i] = 0
        end

    end
    return result
end
######################################################################################################################

