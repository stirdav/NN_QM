#= Dictionary which relates the different problems with their inputs/outputs functions for dataset generation =#
dataset_problems_dictionary = Dict(
        :FL_1step => [FL_1step_NN_inputs, FL_1step_NN_outputs],
        :FL_1step_3p => [FL_1step_3p_NN_inputs, FL_1step_3p_NN_outputs],
        :FL_1step_2drives => [FL_1step_2drives_NN_inputs, FL_1step_2drives_NN_outputs]
)

#= Dictionary which relates the different problems with the dynamics function to call for trajectory plotting =#
plot_problem_dictionary = Dict(
        :FL_1step => FLstep_dynamics,
        :FL_1step_3p => FLstep_dynamics,
        :FL_1step_2drives => FLstep_2drives_dynamics
    )

#Ciao
######################################################################################################
######################
#= SPECIAL MATRICES =#
######################

#= Gell-Mann matrices =#
#= Pauli matrices for N=2 =#
function gellmann_operators(N::Int)
    matrices = Operator[]
    
    # Symmetric Matrices
    for j in 1:N
        for k in j+1:N
            M = zeros(ComplexF64, N, N)
            M[j, k] = 1.0
            M[k, j] = 1.0
            push!(matrices, Operator(basis, M))
        end
    end

    # Antisymmetric Matrices
    for j in 1:N
        for k in j+1:N
            M = zeros(ComplexF64, N, N)
            M[j, k] = -im
            M[k, j] = im
            push!(matrices, Operator(basis, M))
        end
    end

    # Diagonal, Traceless Matrices
    for d in 1:(N - 1)
        M = zeros(ComplexF64, N, N)
        coeff = sqrt(2 / (d * (d + 1)))
        for j in 1:d
            M[j, j] = 1.0
        end
        M[d + 1, d + 1] = -d
        M *= coeff
        push!(matrices, Operator(basis, M))
    end

    return matrices
end

function pauli_operators()
    return gellmann_operator(2)
end

function rand_hermitian_orthonormal_basis(d, bases)
    n = d^2  # Dimension of the space of Hermitian matrices
    mats = Matrix{ComplexF64}[]  # Array to hold the random Hermitian matrices

    # Step 1: Generate n random Hermitian matrices
    for i in 1:n
        A = randn(ComplexF64, d, d)           # Random complex matrix
        H = (A + A') / 2                      # Make it Hermitian (A' is conjugate transpose in Julia)
        push!(mats, H)
    end

    # Step 2: Flatten matrices into real vectors by separating real and imaginary parts
    # This converts each Hermitian matrix into a 2*d^2-dimensional real vector
    vecs = zeros(Float64, 2*d*d, n)
    for (i, H) in enumerate(mats)
        vecs[1:d*d, i] = vec(H) |> real      # Real parts flattened
        vecs[d*d+1:end, i] = vec(H) |> imag  # Imaginary parts flattened
    end

    # Step 3: Orthonormalize the vectors using QR decomposition
    Q, R = qr(vecs)

    # Step 4: Map the orthonormal vectors back to Hermitian matrices
    basis = Matrix{ComplexF64}[]
    half = d*d
    for i in 1:n
        real_part = reshape(Q[1:half, i], d, d)
        imag_part = reshape(Q[half+1:end, i], d, d)
        H = real_part + im*imag_part
        # Force Hermiticity to correct numerical errors
        H = (H + H') / 2
        push!(basis, H)
    end

    return [Operator(bases, H) for H in basis]
end

#########################################################################################################


#########################################################################################################
########################################
#= QM PROBLEMS DYNAMICS and EXPECTATION VALUES =#
########################################
#= Expectation value of operators on states =#
function expectation_value(operator, states)
    return real(expect(operator, states))
end


#= Plot the expectation values =#
function plot_expectation(tspan, operator, expectation_data)
    ev_scatter=scatter(x=tspan, y=expectation_data)

    plot([ev_scatter],
    Layout(
        title = "Expectation value of operator "*operator,
        xaxis_title = "t (μs)",
        yaxis_title = "<" * operator * ">"
        )
    )
end

#= Merging expectation and plots =#
function expectation_and_plot(tspan, operator, operator_str, states)
    data = expectation_value(operator, states)
    plot_expectation(tspan, operator_str, data)
end

#= expectation and plot of two operators =#
function expectation_and_plot_comparison(tspan, operator1, operator1_str, operator2, operator2_str, states)
    data1 = expectation_value(operator1, states)
    data2 = expectation_value(operator2, states)

    e1_scatter=scatter(x=tspan, y=data1)
    e2_scatter=scatter(x=tspan, y=data2)

    plt = plot([e1_scatter, e2_scatter],
    Layout(
        xaxis_title = "t (μs)",
        yaxis_title = "<n>",
        showlegend = false
        )
    )
    display(plt)
    return data1, data2, plt
end


#= Run the dynamical evolution of states, depending if it is unitary or dissipative =#
function dynamic_evolution(time, ψ0, dynamics_input, type_dynamics::Symbol)
    tspan = time[1]:time[2]:time[end]
    dt = time[2]

    # Define a mapping from symbol to function
    dynamics_map = Dict(
        :schroedinger => timeevolution.schroedinger,
        :schroedinger_dynamic => timeevolution.schroedinger_dynamic,
        :master => timeevolution.master,
        :master_dynamic => timeevolution.master_dynamic
    )

    # Get the appropriate function
    evolve_func = dynamics_map[type_dynamics]

    return evolve_func(tspan, ψ0, dynamics_input;
    adaptive=false, 
    dt=dt,
    reltol=1e-9,
    abstol=1e-9
    )
end

function Quantum_solver_ODE(prob)
    sol = solve(prob, Tsit5())

    return sol.t, sol.u
end

########################################################################################################


###############
#= QM PULSES =#
###############
#= pi-pulse =#
function π_pulse_shape(t, t0, duration, eps=1e-12)
    δt = t - t0
    if 0.0 <= δt < duration
        s = sin(pi * δt / duration)^2
        return s / (s + eps^2)
    else
        return 0.0
    end
end

##################
#= INFIDELITIES =#
##################

#= mixed-state infidelity for density matrices=#
function qo_infidelity(ρ::Operator{B1, B2, T}, σ::Operator{B1, B2, T}) where {B1, B2, T}
    return 1.0 - min(real(QuantumOptics.fidelity(ρ, σ)), 1)
end

#= mixed-state infidelity for pure kets =#
function qo_infidelity(ψ1::Ket{B, T}, ψ2::Ket{B, T}) where {B,T}
    ρ = ψ1 ⊗ dagger(ψ1)
    σ = ψ2 ⊗ dagger(ψ2)

    return 1.0 - min(real(QuantumOptics.fidelity(ρ, σ)), 1)
end



function in_qo_infidelity(states::Vector{Any}, target_state::Ket{B, T}) where {B, T}
    return [qo_infidelity(state, target_state) for state in states]
end

function in_qo_infidelity(states::Vector{Any}, target_state::Operator{B1, B2, T}) where {B1, B2, T}
    return [qo_infidelity(state, target_state) for state in states]
end


##############################################################################################################################################
################################
#= SAMPLED DATASET GENERATION =#
################################

#= Final call for dataset dataset_creation =#
function dataset_creation(input_data, output_data) #ok
    len_dataset = length(input_data)
    dim_input = length(input_data[1])
    dim_output = length(output_data[1])

    total_dim = dim_input + dim_output

    dataset = Vector{Vector{Float64}}(undef, len_dataset)

    for i in 1:len_dataset
        row = Vector{Float64}(undef, total_dim)

        for j in 1:dim_input
            row[j] = input_data[i][j]
        end
        for k in 1:dim_output
            row[dim_input + k] = output_data[i][k]
        end
        dataset[i] = row
    end
    return dataset
end

#= This function needs to be recalled to generate the training/testing dataset for the NN =#
function dataset_generation_v2(dataset_features) #ok
    n_samples, n_training = dataset_features.len_dataset

    input_func, output_func = dataset_problems_dictionary[dataset_features.problem]

    #Outputs
    outputs = output_func(dataset_features.pr, dataset_features.parameters_range, dataset_features.dim_parameters_space, n_samples)

    #Inputs
    inputs = input_func(dataset_features.t0, dataset_features.initial_state, dataset_features.problem_features, outputs, n_samples, dataset_features.dynamics)


    #NN_input and NN_output preparation
    n_input, n_output = dataset_features.dim_dataset
    dataset_matrix = Float64.(reduce(hcat, dataset_creation(inputs, outputs))')

    return  dataset_matrix[1:n_training, 1:n_input], 
            dataset_matrix[1:n_training, (n_input + 1):end],
            dataset_matrix[(n_training+1):end, 1:n_input],
            dataset_matrix[(n_training+1):end, (n_input + 1):end]

end


###################################################################################################
###################################
#= PARAMETERS SPACE FOR SAMPLING =#
###################################
function twoD_parameter_space(p, parameters_range, dim_parameters_space)
    para1 = LinRange(parameters_range[1][1], parameters_range[1][2], dim_parameters_space[1])
    para2 = LinRange(parameters_range[2][1], parameters_range[2][2], dim_parameters_space[2])

    parameters_space = vec([(x, y) for x in para1, y in para2])
    prob = vec([p(x,y) for (x,y) in parameters_space])

    return parameters_space, prob
end

function threeD_parameter_space(p, parameters_range, dim_parameters_space)
    para1 = LinRange(parameters_range[1][1], parameters_range[1][2], dim_parameters_space[1])
    para2 = LinRange(parameters_range[2][1], parameters_range[2][2], dim_parameters_space[2])
    para3 = logrange(parameters_range[3][1], parameters_range[3][2], dim_parameters_space[3])

    parameters_space = vec([(x, y, z) for x in para1, y in para2, z in para3])
    prob = vec([p(x,y,z) for (x,y,z) in parameters_space])

    return parameters_space, prob
end

function sampling_parameter_space(p, parameters_range, dim_parameters_space)
    para1 = LinRange(parameters_range[1][1], parameters_range[1][2], dim_parameters_space[1])
    para2 = LinRange(parameters_range[2][1], parameters_range[2][2], dim_parameters_space[2])
    para3 = logrange(parameters_range[3][1], parameters_range[3][2], dim_parameters_space[3])

    parameters_space = vec([(x, y, z) for x in para1, y in para2, z in para3])
    prob = vec([p(x,y,z) for (x,y,z) in parameters_space])

    return parameters_space, prob
end


#####################################################################################################


##################################################################################################
##################
#= MISCELLANOUS =#
##################
function recomposition(vector)
    exit = ComplexF64[]
    N = length(vector)

    for k in 1:2:N
        push!(exit, vector[k] + im*vector[k+1])
    end

    return exit
end
function to_real_vec(vector)
    real_imag = Vector{Float64}(undef, 2 * length(vector))
    for i in eachindex(vector)
        real_imag[2i - 1] = real(vector[i])
        real_imag[2i]     = imag(vector[i])
    end
    return real_imag
end


function dm2ket(ρ; atol=1e-6)
    vals, vecs = eigenstates(ρ)
    for (i, val) in enumerate(vals)
        if isapprox(val, 1; atol=atol)
            return normalize(vecs[i])
        end
    end
    error("This density matrix does not represent a pure state.")
end

function trajectory_plot(dataset_features, problem_features, parameters) #ok
    plots = []
    trajectories = []

    #get the problem
    dynamic_func = plot_problem_dictionary[dataset_features.problem]
    
    for k in 1:size(parameters)[1]
        traces = Vector{GenericTrace{Dict{Symbol, Any}}}[]

        params = parameters[k,:]
        if dataset_features.problem == :FL_1step
            params = [params[1], π / params[1], params[2]]
        end
        
        tspan, ρ_out, τ_exc, τ_SWAP = dynamic_func(dataset_features.t0, dataset_features.initial_state, params, dataset_features.dynamics, problem_features, :plot) #[typeofcorrection, n_phonon, :plot]) 

        push!(trajectories, ρ_out[end])

        expectation_qub = []
        expectation_mech = []
        for ρ in ρ_out
            push!(expectation_qub, expectation_value(qubit_mech.n_qubit , ρ))
            push!(expectation_mech, expectation_value(qubit_mech.n_mech , ρ))
        end
        
        exp_σz = PlotlyJS.scatter(
            x = tspan,
            y = expectation_qub,
            mode = "lines",
            name = "<σz>"
        )
        exp_n = PlotlyJS.scatter(
            x = tspan,
            y = expectation_mech,
            mode = "lines",
            name = "<n>"
        )

        layout = PlotlyJS.Layout(
        xaxis_title="Time",
        yaxis_title="Expectation Value",
        showlegend = false
        )

        plt = PlotlyJS.plot([exp_σz, exp_n], layout)
        PlotlyJS.display(plt)
        println("τ_exc = ", τ_exc, ", τ_SWAP = ", τ_SWAP)
        push!(plots, plt)

        if size(parameters)[1] == 1
            return trajectories, expectation_qub, expectation_mech
        end
    end
end

#################################################################
                         #= BSplines =# 
#################################################################
function generate_Bspline_basis(degree::Int, n_basis::Int, domain::Tuple{Float64, Float64})
    knots = range(domain[1], domain[2], length = n_basis - degree + 2)
    basis = BSplineBasis(BSplineOrder(degree), knots)
    return basis
end

# === 2. Compose a spline function from coefficients ===
function Bspline_composition(coeffs::Vector{Float64}, basis::BSplineBasis)
    spline = Spline(basis, coeffs)  # Spline object
    return x -> spline(x)           # Evaluate by calling it like a function
end

function drive_from_normalized_spline(spline, T::Float64)
    return t -> spline(t / T)  # t in [0, T] → τ in [0, 1]
end



####################################################################################
####################################################################################
####################################################################################
####################################################################################
                               # ML-NN section =#
####################################################################################


########################
#= Data normalization =#
########################
normalization(x, x_max, x_min) = (x - x_min) / (x_max - x_min)
denormalization(x, x_max, x_min) = x*(x_max - x_min) + x_min

function max_min(data)
    len_dataset = length(data)[1]
    dim_dataset = size(data)[2]
    maxs, mins = Float64[], Float64[]
    
    for i in 1:dim_dataset
        push!(maxs, maximum( data[:,i] ))
        push!(mins, minimum( data[:,i] ))
    end
    
    return maxs, mins
end


function normalize_data(data::Matrix{Float64}, maxs, mins)
    data_copy = deepcopy(data)
    len_dataset = size(data_copy)[1]
    dim_dataset = size(data_copy)[2]

    results = Matrix{Float64}(undef, len_dataset, dim_dataset)
    
    for k in 1:dim_dataset
        results[:,k] = normalization.(data_copy[:,k], maxs[k], mins[k])
    end

    return results
end

function normalize_data(data::Vector{Float64}, maxs, mins)
    data_copy = deepcopy(data)
    dim_dataset = size(data_copy)[1]

    results = Vector{Float64}(undef, dim_dataset)
    
    for k in 1:dim_dataset
        results[k] = normalization(data_copy[k], maxs[k], mins[k])
    end

    return results
end

function denormalize_data(data::Vector{Float64}, maxs, mins)
    len_dataset = size(data)[1]
    results = Vector{Float64}(undef, len_dataset)
    
    for k in 1:len_dataset
        results[k] = denormalization(data[k], maxs[k], mins[k])
    end
    return results
end

function denormalize_data(data::Matrix{Float64}, maxs, mins)
    data_copy = deepcopy(data)
    len_dataset = size(data_copy)[1]
    dim_dataset = size(data_copy)[2]

    results = Matrix{Float64}(undef, len_dataset, dim_dataset)
    
    for k in 1:dim_dataset
        results[:,k] = denormalization.(data_copy[:,k], maxs[k], mins[k])
    end
    return results
end

function train_test_dataset_normalization(dataset_vector)
    train_input, train_output, test_input, test_output = dataset_vector


    maxs_input, mins_input = max_min(train_input)
    maxs_output, mins_output = max_min(train_output)

    #=
    maxs_input, mins_input = max_min(vcat(train_input, test_input))
    maxs_output, mins_output = max_min(vcat(train_output, test_output))

    =#
    #Training dataset normalization
    normalized_train_input = normalize_data(train_input, maxs_input, mins_input)
    normalized_train_output = normalize_data(train_output, maxs_output, mins_output);
    

    #Testing dataset normalization
    normalized_test_input = normalize_data(test_input, maxs_input, mins_input)
    normalized_test_output = normalize_data(test_output, maxs_output, mins_output);
    
    return normalized_train_input, normalized_train_output, normalized_test_input, normalized_test_output

end

function dataset_variance(data::Matrix{Float64})
    return  var.([data[:,k] for k in 1:size(data)[2]])
end

##########################
#= Data standardization =#
##########################
standardization(x, μ, σ) = (x - μ) / σ

function mean_variance(data)
    len_dataset = size(data)[1]
    dim_dataset = size(data)[2]

    means, variances = Float64[], Float64[]
    
    for i in 1:dim_dataset
        push!(means, mean(data[:,i]))
        push!(variances, (1/len_dataset)*  sum((data[:,i] .- means[end]).^2))
    end
    
    return means, variances
end


function standardize_data(data::Matrix{Float64}, means, variances)
    data_copy = deepcopy(data)
    len_dataset = size(data_copy)[1]
    dim_dataset = size(data_copy)[2]

    results = Matrix{Float64}(undef, len_dataset, dim_dataset)
    
    for k in 1:dim_dataset
        results[:,k] = standardization.(data_copy[:,k], means[k], variances[k])
    end

    return results
end


##################################
#= Dataset saving and importing =#
##################################

function saving_dataset(dataset_vector, names_vector)
    X_train, Y_train, X_test, Y_test = dataset_vector
    Xtr_name, Ytr_name, Xte_name, Yte_name = names_vector

    train_input_CSV = DataFrame(X_train, :auto);
    CSV.write(Xtr_name, train_input_CSV)

    train_output_CSV = DataFrame(Y_train, :auto);
    CSV.write(Ytr_name, train_output_CSV)

    test_input_CSV = DataFrame(X_test, :auto);
    CSV.write(Xte_name, test_input_CSV)

    test_output_CSV = DataFrame(Y_test, :auto);
    CSV.write(Yte_name, test_output_CSV);

end


function importing_dataset(names_vector)
    Xtr_name, Ytr_name, Xte_name, Yte_name = names_vector
    
    train_input = Matrix{Float64}(CSV.read(Xtr_name, DataFrame))
    train_output = Matrix{Float64}(CSV.read(Ytr_name, DataFrame))

    test_input = Matrix{Float64}(CSV.read(Xte_name, DataFrame))
    test_output = Matrix{Float64}(CSV.read(Yte_name, DataFrame));

    return train_input, train_output, test_input, test_output 

end



#################################
#= Differentiable Loss function=#
#################################
#̂y is predicted, y is expected
function relative_mse(ŷ, y; ϵ = 1e-12)
    return mean(((ŷ .- y).^2) ./ (y .+ ϵ).^2)
end

function mape(ŷ, y; ϵ = 1e-12)
    return mean(
        abs.((ŷ .- y) ./ (y .+ ϵ))
    )
end

function loss_1(ŷ, y)
    return 0.5*( abs(ŷ[1] - y[1])^(0.5) + abs(ŷ[2] - y[2])^(0.5) )

end



##############
#= Training =#
##############
function training!(model, epochs::Int, type_loss_function::Symbol, train_X, train_Y)
    N = size(train_X)[1]
    losses = []


    loss_functions = Dict(
        :mse => Flux.Losses.mse,
        :relative_mse => relative_mse,
        :mape => mape,
        :enhancing1 => loss_1,
        :mae => Flux.Losses.mse
    )

    loss_func = loss_functions[type_loss_function]


    @showprogress for epoch in 1:epochs
        error_tot = 0.0
        for (x, y) in zip(eachrow(train_X), eachrow(train_Y))
            loss_value, grads = Flux.withgradient(model) do m
                loss_func(m(x), y) 

            end
            
            Flux.Optimisers.update!(opt, model, grads[1])
            error_tot += loss_value / N
        end
        push!(losses, error_tot)
    end
    return losses
end




#############
#= Testing =#
#############

#= Infidelity on the test need to be implemented =#
function testing_model_infidelity(model, type_loss_function::Symbol, test_input, test_output)
    n_test = size(test_input)[1]
    dim_output = size(test_output)[2]
    error = 0.0
    #infidelities = Float64[]
    predictions = Matrix{Float64}(undef, n_test, dim_output)

    #dynamics_inputs = create_input_dynamics(H0)

    loss_functions = Dict(
        :mse => Flux.Losses.mse,
        :relative_mse => relative_mse,
        :mape => mape,
        :enhancing1 => loss_1,
        :mae => Flux.Losses.mse
    )
    loss_func = loss_functions[type_loss_function]


    for i in 1:n_test
        prediction = model(test_input[i,:])
        predictions[i,:] =  prediction
        error += loss_func(prediction, test_output[i,:]) 
        #=
        push!(infidelities, 
        qo_infidelity(
            recomposition(test_input[i,1:end-2]), 
            dynamics_from_π_SWAP_pulses(dynamics_inputs, [prediction[1], π / prediction[1], prediction[2]], :schroedinger_dynamic)[2].data
            )
        )
        =#
    end

    return error / n_test, predictions #, mean(infidelities)
end



###################
#= NN prediction =#
###################
function identity_func(x, y, z)
    return x
end

function prediction_normalize_data(x, maxs_input, mins_input)
    return normalize_data(x, maxs_input, mins_input)

end

function prediction_denormalize_data(x, maxs_output, mins_output)
    return denormalize_data(x, maxs_output, mins_output)

end


function prediction_infidelity(dataset_features, NN_features)
    t0, ψ0, prediction_input, target_state, type_dynamics, type_dataset, typeofcorrection, n_phonon, n_output, maxs_mins = NN_features.features_prediction
    maxs_input, mins_input, maxs_output, mins_output = maxs_mins

    normalization_dictionary = Dict(
        :unormalized => [identity_func, identity_func],
        :normalized => [prediction_normalize_data, prediction_denormalize_data]
    )
    dataset_func_1, dataset_func_2 = normalization_dictionary[type_dataset]

    #parameters = denormalize_data(model(target_input), maxs_output, mins_output)
    predicted_drives = reshape(dataset_func_2(NN_features.model(dataset_func_1(prediction_input, maxs_input, mins_input)), maxs_output, mins_output), 1, n_output)
    #println("Optimal τ_exc = ", predicted_drives[1], ",  and optimal τ_SWAP = ", predicted_drives[2])

    #predicted_drives = reshape([parameters[1], π / parameters[1], parameters[2]], 1, 3) #1x3 matrix
    predicted_trajectory, n_qub, n_mech = trajectory_plot(dataset_features, dataset_features.problem_features,  predicted_drives)

    final_mech = n_mech[end]
    infidelity_error = qo_infidelity(predicted_trajectory[1], target_state)
    

    println("Prediction error = infidelity(optimal_state, target) = $infidelity_error")
    println("<n> on optimal state = $final_mech")
     

    return predicted_drives, predicted_trajectory[1], infidelity_error

end


function train_test_prediction!(dataset_vector, dataset_features, NN_features)
    train_input, train_output, test_input, test_output = dataset_vector

    D_predictions = Dict(
        :prediction_infidelity => prediction_infidelity
    )


    #NN training
    println("Progress of the training")
    losses = training!(NN_features.model, NN_features.N_epochs, NN_features.loss_func, train_input, train_output)

    #Plotting the loss funciton
    epochs = 1:NN_features.N_epochs
    #Define trace
    trace_ = scatter(x=epochs, y=losses, mode="lines", name="Loss")

    # Define layout with log-scale y-axis
    layout = Layout(
        title="Training Loss (Logarithmic Scale)",
        xaxis=attr(title="Epoch"),
        yaxis=attr(title="Loss", type = "log")
    )
    step1 = plot([trace_], layout)
    display(step1)

    

    #NN Test
    errors, predictions = testing_model_infidelity(NN_features.model, NN_features.loss_func, test_input, test_output)

    println("Training loss = ", losses[end])
    println("Test error =", errors)



    #Prediction
    prediction_func = D_predictions[NN_features.type_prediction]
    predicted_parameters, predicted_final_state, infidelity_error =  prediction_func(dataset_features, NN_features)

    return losses, errors, predictions, predicted_parameters, predicted_final_state, infidelity_error

end

###############################################################################################
###############################################################################################
###############################################################################################

#= Final algorithm executing 
1. dataset generation,
2. training, 
3. testing,
4. prediction. =#
function execute_problem_NN(dataset_features, NN_features)
    #The function get the problem from the dataset_features

    #dataset generation, saving or importing
    if dataset_features.modality_dataset == :generating
        println("Dataset generation")
        train_input, train_output, test_input, test_output = dataset_generation_v2(Dataset_features)
        dataset_vector = [train_input, train_output, test_input, test_output]

    elseif dataset_features.modality_dataset == :generating_and_saving
        println("Dataset generation")
        train_input, train_output, test_input, test_output = dataset_generation_v2(Dataset_features)
        dataset_vector = [train_input, train_output, test_input, test_output]
        saving_dataset(dataset_vector, dataset_features.names_dataset)

    else
        dataset_vector = importing_dataset(dataset_features.names_dataset)

    end

    #normalization
    if dataset_features.norm_dataset == :normalized
        dataset_vector = train_test_dataset_normalization(dataset_vector)


        maxs_input, mins_input = max_min(train_input)
        maxs_output, mins_output = max_min(train_output)

        NN_features.features_prediction[end] = [maxs_input, mins_input, maxs_output, mins_output]

    end

    #NN t-t-p
    losses, test_error, test_predictions, predicted_parameters, ρ_1, infidelity_prediction = train_test_prediction!(dataset_vector, dataset_features, NN_features);

    return dataset_vector, predicted_parameters, ρ_1, infidelity_prediction
end



