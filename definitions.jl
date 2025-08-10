using QuantumOptics
####################################################################################################################
###############
#= Constants =#
###############
# Physical constants in SI units

const h = 6.62607015e-34          # Planck constant (J·s)
const hbar = 1.054571817e−34;     # Reduced Planck constant (J·s)
const kb = 1.380649e-23           # Boltzmann constant (J/K)

####################################################################################################################



            #= STRUCTs =# 
####################################################################################################################
#####################################
#= QM SYSTEMS WITHIN QUANTUMOPTICS =#
#####################################

#= Struct to define the different qm systems -> agebra operators, basis =#
struct ho{ho_basis}
    basis::ho_basis
    a::Operator{ho_basis, ho_basis, <:AbstractMatrix} 
    ad::Operator{ho_basis, ho_basis, <:AbstractMatrix} 
    n::Operator{ho_basis, ho_basis, <:AbstractMatrix} 
    Id::Operator{ho_basis, ho_basis, <:AbstractMatrix}
end

struct qubit
    basis::SpinBasis{1//2, Int64}
    σm::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σp::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σx::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64},<:AbstractMatrix}
    σy::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    σz::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
    Id::Operator{SpinBasis{1//2, Int64}, SpinBasis{1//2, Int64}, <:AbstractMatrix}
end

struct qub_ho{ho_basis}
    zI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    xI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    mI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    pI::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    II::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    pa::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    mad::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}

    Ia::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    Iad::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}

    n_mech::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
    n_qubit::Operator{CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, CompositeBasis{Vector{Int64}, Tuple{SpinBasis{1//2, Int64}, ho_basis}}, <:AbstractMatrix}
end



#Built-in function to fill the structs =#
function Harmonic_oscillator(N_particle::Int64, type_basis::Symbol)
    basis_Dict = Dict(
        :FockBasis => FockBasis
    )

    basis_type = basis_Dict[type_basis]
    basis = basis_type(N_particle)

    return ho(basis,
    destroy(basis), 
    create(basis), 
    number(basis),
    one(basis)
    )

end

function Qubit(spin)
    basis = SpinBasis(spin)

    return qubit(basis,
    sigmam(basis),
    sigmap(basis),
    sigmax(basis),
    sigmay(basis),
    sigmaz(basis),
    one(basis)
    )
    
end


function Qubit_HO(N_mech, type_basis_mech::Symbol, type_basis_qubit)
    qub = Qubit(type_basis_qubit)
    mech_res = Harmonic_oscillator(N_mech, type_basis_mech)
    
    return qub, mech_res,
            qub_ho(
            tensor(qub.σz, mech_res.Id),
            tensor(qub.σx, mech_res.Id),
            tensor(qub.σm, mech_res.Id),
            tensor(qub.σp, mech_res.Id),
            tensor(qub.Id, mech_res.Id),
            tensor(qub.σm, mech_res.a),
            tensor(qub.σp, mech_res.ad),
            tensor(qub.Id, mech_res.a),
            tensor(qub.Id, mech_res.ad),
            tensor(qub.Id, mech_res.n),
            0.5 * (tensor(qub.Id, mech_res.Id) + tensor(qub.σz, mech_res.Id))
           )
end


######################################################################################################
#= Structs for NN_ML =#

struct dataset_features
    problem::Symbol 
    len_dataset::Vector{Int64} #len_dataset = [n_samples, n_training]
    dim_dataset::Vector{Int64} #dim_dataset = [n_input, n_output]
    dim_parameters_space::Vector{Int64}
    parameters_range::Vector{Vector{Float64}}
    pr::Any
    dynamics::Symbol
    t0::Float64
    initial_state::Any
    problem_features::Any
    names_dataset::Vector{Any}
    modality_dataset::Symbol
    norm_dataset::Symbol
end

struct step_states
    ψ0::Ket
    ρ0::Any
    ψ_target_spinflip::Ket
    ψ_target_1step::Ket
    ρ_target_spinflip::Any
    ρ_target_1step::Any
end

struct fl_1step_features
    state_target_spinflip::Any
    state_target_1step::Any
    decom_basis::Any
    phonon_n::Int64
    correction::Symbol
end

struct fl_1step_nn_features
    model::Any
    N_epochs::Int64
    η::Float64
    target_input_1step::Any
    features_prediction::Vector{Any}
    loss_func::Symbol
    type_prediction::Symbol

end


