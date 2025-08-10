#= ML_QM_execution.jl READ ME
This .jl file contains all the necessary ingredients to run the NN-based method of prediction 
for quantum state preparation. Before execution, the user has to define the quantum mechanical
problem under analysis, by following the file problem_prototype.jl. Once all the steps there 
have been followed, the user can set all the settings and the parameters and then execute The
file. For debugging and control of the code's flow, the user can use a juputer notebook to 
execute the code.
=#

#=
##############################################
using Pkg
Pkg.activate(".")

#=
Pkg.add(
["QuantumOptics","Flux", "ProgressMeter", "Random", 
"Statistics", "SparseArrays", "Zygote", "LinearAlgebra", 
"StatsBase","DiffEqFlux","Optimisers","Compat",
"PlotlyJS","DataFrames", "CSV","BSON", "Revise", "Distributions"]
)
=#

Pkg.instantiate()
Pkg.status()

#############################################
#= Packages =#
using Flux, ProgressMeter, Random, Statistics, QuantumOptics, SparseArrays, StatsBase, LinearAlgebra, Revise
using Zygote, DifferentialEquations, SciMLSensitivity, DiffEqFlux, Optimisers, Compat, PlotlyJS, CSV, DataFrames, BSON
using Distributions, BSplineKit
############################################



#############################################
################################################
  #= PRELIMINARY DEFINITIONS AND DYNAMICS =#
################################################
N_mech = ...; #Cut-off Fock basis for the mech resonator
dimension = ...; #Ket state vector's dimension

#BSpline definition
degree_spline = ...            # spline order (cubic spline)
n_basis_spline = ...         # number of basis functions
domain_spline = ...  # interval for the spline
#####################################################################################

include("definitions.jl")

#= Loading personal libraries =#
include("HBAR-qubit_problem.jl")
include("ML_QM_library.jl")

basis_spline = generate_Bspline_basis(degree_spline, n_basis_spline, domain_spline)
####################################################################################




#########################################################
#= Decomposition basis =#
#= SUN BASIS =#

SUN_basis = rand_hermitian_orthonormal_basis(dimension, basis);

#=
#for saving
SUN_basis_matrices = [matrix.data for matrix in SUN_basis]
SUN_basis_CSV = DataFrame(matrices = SUN_basis_matrices)
CSV.write("SUN_basis_matrices.csv", SUN_basis_CSV)
=#
#=
df = CSV.read("SUN_basis_matrices.csv", DataFrame)
SUN_basis_nop = [eval(Meta.parse(df.matrices[i])) for i in 1:length(df.matrices)]
SUN_basis = [Operator(basis, matrix) for matrix in SUN_basis_nop];
=#
##########################################################






###########################################################
###########################################################
n_pulses = ...
n_input, n_output = ...

######################################################################################
#= Dynamic features and initial conditions =#
step_number = ...

t0 = ...
ψ0 = ...

type_of_problem = ...
type_of_dynamics = ...
type_of_correction = ...
norm_dataset = ...

#= dimension of training+testing dataset =#
n_samples = ...
n_training = ...

#= Physical parameters =#
...
...
...


#= probability distribution for sampling parameters space =#
prs = ...
dim_x, dim_y, dim_z, ... = ...

#= Features (see definitions.jl for further information) =#
#
Step_states = creation_step_states(
    ψ0,
    ...
)

FL_1step_2drives_features = fl_1step_features(
...
)


Dataset_features = dataset_features(
    ...
)

###############################################################################
#= NN =#

#= Main features =#
η = ...
N_epochs = ...

#= Model_input =#
# Set seed for reproducibility
Random.seed!(42)

# Define neural network model
model = Flux.f64(
    ...
    )
)

target_input = ... 
NN_features = fl_1step_nn_features(
...
)

opt = Flux.Optimise.setup(Flux.Optimisers.Adam(NN_features.η), NN_features.model)

#= EXECUTE ALGORITHM =#
dataset_vector, predicted_pulses, final_state, infidelity_prediction = execute_problem_NN(Dataset_features, NN_features);

####################################################################################################################








####################################################################################################################
# Dataset creation test

######################################################################################
#= Dynamic features and initial conditions =#
step_number = 1

t0 = ...
ψ0 = ...

type_of_problem = ...
type_of_dynamics = ...
type_of_correction = ...
norm_dataset = ...
n_phonon = ...

#= dimension of training+testing dataset =#
n_samples = ...
n_training = ...

#= Physical parameters =#
...
...
...


#= probability distribution for sampling parameters space =#
prs = ...
dim_x, dim_y, dim_z, ... = ...

#= Features (see definitions.jl for further information) =#
#
Step_states = creation_step_states(
    ψ0,
    ...
)

FL_1step_2drives_features = fl_1step_features(
        ...
)


Dataset_features = dataset_features(
    ...
)
###################################################
train_input, train_output, test_input, test_output = dataset_generation(Dataset_features)

#= Saving the dataset =#
#saving_dataset(dataset_vector, names_vector)

#= Importing the dataset =#
#train_input, train_output, test_input, test_output = importing_dataset(names_vector)
#=
#= Normalization of the dataset =#
normalized_train_input, normalized_train_output, normalized_test_input, normalized_test_output = train_test_dataset_normalization(dataset_vector)
normalized_dataset_vector = [normalized_train_input, normalized_train_output, normalized_test_input, normalized_test_output]
=#

=#