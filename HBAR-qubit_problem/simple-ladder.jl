using ArgParse

using ProgressMeter, QuantumOptics, DelimitedFiles, Printf
N_mech = 15; #Cut-off Fock basis for the mech resonator
N_steps = 10

include("../definitions.jl")
include("HBAR-qubit_problem.jl")
include("../ML_QM_library.jl")

function save_vector_to_csv(vector, filename::String; header="Header")
    #Write vector to file, with new line as delimiter
    writedlm(filename, [header;vector], "\n")
end

function save_vector_to_csv(vector, filename::String; header="Header")
    #Write vector to file, with new line as delimiter
    writedlm(filename, [header;vector], "\n")
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--g_rel", "-g"
            help = "Coupling relative to detuning"
            arg_type = Float32
            required = true
        "--A_rel", "-A"
            help = "Drive amplitude relative to detuning"
            arg_type = Float32
            required = true
        "--dynamics", "-d"
            help = "Type of dynamics (unitary/non-unitary)"
            arg_type = String
            required = true
        "--correction", "-c"
            help = "Add mean field correction (yes/no)"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function run_simulation(g_rel, ΩR_rel, time_evo, corr)
    println("Running with options: g_rel $(g_rel), ΩR_rel $(ΩR_rel), time_evo $(time_evo), corr $(corr)...")
    global Δ0 = (g/g_rel)
    ΩR = abs(ΩR_rel*Δ0) # These are already scaled through g

    
    type_of_dynamics = time_evo
    type_of_correction = corr

    pulse_parameters = [[π / ΩR, π / (2*g*sqrt(n))] for n in 1:N_steps]
    initial_state = tensor(spindown(qub.basis),fockstate(mech_res.basis, 0))
    
    if type_of_dynamics ==:schroedinger_dynamic
        target_states = [tensor(spindown(qub.basis),fockstate(mech_res.basis, n)) for n in 1:N_steps]
    elseif type_of_dynamics ==:master_dynamic
        target_states = [dm(tensor(spindown(qub.basis),fockstate(mech_res.basis, n))) for n in 1:N_steps]
    else 
        error("Illegal time_evo option")
    end
    
    chu_protocol_features1 = fl_1step_features(
        [0.0],
        target_states, 
        [0], 
        0, 
        type_of_correction
    )
    
    time_final, solution_final, infidelities_ = dynamics_n_steps_FL(N_steps, 
        initial_state,
        pulse_parameters, 
        type_of_dynamics, 
        chu_protocol_features1
    ) 

    g_dir = replace("g_$(@sprintf("%.2f", g_rel))", "." => "_")
    
    save_vector_to_csv(infidelities_, "$(pwd())/data/simple_ladder/$(time_evo)/$(corr)/$(g_dir)/Chu_g$(g_rel)D_A$(ΩR_rel)D_$(corr)_$(time_evo).csv", header = "Infidelities")
    println("Done.")
end 

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    println("G rel is ", parsed_args["g_rel"])

    #Mapping between command line options and symbols used internally
    dynamics = Dict{String, Symbol}([
        "unitary" => :schroedinger_dynamic,
        "non-unitary" => :master_dynamic,
    ])
    correction = Dict{String, Symbol}([
        "yes" => :correction_on,
        "no" => :correction_off,
    ])

    #println("Options: ", parsed_args["g_rel"], parsed_args["A_rel"], dynamics[parsed_args["dynamics"]], correction[parsed_args["correction"]] )

    run_simulation(parsed_args["g_rel"], parsed_args["A_rel"], dynamics[parsed_args["dynamics"]], correction[parsed_args["correction"]])
end

main()