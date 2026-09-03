# NNQuantum.jl
#
# NNQuantum's NN/dataset library file — DESIGN.md Part 6, sub-step (i).
#
# Takes over the role Part 4 flagged as open when it "provisionally
# colocated" qo_infidelity/rand_hermitian_orthonormal_basis/
# threeD_parameter_space inside FockLadder_problem.jl "since no library
# file exists yet": this is that file. These three are problem-agnostic —
# nothing here references :FL_1step_3p, τ_exc/ωd/τ_SWAP, or any other
# HBAR-qubit-specific quantity — so they move here unchanged, out of the
# problem file.
#
# weighted_sample stays in FockLadder_problem.jl: DESIGN.md Part 4 treats
# it separately from this "generic helpers" trio ("One deliberate
# non-port", its own paragraph, not grouped with these three), so it
# wasn't moved here without being asked to.
#
# DESIGN.md Part 6 (ii) adds the actual training loop / normalization /
# loss dispatch / prediction orchestration below, ported from old_version/
# ML_QM_library.jl and rebased onto this project's data shapes — see that
# section's own header comment further down for what changed in the rebase.

using QuantumOptics   # fidelity/expect/Ket/Operator/dm — qo_infidelity
using LinearAlgebra   # qr — rand_hermitian_orthonormal_basis
using Statistics      # mean — relative_mse/mape
using Random          # shuffle — train_model!'s per-epoch minibatching
using JLD2            # jldopen — load_dataset
using Flux            # withgradient/Optimisers.update!/Losses — train_model!

# Mixed-state infidelity, 1 - min(fidelity,1) — ported from old_version/
# ML_QM_library.jl:203-213, but collapsed from two separate Ket/Operator
# methods into one via `_as_density_operator`: FLstep_dynamics_3p's own
# states are always density operators (J is always passed, per (f)), while
# a caller-supplied target state may still be given as a plain Ket.
_as_density_operator(ψ::Ket) = dm(ψ)
_as_density_operator(ρ::Operator) = ρ

function qo_infidelity(a, b)
    return 1.0 - min(real(fidelity(_as_density_operator(a), _as_density_operator(b))), 1.0)
end

# Ported unchanged (mod. renaming the inner loop variable that shadowed the
# `basis` argument in old_version) from old_version/ML_QM_library.jl:65-97.
# Generalizes Gell-Mann matrices to arbitrary dimension: d^2 random
# Hermitian matrices, projected to an orthonormal set via QR of their
# stacked real/imaginary-part vectorization, then re-Hermitized to correct
# numerical QR error.
function rand_hermitian_orthonormal_basis(d::Int, basis)
    n = d^2
    mats = Matrix{ComplexF64}[]
    for _ in 1:n
        A = randn(ComplexF64, d, d)
        push!(mats, (A + A') / 2)
    end

    vecs = zeros(Float64, 2 * d * d, n)
    for (i, H) in enumerate(mats)
        vecs[1:d*d, i]     = real(vec(H))
        vecs[d*d+1:end, i] = imag(vec(H))
    end

    Q, _ = qr(vecs)
    Q = Matrix(Q)   # materialize once: Q from qr() is a lazy AbstractQ
                     # (Householder-reflector representation) — indexing it
                     # n times below, as old_version's own loop did, would
                     # re-derive each requested column from the reflectors
                     # on every access instead of reading a plain array

    half = d * d
    ops = Matrix{ComplexF64}[]
    for i in 1:n
        real_part = reshape(Q[1:half, i], d, d)
        imag_part = reshape(Q[half+1:end, i], d, d)
        H = real_part + im * imag_part
        push!(ops, (H + H') / 2)
    end

    return [Operator(basis, H) for H in ops]
end

# --- Operator-basis persistence (JLD2) ------------------------------------------
#
# rand_hermitian_orthonormal_basis is unseeded — every call produces a
# genuinely different basis. That's harmless when it's built once and reused
# for everything within a single run, but breaks any later attempt to reuse
# a dataset or a trained NN *across* separate runs: both are only meaningful
# relative to the exact basis their feature vectors were built against, and
# without this, a reload would silently regenerate a different one instead
# of erroring — a real bug caught this way in run_Fock_ladder's own
# dataset/NN reuse feature (DESIGN.md). Same "reconstructible spec, not the
# raw struct" convention as save_dataset/save_nn: stores each operator's
# plain matrix data, not the `Operator` struct itself, and rebuilds against
# whatever basis the caller supplies at load time (its own `cs.basis`, not
# re-derived from the file).
const BASIS_FORMAT_VERSION = 1

"""
    save_operator_basis(path, ops; description="", params=Dict{Symbol,Any}())

Save a vector of operators (e.g. `rand_hermitian_orthonormal_basis`'s own
return value) to `path` (a `.jld2` file).
"""
function save_operator_basis(path::AbstractString, ops::AbstractVector;
                              description::AbstractString="", params::Dict{Symbol,Any}=Dict{Symbol,Any}())
    jldsave(path;
        format_version=BASIS_FORMAT_VERSION,
        data=[Matrix(op.data) for op in ops],
        description=description, params=params,
    )
    nothing
end

"""
    load_operator_basis(path, basis; expected_format_version=nothing)

Reload a basis saved by `save_operator_basis`, rebuilding each `Operator`
against `basis` (the caller's own, e.g. `cs.basis`) — not a basis recovered
from the file, since none is stored (bases aren't cheaply/robustly
serializable the way plain matrix data is, and the caller already has the
right one in hand). `expected_format_version`, if given, is checked the same
way `load_dataset`'s/`load_nn`'s own checks work.
"""
function load_operator_basis(path::AbstractString, basis; expected_format_version::Union{Nothing,Integer}=nothing)
    data, format_version = jldopen(path, "r") do file
        file["data"], file["format_version"]
    end
    if expected_format_version !== nothing && format_version != expected_format_version
        throw(ArgumentError(
            "load_operator_basis: $path has format_version=$format_version, expected $expected_format_version"))
    end
    return [Operator(basis, d) for d in data]
end

# Ported unchanged from old_version/ML_QM_library.jl:294-303 (needs Julia's
# Base.logrange for the third, log-spaced dimension — available since 1.11,
# no extra dependency).
function threeD_parameter_space(p, parameters_range, dim_parameters_space)
    para1 = LinRange(parameters_range[1][1], parameters_range[1][2], dim_parameters_space[1])
    para2 = LinRange(parameters_range[2][1], parameters_range[2][2], dim_parameters_space[2])
    para3 = logrange(parameters_range[3][1], parameters_range[3][2], dim_parameters_space[3])

    parameters_space = vec([(x, y, z) for x in para1, y in para2, z in para3])
    prob = vec([p(x, y, z) for (x, y, z) in parameters_space])

    return parameters_space, prob
end

# Direct continuous sampling of n_samples (x,y,z) triples uniformly within
# parameters_range's box (log-uniform in the third dimension — the same axis
# threeD_parameter_space itself log-spaces via Base.logrange), in O(n_samples)
# rather than threeD_parameter_space+weighted_sample's O(∏dim_parameters_space)
# dense-grid-then-weighted-draw. Only equivalent to that pair when the desired
# sampling probability really is uniform over the box, as every current
# caller's own `p` happens to be (e.g. FockLadder_problem.jl's `prs`, always
# `1/prod(dim_parameters_space)` — constant, independent of the point) — for a
# genuinely non-uniform p, threeD_parameter_space/weighted_sample is still the
# right tool; this function takes no probability function at all, deliberately,
# since one isn't needed under uniform sampling.
function uniform_parameter_sample(parameters_range, n_samples)
    lo1, hi1 = parameters_range[1]
    lo2, hi2 = parameters_range[2]
    lo3, hi3 = parameters_range[3]
    return [(lo1 + rand() * (hi1 - lo1),
             lo2 + rand() * (hi2 - lo2),
             lo3 * (hi3 / lo3)^rand())
            for _ in 1:n_samples]
end

# =============================================================================
# DESIGN.md Part 6 (ii): training loop / normalization / loss dispatch /
# prediction orchestration, ported from old_version/ML_QM_library.jl and
# rebased onto this project's data shapes. Not ported (out of scope for (ii),
# either superseded elsewhere in NNQuantum or dead/unused in old_version
# itself): gellmann_operators/pauli_operators, the PlotlyJS-based plotting
# functions and trajectory_plot (superseded by test.jl's CairoMakie
# convention), CSV-based saving_dataset/importing_dataset/save_matrices_basis
# (superseded by save_dataset's JLD2 convention, Part 4), standardize_data/
# mean_variance/dataset_variance (standardization utilities old_version
# itself never calls — train_test_dataset_normalization only ever used
# min-max), BSpline machinery (:FL_1step_2drives dropped per (a)),
# execute_problem_NN/dataset_generation/dataset_creation/dataset_problems_
# dictionary (old_version's Dict{Symbol,Function} problem-registration
# layer — there is exactly one problem/one dynamics function in NNQuantum,
# so this layer doesn't get rebased, it's dropped; see the re-simulation
# note below for what replaces its one live use).
#
# What "rebased" means concretely, per DESIGN.md Part 6 (ii): dataset
# loading below reads save_dataset's own JLD2 layout (dim_input/dim_output
# keys, Part 4) instead of old_version's in-memory dataset_features-driven
# dataset_generation; and predict_and_score's re-simulation/scoring step
# takes the dynamics function to call as a plain argument (`simulate`)
# instead of going through old_version's plot_problem_dictionary — there is
# only one dynamics function in this project (FLstep_dynamics_3p), so a
# caller just passes it directly, no Dict{Symbol,Function} dispatch layer
# to rebuild. This is also why predict_and_score itself stays here, in the
# problem-agnostic file, rather than moving to FockLadder_problem.jl: it
# never names FLstep_dynamics_3p, it only calls whatever `simulate` it's
# given.

# --- Dataset loading (JLD2), replaces old_version's in-memory dataset_generation split ---
#
# save_dataset (FockLadder_problem.jl, Part 4) already writes dim_input/
# dim_output alongside the stacked dataset matrix specifically so a reload
# doesn't need inputs/outputs still in memory — this is that reload.
#
# `expected_format_version`, if given, is checked against the file's own
# `format_version` key and raises loudly on a mismatch, rather than
# silently misinterpreting a differently-shaped dataset's columns — the
# same "surface problems instead of producing a quietly-wrong result"
# principle `QuantumDynamics/framework`'s own `io.jl` documents (DESIGN.md
# Part 2), which `save_dataset` already writes `format_version` to support
# but this function didn't actually check until now. Left `nothing`
# (no check) by default rather than hardcoded to any one problem's
# version constant (e.g. FockLadder_problem.jl's own DATASET_FORMAT_VERSION)
# — NNQuantum.jl doesn't know which problem's dataset it's loading, or
# what versioning scheme that problem uses, so the check is opt-in: a
# caller that cares passes its own expected value.
function load_dataset(path::AbstractString; expected_format_version::Union{Nothing,Integer}=nothing)
    dataset, format_version, dim_input, dim_output = jldopen(path, "r") do file
        file["dataset"], file["format_version"], file["dim_input"], file["dim_output"]
    end
    if expected_format_version !== nothing && format_version != expected_format_version
        throw(ArgumentError(
            "load_dataset: $path has format_version=$format_version, expected $expected_format_version"))
    end
    return dataset[:, 1:dim_input], dataset[:, dim_input+1:end]
end

# Positional train/test split (first n_training rows train, rest test) —
# same convention old_version's dataset_generation used (dataset_matrix[
# 1:n_training,...] / [(n_training+1):end,...]); safe here for the same
# reason it was safe there: rows come from weighted_sample's i.i.d. draws,
# not sorted by anything, so a positional split is as good as a shuffled one.
function train_test_split(inputs::AbstractMatrix, outputs::AbstractMatrix, n_training::Int)
    size(inputs, 1) == size(outputs, 1) || throw(ArgumentError(
        "train_test_split: inputs and outputs have different numbers of rows ($(size(inputs,1)) vs $(size(outputs,1)))"))
    0 < n_training < size(inputs, 1) || throw(ArgumentError(
        "train_test_split: n_training=$n_training must satisfy 0 < n_training < $(size(inputs,1)) (need at least one row left for testing)"))

    return inputs[1:n_training, :], outputs[1:n_training, :],
           inputs[n_training+1:end, :], outputs[n_training+1:end, :]
end

# --- Normalization ------------------------------------------------------------
#
# Ported unchanged from old_version/ML_QM_library.jl:442-532 (min-max only —
# old_version's separate standardization utilities are unused there too, so
# not ported, see the header note above).
normalization(x, x_max, x_min) = (x - x_min) / (x_max - x_min)
denormalization(x, x_max, x_min) = x * (x_max - x_min) + x_min

function max_min(data::AbstractMatrix)
    return [maximum(view(data, :, k)) for k in axes(data, 2)],
           [minimum(view(data, :, k)) for k in axes(data, 2)]
end

function normalize_data(data::AbstractMatrix, maxs, mins)
    result = similar(data, Float64)
    for k in axes(data, 2)
        result[:, k] = normalization.(view(data, :, k), maxs[k], mins[k])
    end
    return result
end

normalize_data(data::AbstractVector, maxs, mins) =
    [normalization(data[k], maxs[k], mins[k]) for k in eachindex(data)]

denormalize_data(data::AbstractVector, maxs, mins) =
    [denormalization(data[k], maxs[k], mins[k]) for k in eachindex(data)]

function denormalize_data(data::AbstractMatrix, maxs, mins)
    result = similar(data, Float64)
    for k in axes(data, 2)
        result[:, k] = denormalization.(view(data, :, k), maxs[k], mins[k])
    end
    return result
end

# Fits min-max stats on the training set only, applies to both train/test —
# old_version's own choice (its fit-on-everything alternative is present
# but commented out, ML_QM_library.jl:516-520). Unlike old_version's
# version, also returns the fitted stats: old_version's own caller
# (execute_problem_NN, ML_QM_library.jl:876-877) recomputed max_min(
# train_input)/max_min(train_output) right after calling this, from the
# same train_input/train_output already closed over inside it — a
# redundant second pass this port avoids by just returning what it already
# computed.
function train_test_dataset_normalization(train_input, train_output, test_input, test_output)
    maxs_input, mins_input = max_min(train_input)
    maxs_output, mins_output = max_min(train_output)

    normalized_train_input = normalize_data(train_input, maxs_input, mins_input)
    normalized_train_output = normalize_data(train_output, maxs_output, mins_output)
    normalized_test_input = normalize_data(test_input, maxs_input, mins_input)
    normalized_test_output = normalize_data(test_output, maxs_output, mins_output)

    return normalized_train_input, normalized_train_output,
           normalized_test_input, normalized_test_output,
           maxs_input, mins_input, maxs_output, mins_output
end

# --- Loss functions and dispatch ------------------------------------------------
#
# relative_mse/mape/loss_1 ported unchanged from old_version/ML_QM_library.jl:
# 652-665 (ŷ predicted, y expected) — except loss_1 is only meaningful on a
# single sample's length-≥2 output vector (it indexes ŷ[1]/ŷ[2] directly);
# it was never made batch-safe in old_version either, and isn't used
# anywhere live in this project, so this port doesn't try to generalize it.
#
# `train_model!`/`test_model` below take `loss` directly as a `Function`
# (default `Flux.Losses.mse`) rather than a `Symbol` looked up in a table —
# the same reasoning `predict_and_score`'s `simulate` argument already
# follows (DESIGN.md Part 6 (ii)/(iv)): a plain function value is more
# "black box" than a Dict{Symbol,Function} dispatch layer, and lets a
# caller pass any loss function, including ones NNQuantum.jl has never
# heard of, without registering it here first. LOSS_FUNCTIONS/resolve_loss
# are kept only as an optional convenience for callers that still want to
# name a loss by symbol (e.g. `:mse`, matching old_version's own usage) —
# `loss::Union{Function,Symbol}` accepts either. One correction made in
# porting: old_version's own dict has `:mae => Flux.Losses.mse`
# (ML_QM_library.jl:682,726) — both :mse and :mae point at mse there, so
# :mae was never actually exercised as MAE anywhere old_version used it.
# Fixed here to Flux.Losses.mae, since porting the bug forward silently
# would be worse than fixing a self-evidently mistyped mapping.
relative_mse(ŷ, y; ϵ=1e-12) = mean(((ŷ .- y) .^ 2) ./ (y .+ ϵ) .^ 2)
mape(ŷ, y; ϵ=1e-12) = mean(abs.((ŷ .- y) ./ (y .+ ϵ)))
loss_1(ŷ, y) = 0.5 * (abs(ŷ[1] - y[1])^0.5 + abs(ŷ[2] - y[2])^0.5)

const LOSS_FUNCTIONS = Dict{Symbol,Function}(
    :mse => Flux.Losses.mse,
    :mae => Flux.Losses.mae,
    :relative_mse => relative_mse,
    :mape => mape,
    :enhancing1 => loss_1,
)

resolve_loss(loss::Function) = loss
resolve_loss(loss::Symbol) = LOSS_FUNCTIONS[loss]

# --- Default model architecture -------------------------------------------------
#
# DESIGN.md Part 6 (iv): ported from Chu_DFL_execution.ipynb's :FL_1step_3p/
# :master_dynamic NN cell (id dfd162e7) — Dense(dim_input,hidden,relu) ->
# Dense(hidden,dim_output) -> softplus, cast to Float64 (`Flux.f64`, since
# this project's data is Float64 throughout, not Flux's Float32 default),
# `Flux.Adam(η)` via `Flux.setup`. The trailing `softplus` forces the raw
# (pre-denormalization) output positive — matches old_version's own choice;
# not revisited here.
#
# This is a function of dim_input/dim_output alone — it never sees a pulse
# parameter, a decom_basis, or anything else problem-specific — so it lives
# here, not in FockLadder_problem.jl: the same default architecture is
# reusable for any quantum problem's train_NN wrapper, whatever its own
# input/output dimensionality turns out to be.
function build_default_model(dim_input::Int, dim_output::Int; hidden::Int=500, η::Real=1e-4)
    model = Flux.f64(Flux.Chain(
        Flux.Dense(dim_input, hidden, Flux.relu),
        Flux.Dense(hidden, dim_output),
        Flux.softplus,
    ))
    opt_state = Flux.setup(Flux.Adam(η), model)
    return model, opt_state
end

# --- Training ------------------------------------------------------------------
#
# Originally ported from old_version/ML_QM_library.jl:672-702 (training!) as
# one Flux.withgradient+Flux.Optimisers.update! step per individual sample —
# a direct, unbatched port of old_version's own choice. Since revised: this
# is now genuine mini-batch gradient descent (shuffled each epoch, one
# batched matmul per `batch_size`-row chunk instead of one scalar
# forward/backward pass per row), a deliberate behavior change from the
# straight port, not just a refactor — flagged as such rather than done
# quietly. Two reasons: (1) at the real scale this project's own
# hyperparameters target (n_samples=800, epochs=350, per
# Chu_DFL_execution.ipynb), the per-sample loop was ~280,000 individual
# gradient steps per Fock-ladder step — Flux/Zygote are built around batched
# matrix ops, so this is a large, standard-practice speedup, not a
# micro-optimization. (2) old_version's per-sample loop also never shuffled
# row order between epochs (`zip(eachrow(train_X), eachrow(train_Y))`, same
# order every epoch) — a real training-quality gap independent of batching,
# fixed here as `perm = shuffle(1:n)` recomputed each epoch.
#
# `epoch_loss` is accumulated as a batch-size-weighted mean (`loss_value *
# length(batch_idx) / n`) so a shorter final batch doesn't skew the
# reported per-epoch loss. `batch_size` is clamped to the training-set size
# so a small smoke-test dataset degrades gracefully to full-batch rather
# than erroring. `opt_state` remains the caller's responsibility
# (`Flux.setup(optimiser, model)`), same as old_version's `opt` argument.
# `loss` accepts a `Function` directly or a `Symbol` resolved via
# `resolve_loss`/`LOSS_FUNCTIONS` (see that section's own note on why).
# old_version's @showprogress (ProgressMeter) stays dropped — see the (ii)
# status note in DESIGN.md for why. Renamed from `training!` to
# `train_model!` only to read next to `test_model` below without a name
# clash with Flux's own `Flux.train!`.
function train_model!(model, opt_state, epochs::Int, loss,
                       train_X::AbstractMatrix, train_Y::AbstractMatrix;
                       batch_size::Int=32)
    loss_fn = resolve_loss(loss)
    n = size(train_X, 1)
    bs = clamp(batch_size, 1, n)
    losses = Float64[]

    for epoch in 1:epochs
        perm = shuffle(1:n)
        epoch_loss = 0.0
        for batch_start in 1:bs:n
            batch_idx = perm[batch_start:min(batch_start + bs - 1, n)]
            Xb = permutedims(train_X[batch_idx, :])   # dim_input x batch
            Yb = permutedims(train_Y[batch_idx, :])   # dim_output x batch

            loss_value, grads = Flux.withgradient(model) do m
                loss_fn(m(Xb), Yb)
            end
            Flux.Optimisers.update!(opt_state, model, grads[1])
            epoch_loss += loss_value * length(batch_idx) / n
        end
        push!(losses, epoch_loss)
    end

    return losses
end

# --- Testing ---------------------------------------------------------------------
#
# Originally ported from old_version/ML_QM_library.jl:712-746
# (testing_model_infidelity): average loss over the test set, plus the raw
# predictions matrix. Also revised to a single batched forward pass rather
# than a per-row loop, matching train_model!'s change above — for the loss
# functions actually used here (Flux.Losses.mse/mae, relative_mse, mape,
# all plain `mean(...)` over whatever shape they're given), a batched call
# is numerically identical to averaging the per-sample losses the old loop
# computed, just without the per-row overhead (loss_1/:enhancing1 is the
# one exception — see the loss-dispatch section's own note; it isn't
# batch-safe, and isn't used live anywhere in this project). Renamed to
# `test_model` (dropping "_infidelity") because that's actually all this
# function ever computed in old_version too — its own infidelity/
# re-simulation code was already commented out there (ML_QM_library.jl:
# 735-742), so the old name promised something the function didn't do. The
# real infidelity/re-simulation scoring is predict_and_score below, which —
# unlike that dead code — is live and does call qo_infidelity, just on one
# predicted point at a time, not swept over a whole test set.
function test_model(model, loss, test_X::AbstractMatrix, test_Y::AbstractMatrix)
    loss_fn = resolve_loss(loss)
    predictions_t = model(permutedims(test_X))   # dim_output x n_test
    total_error = loss_fn(predictions_t, permutedims(test_Y))
    return total_error, permutedims(predictions_t)
end

# --- End-to-end train+test orchestration ------------------------------------------
#
# DESIGN.md Part 6 (iv): bundles train_test_split/train_test_dataset_
# normalization/build_default_model/train_model!/test_model into the one
# call a problem-specific train_NN wrapper actually needs — this is the
# piece that makes such a wrapper "thin" per (i)'s intended split. X/Y are
# plain matrices (e.g. straight from load_dataset) — this function never
# names a pulse parameter or a specific quantum problem, so it's reusable
# unchanged for any other problem plugged into NNQuantum's dataset/train/
# predict shape, not just :FL_1step_3p.
#
# Returns a NamedTuple rather than a long positional tuple — a
# `predict_and_score` caller downstream needs the model plus all four
# normalization stats together as one unit, and a NamedTuple keeps that
# bundle self-documenting at the call site instead of four more positional
# arguments to keep in order.
#
# `model`/`opt_state`, if both given, are used as-is instead of calling
# `build_default_model` — closes the one architectural assumption this
# function otherwise bakes in unconditionally. Without this, a problem
# whose own input/output *dimensions* fit the generic dataset shape fine
# but whose own architecture needs (depth, width, activations) don't match
# `build_default_model`'s fixed choice would have had no way to use
# `train_and_test_NN` at all, only the lower-level pieces it bundles —
# defeating the "thin wrapper" point of having this function in the first
# place. `hidden`/`η` are simply ignored when a model/opt_state pair is
# supplied. Must be given together, not one without the other (an
# `opt_state` is only valid for the exact model it was built from).
#
# Also returns dim_input/dim_output/hidden/η/opt_state alongside the fields
# documented above — not needed by predict_and_score (which only reads
# model/the four normalization stats), but needed so the same bundle this
# function returns can be handed directly to save_nn (below) regardless of
# whether it came from a fresh train_and_test_NN call or from load_nn
# (which returns the same field set): save_nn needs dim_input/dim_output/
# hidden to rebuild an identical model skeleton on reload, and a caller
# chaining train_and_test_NN calls across a session (not through disk) can
# reuse opt_state directly.
function train_and_test_NN(X::AbstractMatrix, Y::AbstractMatrix, n_training::Int;
                            hidden::Int=500, η::Real=1e-4, epochs::Int=350,
                            loss::Union{Function,Symbol}=:mse, batch_size::Int=32,
                            model=nothing, opt_state=nothing)
    (model === nothing) == (opt_state === nothing) || throw(ArgumentError(
        "train_and_test_NN: model and opt_state must be provided together, or not at all"))

    train_X, train_Y, test_X, test_Y = train_test_split(X, Y, n_training)
    norm_train_X, norm_train_Y, norm_test_X, norm_test_Y,
        maxs_input, mins_input, maxs_output, mins_output =
        train_test_dataset_normalization(train_X, train_Y, test_X, test_Y)

    dim_input, dim_output = size(X, 2), size(Y, 2)
    if model === nothing
        model, opt_state = build_default_model(dim_input, dim_output; hidden=hidden, η=η)
    end

    losses = train_model!(model, opt_state, epochs, loss, norm_train_X, norm_train_Y; batch_size=batch_size)
    test_error, test_predictions = test_model(model, loss, norm_test_X, norm_test_Y)

    return (model=model, opt_state=opt_state, losses=losses, test_error=test_error, test_predictions=test_predictions,
            maxs_input=maxs_input, mins_input=mins_input, maxs_output=maxs_output, mins_output=mins_output,
            dim_input=dim_input, dim_output=dim_output, hidden=hidden, η=η)
end

# --- Prediction orchestration ----------------------------------------------------
#
# predict_output replaces old_version's prediction_normalize_data/
# prediction_denormalize_data/identity_func trio (ML_QM_library.jl:753-765,
# a :normalized/:unnormalized Dict{Symbol,Function} toggle) with the one
# path it's actually used for in a normalized pipeline — normalize the
# input, run the model, denormalize the output. There is no unnormalized
# path to preserve: old_version's own prediction_infidelity (its only
# caller) is always invoked from execute_problem_NN after normalization has
# already happened (dataset_features.norm_dataset==:normalized is what
# populates NN_features.features_prediction's maxs/mins in the first
# place), so :unnormalized was dead there too.
function predict_output(model, x::AbstractVector, maxs_input, mins_input, maxs_output, mins_output)
    x_norm = normalize_data(x, maxs_input, mins_input)
    y_norm = model(x_norm)
    return denormalize_data(y_norm, maxs_output, mins_output)
end

# Ported from old_version's prediction_infidelity (ML_QM_library.jl:768-795),
# rebased per DESIGN.md Part 6 (ii): where that function got its dynamics
# function via plot_problem_dictionary[dataset_features.problem] and called
# it through trajectory_plot's own Dict{Symbol,Function} indirection, this
# takes the dynamics call directly as `simulate` — a plain function value,
# `predicted_output::AbstractVector -> final_state`, that the caller builds
# (a closure over FLstep_dynamics_3p, t0, and the current initial_state, in
# FockLadder_problem.jl's :FL_1step_3p wrapper predict_drive_parameters,
# DESIGN.md (iv)). Scores the resulting final_state against target_state with
# qo_infidelity (already in this file). This is why predict_and_score can
# stay problem-agnostic even though its whole purpose is to drive a
# problem-specific simulation: the problem-specific part is pushed into the
# argument, not hardcoded here.
function predict_and_score(model, x, maxs_input, mins_input, maxs_output, mins_output,
                            simulate, target_state)
    predicted_output = predict_output(model, x, maxs_input, mins_input, maxs_output, mins_output)
    final_state = simulate(predicted_output)
    infidelity = qo_infidelity(final_state, target_state)
    return predicted_output, final_state, infidelity
end

# --- NN persistence (JLD2) -------------------------------------------------------
#
# Lets a trained NN (train_and_test_NN's own return bundle) be reloaded in a
# later run — for run_Fock_ladder's :continue_training / :fixed NN modes,
# added when generalizing it to accept dataset/NN reuse rather than always
# generating+training fresh each step. Follows the same "store the
# reconstructible spec, not the raw derived object" convention save_dataset
# and QuantumDynamics/framework's own io.jl already use: dim_input/
# dim_output/hidden (to rebuild an architecturally-identical Chain skeleton
# via build_default_model on load) plus Flux.state(model) — Flux's own
# documented portable model-serialization format, robust to internal
# Flux/Zygote struct changes across versions the way saving the raw model
# struct directly wouldn't be — and the four normalization stats a
# prediction made from this NN must be denormalized with (they have to
# travel with the model: a caller reloading a :fixed NN has no dataset of
# its own to refit them from, and reusing the wrong stats would silently
# corrupt every prediction, not error).
#
# Deliberately NOT persisted: optimizer state. Flux.state/loadmodel! is
# documented and portable for models; there's no equivalent documented
# convention for round-tripping an Optimisers.jl state tree through disk,
# and faithfully doing so (matching Adam's momentum/variance estimates
# exactly) isn't needed for what :continue_training actually asks for —
# resuming training from the saved weights. load_nn below builds a fresh
# opt_state via Flux.setup after loading the model, the same way starting
# any fine-tuning run from pretrained weights ordinarily does.
const NN_FORMAT_VERSION = 1

"""
    save_nn(path, nn; description="", params=Dict{Symbol,Any}())

Save a trained-NN bundle (`train_and_test_NN`'s own return value, or a
bundle previously returned by `load_nn`) to `path` (a `.jld2` file).
"""
function save_nn(path::AbstractString, nn; description::AbstractString="",
                  params::Dict{Symbol,Any}=Dict{Symbol,Any}())
    jldsave(path;
        format_version=NN_FORMAT_VERSION,
        dim_input=nn.dim_input, dim_output=nn.dim_output,
        hidden=nn.hidden, η=nn.η,
        model_state=Flux.state(nn.model),
        maxs_input=nn.maxs_input, mins_input=nn.mins_input,
        maxs_output=nn.maxs_output, mins_output=nn.mins_output,
        description=description, params=params,
    )
    nothing
end

"""
    load_nn(path; expected_format_version=nothing)

Reload a bundle saved by `save_nn`, shaped identically to
`train_and_test_NN`'s own return value (`model`, `opt_state`, the four
normalization stats, `dim_input`/`dim_output`/`hidden`/`η`) so it can be used
directly as `predict_and_score`'s/`predict_drive_parameters`'s `nn` argument
(no further training needed — `:fixed` mode), or passed straight through as
`train_and_test_NN`'s `model=`/`opt_state=` keywords to resume training
(`:continue_training` mode). `opt_state` is freshly built (`Flux.setup`), not
restored — see this section's header note for why. `expected_format_version`,
if given, is checked the same way `load_dataset`'s own check works.
"""
function load_nn(path::AbstractString; expected_format_version::Union{Nothing,Integer}=nothing)
    dim_input, dim_output, hidden, η, model_state,
        maxs_input, mins_input, maxs_output, mins_output, format_version =
        jldopen(path, "r") do file
            file["dim_input"], file["dim_output"], file["hidden"], file["η"], file["model_state"],
            file["maxs_input"], file["mins_input"], file["maxs_output"], file["mins_output"],
            file["format_version"]
        end
    if expected_format_version !== nothing && format_version != expected_format_version
        throw(ArgumentError(
            "load_nn: $path has format_version=$format_version, expected $expected_format_version"))
    end

    model, opt_state = build_default_model(dim_input, dim_output; hidden=hidden, η=η)
    Flux.loadmodel!(model, model_state)

    return (model=model, opt_state=opt_state,
            maxs_input=maxs_input, mins_input=mins_input,
            maxs_output=maxs_output, mins_output=mins_output,
            dim_input=dim_input, dim_output=dim_output, hidden=hidden, η=η)
end
