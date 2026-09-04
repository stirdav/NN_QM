# NNQuantum.jl — a guide for users

This file explains what `NNQuantum.jl` is, how it is organized, and what every function in it does. It is written for someone who wants to *use* `NNQuantum.jl` — either to understand the current Fock-ladder project, or to plug a *different* quantum problem into the same machinery.

If you want the history of *why* things were built this way, or the story of bugs found along the way, read `DESIGN.md` instead. This file only explains what things do today.

## What `NNQuantum.jl` is

`NNQuantum.jl` is the neural-network (NN) library of this project. It holds every function that trains a model, tests it, saves it, or uses it to predict something — plus a few small math helpers those functions need.

The key idea: **`NNQuantum.jl` knows nothing about quantum physics.** It never mentions a pulse, a drive frequency, a qubit, or any other detail of the Fock-ladder problem. It only works with plain things: numbers, matrices, a trained model, and a "basis" of operators. This is on purpose — it means the exact same file can be reused for a completely different quantum problem later, without changing a single line in it.

The actual quantum-physics problem (the Fock ladder) lives in `FockLadder_problem.jl` and `FockLadder_execution.jl`. Those two files are the ones that know about qubits, pulses, and the specific physics. They call into `NNQuantum.jl` to do the "learning" part.

## How the pieces connect

Think of it as three layers:

- **`NNQuantum.jl` (this file)** — generic. Give it plain numbers in, get plain numbers out. Knows nothing about qubits or pulses. Also where dataset/basis/model files get saved and loaded (`save_dataset`, `save_operator_basis`, `save_nn`, ...) — the file-management side of things lives here too, not just training itself.
- **`FockLadder_problem.jl`** — the physics side: the qubit/oscillator system, its Hamiltonian, its dissipators, the raw functions that simulate a trajectory (`FLstep_dynamics_3p` and friends) — *and* this problem's own dataset-generation functions (`FL_1step_3p_NN_outputs`/`FL_1step_3p_NN_inputs`) and NN-facing glue (`predict_drive_parameters`), since building a dataset row or a prediction target is itself a "physics meets the NN's expected shape" job specific to this problem, not something `NNQuantum.jl` could do generically.
- **`FockLadder_execution.jl`** — thin orchestration only: the `run_Fock_ladder`/`generate_decom_basis` loop that calls into `FockLadder_problem.jl` for settings/dynamics/dataset-creation and into `NNQuantum.jl` for training/saving/loading. It holds no dataset-creation or NN-wrapper logic of its own.

The one function you'd actually need to write yourself for a new quantum problem is `predict_drive_parameters` (in your own `*_problem.jl`) — it builds the one input `predict_and_score` needs (a feature vector for your target state) and a `simulate` closure over your own dynamics function. `train_NN` itself, despite living in `NNQuantum.jl`, needs no rewriting per problem — it turns any problem's own dataset shape (`Vector{Vector}`/`Vector{<:Tuple}`, or plain matrices) into `X`/`Y` and calls `train_and_test_NN`; nothing about it is Fock-ladder-specific.

## What the file depends on

At the top of the file:

```julia
using QuantumOptics   # Ket/Operator types, and fidelity/expect — used for comparing quantum states
using LinearAlgebra   # qr — used to build a random operator basis
using Statistics      # mean — used in a couple of loss functions
using Random          # shuffle — used to mix up the training data each epoch
using JLD2            # saving/loading files
using Flux            # the actual neural-network library
using CairoMakie      # plotting a trajectory
```

## Overview: every function, in one line each

| Function | What it does |
|---|---|
| `qo_infidelity(a, b)` | How different two quantum states are (0 = identical, 1 = as different as possible). |
| `rand_hermitian_orthonormal_basis(d, basis)` | Makes a random set of operators, used to turn a quantum state into a list of plain numbers. |
| `save_operator_basis` / `load_operator_basis` | Save/reload that operator set to/from a file. |
| `generate_decom_basis(cs, basis_id)` | Make a new decomposition basis for a composite system `cs` and save it, once, on purpose. |
| `threeD_parameter_space(p, ranges, sizes)` | Builds a grid of points over 3 ranges, weighted by a probability function `p`. |
| `weighted_sample(items, weights, n)` | Draws `n` items from a list, favoring higher-weighted ones — typically used together with `threeD_parameter_space`'s output. |
| `uniform_parameter_sample(ranges, n)` | Draws `n` random points directly from 3 ranges — quicker than the grid above, but only correct when every point should be equally likely. |
| `dataset_rows` / `save_dataset` / `load_dataset` | Turn an inputs/outputs pair into one-row-per-sample form, save it to a file, and load it back. |
| `train_test_split` | Split a dataset into a training part and a testing part. |
| `normalize_data` / `denormalize_data` / `max_min` / `train_test_dataset_normalization` | Rescale numbers into a `0`-to-`1`-ish range before training (and back again after). |
| `relative_mse`, `mape`, `loss_1`, `LOSS_FUNCTIONS`, `resolve_loss` | A few ways to measure how wrong a prediction is, and a way to pick one by name. |
| `build_default_model(dim_input, dim_output)` | Builds a fresh, untrained neural network of the right input/output size. |
| `train_model!` | Trains a model on data, for a number of epochs. |
| `test_model` | Measures how well a trained model does on data it wasn't trained on. |
| `train_and_test_NN` | Does the whole thing in one call: split the data, normalize it, build (or reuse) a model, train it, test it. |
| `train_NN` | A convenience wrapper around `train_and_test_NN` that also accepts a problem's own raw dataset shape (`Vector{Vector}`/`Vector{<:Tuple}`), not just plain matrices. |
| `predict_output` | Feed one input into a trained model and get a real-world (denormalized) answer back. |
| `predict_and_score` | Predict an answer, use it to run a simulation, and score how close the result got to a target. |
| `save_nn` / `load_nn` | Save a trained model (and everything needed to use it again) to a file, and load it back later. |
| `save_prediction` / `load_prediction` | Save a predicted output (and the infidelity it scored) to a file, and load it back later. |
| `plot_trajectory(tspan, states, observables)` | Plot the expectation value of one or more chosen observables, over time, along a trajectory. |

The rest of this guide goes through each of these in more detail, in the same order they appear in the file.

---

## Comparing quantum states

### `qo_infidelity(a, b)`

Tells you how different two quantum states `a` and `b` are.

- Returns `0.0` if the two states are identical.
- Returns something closer to `1.0` the more different they are.
- `a` and `b` can each be a `Ket` (a plain quantum state) or an `Operator` (a density matrix, used for a state affected by noise/loss). Any mix of the two is fine — the function converts a `Ket` to the matching `Operator` form automatically before comparing.

This is the standard way, in this project, to answer "how close did we get to the state we wanted?"

---

## Building and saving an operator basis

A "decomposition basis" is a set of operators used as a fixed, repeatable way to turn one quantum state into a list of plain numbers (one number per operator, using each operator's expectation value on that state). The neural network only ever sees these plain numbers — never the quantum state itself.

### `rand_hermitian_orthonormal_basis(d, basis)`

Makes such a set, for a quantum system of dimension `d`, expressed in the given `basis`.

- Returns a list of `d^2` operators.
- **Important**: every call makes a genuinely different, random set — nothing is fixed between calls. That is fine as long as you make the basis once and reuse the *same* one for everything that depends on it (generating data, training, and predicting). Mixing two different bases — say, training on one and predicting with another — silently gives meaningless results, since the same list of numbers would mean two different things.

Because of that risk, this project never calls this function directly to start a run — instead, it goes through `generate_decom_basis` (below), which makes a basis **once**, on purpose, and saves it under a chosen number. `run_Fock_ladder`/`run_FockTarget_Bspline` then always load a saved basis, rather than making a fresh one each time.

### `save_operator_basis(path, ops; description="", params=Dict())` / `load_operator_basis(path, basis; expected_format_version=nothing)`

Save a list of operators (e.g., `rand_hermitian_orthonormal_basis`'s own output) to a `.jld2` file, and load it back later.

- `save_operator_basis` stores the plain numbers behind each operator, not the operator object itself — this makes the file safe to reload even if the underlying quantum library changes internally later.
- `load_operator_basis` needs you to supply the `basis` to rebuild the operators against (your own system's basis, e.g. `cs.basis`) — the file itself does not carry this, since you already have it on hand.
- `expected_format_version` is optional. If you pass a number, the function checks the saved file was written in that same format and stops with a clear error if not, rather than silently reading something that doesn't match.

### `generate_decom_basis(cs, basis_id; save_dir=".", overwrite=false, prefix="decom_basis")`

Makes a new decomposition basis for the composite system `cs` and saves it — the one place `rand_hermitian_orthonormal_basis` actually gets called to start a real run. Call it once, by hand, whenever you want a new basis; reuse the same `basis_id` afterward (even in a later Julia session) until you decide to make a different one.

- `cs` only needs a `.basis` field (e.g. `FockLadder_problem.jl`'s `cs`, or `Bspline_Fock_problem.jl`'s `cs_res`) — this function never assumes which problem it's for.
- Saved to `save_dir/<prefix>_b<basis_id>.jld2`. `prefix` (default `"decom_basis"`) is what keeps two different problems' basis files from colliding in a shared `save_dir` — e.g. this project's own `:FL_1step_Bspline` uses `prefix="decom_basis_bspline"`, since its composite dimension differs from `:FL_1step_3p`'s. Mixing up which basis a dataset/NN was built against silently gives meaningless results (same risk `rand_hermitian_orthonormal_basis`'s own note above describes) — distinct file names are what prevent that.
- Refuses to overwrite an existing file for the same `basis_id`/`prefix` unless `overwrite=true` — a dataset or NN may already be saved against it.

---

## Choosing sample points over a range of parameters

These two functions answer the same question — "give me some sample points inside these three ranges" — in two different ways.

### `threeD_parameter_space(p, parameters_range, dim_parameters_space)`

Builds a full grid of points across three ranges (the third one spaced logarithmically, i.e. more finely near the small end), then computes how likely each grid point is, using your own function `p`.

- `parameters_range` is `[[min1,max1], [min2,max2], [min3,max3]]`.
- `dim_parameters_space` is `[n1, n2, n3]` — how many grid points along each range.
- Returns `(points, probabilities)` — the full list of grid points, and a matching list of how likely each one is.
- This can be slow and use a lot of memory if `n1*n2*n3` is large, since it builds every point up front. Use this only when `p` really does treat some points as more likely than others.

### `weighted_sample(items, weights, n)`

Draws `n` items from a list (`items`), where an item with a bigger `weights` entry is more likely to be drawn — with replacement, so the same item can come up more than once. This is the usual partner for `threeD_parameter_space`'s output: `points, probabilities = threeD_parameter_space(...)`, then `weighted_sample(points, probabilities, n_samples)` turns that whole weighted grid into `n_samples` actual sample points.

### `uniform_parameter_sample(parameters_range, n_samples)`

Directly draws `n_samples` random points from the same three ranges (again, log-spaced in the third one), without building a grid first.

- Much cheaper than the grid-based approach above.
- Only correct when every point in the range should be equally likely to be picked — if you need some points to be more likely than others, use `threeD_parameter_space` instead.

---

## Working with a saved dataset

### `dataset_rows(inputs, outputs)`

Turns an `inputs`/`outputs` pair (e.g. `FockLadder_execution.jl`'s `FL_1step_3p_NN_inputs`/`FL_1step_3p_NN_outputs`) into a list of plain rows, one per sample — each row is just `inputs[i]` followed by `outputs[i]`, joined together. Mostly an internal helper for `save_dataset` below, but usable on its own if you want the row form without saving it.

### `save_dataset(path, inputs, outputs; description="", params=Dict())`

Saves an `inputs`/`outputs` pair to a `.jld2` file, one row per sample (via `dataset_rows`). Also records how many columns are input vs. output, so `load_dataset` can split them back apart correctly.

### `load_dataset(path; expected_format_version=nothing)`

Loads a dataset that was saved to a `.jld2` file by `save_dataset` above.

- Returns `(X, Y)` — two plain matrices: `X` is the input columns, `Y` is the output columns.
- `expected_format_version` works the same way as in `load_operator_basis` above.

### `train_test_split(inputs, outputs, n_training)`

Splits a dataset into a training part and a testing part.

- `inputs`/`outputs` must be matrices with the same number of rows (one row per sample).
- `n_training` says how many rows (counted from the top) go into the training part; everything else becomes the testing part.
- Returns four matrices: `train_inputs, train_outputs, test_inputs, test_outputs`.
- Stops with a clear error if `n_training` is `0`, negative, or leaves nothing for testing.

---

## Normalizing data

Before training, every input and output column gets rescaled so its smallest value maps to `0` and its largest maps to `1`. This helps the neural network learn — without it, columns with very different scales (say, one column of tiny numbers and another of huge numbers) can make training unstable.

- `normalization(x, x_max, x_min)` / `denormalization(x, x_max, x_min)` — rescale one number, and undo it.
- `max_min(data)` — find the smallest and largest value in every column of a matrix.
- `normalize_data(data, maxs, mins)` / `denormalize_data(data, maxs, mins)` — apply that rescaling (or undo it) to a whole matrix or a single row (vector).
- `train_test_dataset_normalization(train_input, train_output, test_input, test_output)` — the one you'll actually call: works out the min/max scale from the **training data only**, and applies it to both the training and the testing data. Returns the four rescaled matrices, plus the four min/max stat lists used — you need to keep those stats, since any later prediction has to be rescaled the exact same way.

---

## Measuring how wrong a prediction is (loss functions)

- `relative_mse(ŷ, y)` — average squared error, scaled by how big `y` is.
- `mape(ŷ, y)` — average absolute percentage error.
- `loss_1(ŷ, y)` — a specific, less general error measure (only meaningful for a single sample with at least 2 output numbers).
- `LOSS_FUNCTIONS` — a lookup table from a short name (like `:mse`) to the matching function. Includes `:mse` and `:mae` (Flux's own mean-squared/mean-absolute error), plus `:relative_mse`, `:mape`, `:enhancing1` (the three above).
- `resolve_loss(loss)` — turns either a function or one of those short names into an actual function to call. Anywhere in this file that takes a `loss` argument accepts either form — a plain Julia function, or a name like `:mse`.

---

## Building, training, and testing a model

### `build_default_model(dim_input, dim_output; hidden=500, η=1e-4)`

Builds a brand-new, untrained neural network with a given input size and output size.

- `hidden` is how many neurons are in its one hidden layer.
- `η` (eta) is the learning rate — how big a step the training takes each update.
- Returns `(model, opt_state)` — the network itself, and its optimizer's starting state (needed by `train_model!` below).

### `train_model!(model, opt_state, epochs, loss, train_X, train_Y; batch_size=32)`

Trains `model` on the given data, for `epochs` full passes over it.

- Data is split into small chunks ("batches") of `batch_size` rows, and shuffled into a new random order every epoch.
- `loss` can be a function or a short name (see the loss-function section above).
- Returns a list of numbers — the loss after each epoch, so you can check training is actually improving (going down).
- This function changes `model` and `opt_state` in place (hence the `!` at the end of its name, a Julia convention for "this function modifies its argument").

### `test_model(model, loss, test_X, test_Y)`

Checks how well an already-trained `model` performs on data it did *not* train on.

- Returns `(total_error, predictions)` — one overall error number, and the model's raw prediction for every test row.

### `train_and_test_NN(X, Y, n_training; hidden=500, η=1e-4, epochs=350, loss=:mse, batch_size=32, model=nothing, opt_state=nothing)`

This is the one you'll normally call — it does the whole job in one step: split the data, normalize it, build a model (or use one you already have), train it, and test it.

- `X`, `Y` — your full dataset, as plain matrices.
- `n_training` — how many rows go to training (the rest go to testing).
- `model`/`opt_state` — optional. Leave both as `nothing` to get a brand-new model from `build_default_model`. Or pass in both together (never just one) to keep training a model you already have — this is how "keep training an existing NN" works.
- Returns one bundle (a `NamedTuple`) holding everything you might need next: `model`, `opt_state`, `losses`, `test_error`, `test_predictions`, the four normalization stats (`maxs_input`, `mins_input`, `maxs_output`, `mins_output`), and `dim_input`, `dim_output`, `hidden`, `η`. Keep this whole bundle together — `predict_and_score` and `save_nn` (below) both expect it as one piece, not its parts separately.

### `train_NN(X, Y; train_fraction=0.9375, kwargs...)` / `train_NN(inputs, outputs; kwargs...)`

A small convenience layer on top of `train_and_test_NN`, for when you'd rather give a fraction than an exact `n_training` row count, or your dataset isn't in plain-matrix form yet.

- The `X`/`Y` (matrix) method just works out `n_training` from `train_fraction` and calls `train_and_test_NN` — `train_fraction` is clamped so there's always at least one row on each side of the split, even for a tiny smoke-test dataset.
- The `inputs`/`outputs` method accepts a problem's own raw dataset shape directly — `inputs` a list of feature vectors, `outputs` a list of tuples or vectors (whichever your dataset-generation code produced) — converts both to plain matrices, then calls the method above.
- Despite living in this file, `train_NN` never looks at what problem produced its data — any quantum problem's own dataset-generation code can call it unchanged.

---

## Making a prediction

### `predict_output(model, x, maxs_input, mins_input, maxs_output, mins_output)`

Feeds one input `x` (a plain vector) into `model` and returns a real-world answer — it rescales `x` down before feeding it in, and rescales the model's raw answer back up afterward, using the normalization stats you give it (the same ones `train_and_test_NN` returned).

### `predict_and_score(model, x, maxs_input, mins_input, maxs_output, mins_output, simulate, target_state)`

Does the same prediction as above, then goes one step further: runs your own `simulate` function on the predicted answer, and compares the result against `target_state` using `qo_infidelity`.

- `simulate` is a function you supply, that takes the predicted answer and returns a quantum state — for this project, that means actually running the physics simulation with the predicted pulse.
- Returns `(predicted_output, final_state, infidelity)` — the raw prediction, the state the simulation actually reached, and how far that state is from the target.
- This function never mentions the physics simulation by name — you hand it a `simulate` function, and it just calls whatever you gave it. That is what keeps this file usable for any quantum problem, not only this project's own.

---

## Saving and loading a trained model

### `save_nn(path, nn; description="", params=Dict())`

Saves a trained-model bundle (from `train_and_test_NN`, or one you got earlier from `load_nn`) to a `.jld2` file, so you can use it again in a later Julia session.

- Saves the model's learned weights, its shape (`dim_input`/`dim_output`/`hidden`), and the four normalization stats.
- Does **not** save the optimizer's internal state — reloading and continuing training just starts a fresh optimizer on the saved weights, which is the normal way fine-tuning works.

### `load_nn(path; expected_format_version=nothing)`

Loads a bundle saved by `save_nn`, ready to use right away.

- Returns a bundle shaped exactly like `train_and_test_NN`'s own return value, so you can either:
  - use it directly for prediction (with `predict_and_score`), needing no further training, or
  - pass its `model`/`opt_state` straight into `train_and_test_NN`'s `model=`/`opt_state=` to keep training it.

---

## Saving and loading a prediction

### `save_prediction(path, predicted_output, infidelity; description="", params=Dict())`

Saves a predicted output (e.g. what `predict_and_score`/`predict_output` returned) together with the infidelity it scored, to a `.jld2` file — so you have a record of what was predicted, without having to keep it around in memory or re-derive it later.

### `load_prediction(path; expected_format_version=nothing)`

Loads a prediction saved by `save_prediction`. Returns `(predicted_output, infidelity)`.

---

## Plotting a trajectory

### `plot_trajectory(tspan, states, observables; title="", xlabel="t", ylabel="⟨·⟩", vlines=[], save_path=nothing)`

Plots the expectation value of one or more observables you choose, as a function of time, along a trajectory.

- `tspan`/`states` are a trajectory — a list of time points and the matching list of quantum states at those times. `NNQuantum.jl` doesn't produce these itself (it has no simulation code); in this project they come from `FockLadder_problem.jl`'s `FLstep_dynamics_3p_dense` (a densely-sampled version of the two-stage pulse protocol — the plain `FLstep_dynamics_3p` only returns 3 points, too coarse to plot a curve from).
- `observables` is a list of `label => operator` pairs, e.g. `["n_qubit" => op(cs,:qubit,:n), "n_osc" => op(cs,:osc,:n)]` — one line is plotted per pair, in the order given, each labeled in the legend by its `label`.
- `vlines` optionally draws dashed vertical reference lines at given time values (e.g. where one pulse stage ends and the next begins).
- `save_path`, if given, saves the plot to a file (e.g. `"trajectory.png"`); the figure is returned either way, so you can also just look at it without saving.
- Returns `(fig, values)`: the plot itself, and a dictionary from each observable's label to its plain list of expectation values along `tspan` — so you can also inspect or reuse the numbers directly, without re-plotting.

Example, using a dense trajectory from `FockLadder_problem.jl`:

```julia
tspan, states = FLstep_dynamics_3p_dense(t0, initial_state, τ_exc, ωd, τ_SWAP)

fig, values = plot_trajectory(tspan, states,
    ["n_qubit" => op(cs, :qubit, :n), "n_osc" => op(cs, :osc, :n)];
    title="predicted trajectory", ylabel="⟨n⟩", vlines=[t0 + τ_exc],
    save_path="predicted_trajectory.png")
```

---

## A short, complete example

This shows the pieces above working together, outside of any specific physics problem — imagine `X`/`Y` came from wherever your own dataset is generated.

```julia
using JLD2
include("NNQuantum.jl")

# X: inputs, one row per sample. Y: matching outputs, one row per sample.
n_training = round(Int, 0.9 * size(X, 1))

nn = train_and_test_NN(X, Y, n_training; epochs=200, loss=:mse)
println("test error: ", nn.test_error)

# Save it, so a later session can use it without retraining:
save_nn("my_model.jld2", nn)

# ... in a later session:
nn = load_nn("my_model.jld2")

# Predict from one new input row:
prediction = predict_output(nn.model, x_new, nn.maxs_input, nn.mins_input, nn.maxs_output, nn.mins_output)
```

## If you want to plug in a different quantum problem

You would not need to change anything in `NNQuantum.jl` — and you can call `train_NN` directly, unchanged, since it's already generic (it only cares about your dataset's shape, not what problem produced it). The one thing you'd write yourself, in your own `*_problem.jl`:

- **A `predict_drive_parameters`-like function**, mirroring `FockLadder_problem.jl`'s own — build the one input `predict_and_score` needs (a feature vector for your target state), and a `simulate` function that runs *your* physics simulation, then call `predict_and_score`.

Everything else — training, testing, normalizing, saving, loading — is already handled, unchanged, by `NNQuantum.jl`.
