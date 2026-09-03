# Fock-ladder validation log (step (v))

## Step 1 validation — 2026-09-03T14:09:15.929

- basis_id = 1, save_dir = C:\Users\dadda\OneDrive\Desktop\GItHub-Aug26\NN_QM\NNQuantum
- n_samples = 150, epochs = 60, hidden = 128, η = 1e-3, batch_size = 32, train_fraction = 0.85, uniform sampling
- dataset: `dataset_step1_b1.jld2`
- NN: `nn_step1_b1.jld2` (test_error = 0.04693843438578395)
- predicted (τ_exc, ωd, τ_SWAP) = [0.004428692669005297, -15790.780547465603, 0.005965250647802716]
- prediction infidelity = 0.04385537646450843
- trajectory plot: `trajectory_step1_b1.png`
- pipeline check: OK (well-formed trajectory, trace preserved to 4.440892098500626e-16, all files saved)
- visual validity: **confirmed by user (2026-09-03)** — trajectory plot shows the expected qubit→oscillator population swap across the spin-flip/SWAP boundary

## Step 1→2 validation — 2026-09-03T14:28:40.236

- basis_id = 1, save_dir = C:\Users\dadda\OneDrive\Desktop\GItHub-Aug26\NN_QM\NNQuantum
- step 1 reused via nn_mode=:fixed (loads nn_step1_b1.jld2, no retraining — reproduces the already-reviewed step 1 prediction exactly)
- step 2: n_samples = 150, epochs = 60, hidden = 128, η = 1e-3, batch_size = 32, train_fraction = 0.85, uniform sampling
- step 1: predicted (τ_exc, ωd, τ_SWAP) = [0.004428692669005297, -15790.780547465603, 0.005965250647802716], infidelity = 0.04385537646450843
- step 2: predicted (τ_exc, ωd, τ_SWAP) = [0.0049987587800279796, -15823.001004406942, 0.004257346157488247], infidelity = 0.08065486849325132, NN test_error = 0.04288939783929979
- per-step plots: `trajectory_step1_b1.png`, `trajectory_step2_b1.png`
- cumulative plot: `trajectory_full_b1.png`
- pipeline check: OK (well-formed trajectories, trace preserved, all files saved)
- visual validity: **confirmed by user (2026-09-03)** — cumulative plot shows a clean two-rung climb (⟨n_osc⟩ steps 0→~0.9→~1.8), each rung driven by the expected qubit population rise-then-fall

## Overall verdict

Both the single-step (N_steps=1) and two-step, ladder-chaining (N_steps=2, step 1 reused, step 2 fresh) runs of the train/predict/reiterate algorithm completed mechanically correctly and were visually confirmed by the user as physically sensible. This is the staged validation called for by (v) in `CLAUDE.md`'s Plan (step 3) — the 0-4 loop, including reiterating from a *predicted* (imperfect) state rather than the target state, works as designed at this small/fast hyperparameter scale.

