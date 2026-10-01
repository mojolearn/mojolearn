# AMD split-plan output cap 1 M cells (PR #52, lane/neural-pass45) + the AMD neural toggle sweep, 2026-10-01

MI325X, released 0.8.32 + the branch's Python and neural bindings (gfx942). lm-forward digest 4a8e781b0739a038 and the
lm-train-step losses identical in every run; gemm_device_check all green (8 gates) at 2 M and at the shipped default.

Sweep (MOJOLEARN_GEMM_SPLIT_CELLS): lm-train-step 121.6 (128 K) / 105.0 (512 K) / 100.9 (1 M) / 101.0 (2 M) ms;
lm-forward 57.9 / 59.3 / 54.2 / 54.5 ms; the MLP projections 1.60 -> 1.18 ms a layer. AMD default set to 1 M.
Confirm with the env unset: lm-train-step 101.0 / 101.0 ms, lm-forward 56.1 / 55.3 ms; control at 128 K: 122.1 / 56.6.

Toggle sweep on AMD (neural_experiments --set nvidia, --calls 10, main before #52): every toggle within 0.96-1.12x
of baseline with identical bits; no_retain_weights is 5-12% slower (keep retention on); nothing worth flipping.
GEMM arms (tuned128/half/quarter/kpack/kfoldv) on transformer-forward and lm-forward: all within 2%.
Attention phase timers (lm-train-step, per layer): bwd_dq_tiled_pf 1.70 ms, fwd_r2_keep_kernel 1.27, core 1.42,
qkv_proj 1.20, bwd_kvgrid_dkdv_pf 0.91, bwd_zdot_estash_dres_pf 0.62, bwd_corner_flag 0.47, o_proj 0.40.
Raw: R2 measurements/2026-10-01/{pr52-amd,pr52b-amd,amd-toggle-sweep}.tar.gz.
