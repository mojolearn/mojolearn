# Public CPU inference for GaussianProcessRegressor, GaussianProcessClassifier and GaussianMixture.sample (2026-09-15)

Branch `lane/inference-neighbors-density`, after merging main's GP `normalize_y` (d5decf418),
`GaussianMixture.sample` (be2a2da5c) and `GaussianProcessClassifier` (3570940e8). Nine lanes serve CPU
inference from a model saved on a GPU: gp, gp-matern12, gp-matern32, gp-matern52-ard, gp-normalize-y,
gpc, gpc-multiclass, gmm-sample and gmm-random-init-sample.

What changed:

- `GaussianProcessRegressor.save` and `load` (`mojolearn-gp-1`): the training matrix, the Cholesky
  factor, the dual coefficients, `info_`, the kernel's postfix nodes (parameters and length scales kept
  as float64, so the binary32 arrays `predict` builds are the fit's), `alpha`, `normalize_y_` and the
  target mean and std. `normalize_y`'s scale-back stays in `predict`'s host Python, shared with the GPU
  class.
- A new INFERENCE-ONLY host binding, `_mojolearn_gp_infer_host`, shipped in the wheel. The gp host
  binding's predict path moved into `bindings/gp_host_predict.mojo` (`_rebuild_kernel_spec`,
  `gpr_predict_binding`, `gpc_predict_binding`), which the reference binding also registers; the
  inference binding registers `gpr_predict` and `gpc_predict` and imports no `gpr_host_fit`,
  `gpc_host_fit` (the Laplace Newton loop), log marginal likelihood or Cholesky door.
- `HostGaussianProcessClassifier` binds `_mojolearn_gp_infer_host` when it is built and the reference
  binding otherwise, so main's tests and a gate that builds only the routed families keep working.
  The classifier's label decoding is host Python; no core binding call is on its predict path.
- `gmm_sample_binding` moved into `bindings/mixture_host_scoring.mojo`, and `_mojolearn_mixture_infer_host`
  registers `gmm_sample` (`mixture/checks/sample.mojo::gmm_sample_host` reads only the model arrays).

## Metal recordings (Apple M4, one core through mac_slot.sh)

`bench/results/classical_host/2026-09-15-apple-m4-gp-gmm-sample/`: 27 models on base, ties and dupes,
recorded after rebuilding the Metal gp and mixture bindings from the merged sources; each reload
predicted the same bits. Every identity hash equals the committed infer cells that carry one:

| lanes | committed columns the recorded identity hash equals |
|---|---|
| gp, gp-matern12, gp-matern32, gp-matern52-ard | the 166-lane record's Apple M4, NVIDIA H100 and AMD MI325X columns, all nine fixtures of the four lanes |
| gpc, gpc-multiclass | `2026-09-15_gpc/apple-m4.json` and `cpu-x86.json` |
| gmm-sample, gmm-random-init-sample | `2026-09-15_gmm-sample/apple-m4.json` and its x86 column (base and ties; that lane ran no dupes) |
| gp-normalize-y | none: no tracked JSON on main carries a `gp-normalize-y` cell (repo-wide search); its evidence is this Metal recording against the x86 host check below, and its NVIDIA and AMD infer cells are OWED |

The only DIFFER lines in that comparison are against the committed host-sabotage columns of the gpc and
gmm-sample lanes, as they should be.

## The isolated installed test wheel (Mac)

A macOS arm64 wheel built from this tree (commit 73b1b4e7d plus the recordings) with no Metal set and no
MAX runtime, carrying the six host bindings this lane uses: core 581,400, estimators 534,192, svm
397,224, gp_infer 288,184, hdbscan_infer 287,280 and mixture_infer 251,872 bytes. The wheel is
1,369,610 bytes. It was pip-installed into an isolated target and imported from there (`vendor cpu`).

| file | verdict |
|---|---|
| `installed_wheel_check.2026-09-15-apple-m4-gp-gmm-sample.txt` | `gate verdict IDENTICAL (27 fixtures, 0 GPU columns, exit 0)` |
| `installed_wheel_check.2026-09-15-apple-m4-iforest-gmm-hdbscan.txt` | `gate verdict IDENTICAL (16 fixtures, 0 GPU columns, exit 0)` |
| `installed_wheel_check.2026-09-15-apple-m4-neighbors-density.txt` | `gate verdict IDENTICAL (54 fixtures, 0 GPU columns, exit 0)` |
| `installed_refusals.txt` | GaussianProcessRegressor, GaussianProcessClassifier and GaussianMixture fits refuse; `host_model` returns HostGaussianProcessRegressor and HostGaussianProcessClassifier bound to `_mojolearn_gp_infer_host` (`gpr_fit` and `gpc_fit` absent) and HostGaussianMixture bound to `_mojolearn_mixture_infer_host` (`gmm_fit` absent) |
| `fit_symbols_mac.txt` | `nm` over the macOS builds: gp_infer carries the predict path (`gpr_host_kernel_matrix`, `chol_host_trsm_lower`, `gpc_latent_var`, the two predict bindings) and no `gpr_host_fit`, `gpc_host_fit`, `chol_host_potrf` or Newton step; mixture_infer carries `gmmh_score_samples` and `gmm_sample_host` and no EM step |
| `wheel.txt` | the wheel's `mojolearn/host/` listing and size |

What enters the wheel from this lane, all three stages: the three inference-only bindings, 827,336 bytes
together on macOS arm64 (gp_infer 288,184, hdbscan_infer 287,280, mixture_infer 251,872). No reference
gp, mixture or hdbscan binding ships. The neighbor, KDE and iforest lanes add no binary.

## x86 (RunPod CPU pod)

One pod (8 vCPU, hoyqtbk4r8oi8n, 272 s billed, $0.0181, delete verified by GET 404), host bindings
built there at x86-64-v3 from afd1dcd2a through the R2 binding cache. Outputs in `x86/`; raw leg
directory untracked, under ~/mojolearn-evidence/inference-neighbors-density/.

| file | verdict |
|---|---|
| `x86/check_x86_host.2026-09-15-apple-m4-gp-gmm-sample.txt` | through only the six wheel bindings this lane uses: `gate verdict IDENTICAL (27 fixtures, 5 GPU columns, exit 0)`, 45 identity hashes EQUAL to committed infer cells, 90 ABSENT |
| `x86/check_x86_host.2026-09-15-apple-m4-iforest-gmm-hdbscan.txt` | `gate verdict IDENTICAL (16 fixtures, 5 GPU columns, exit 0)`, 36 EQUAL, 18 N/A |
| `x86/check_x86_sabotage.2026-09-15-apple-m4-iforest-gmm-hdbscan.txt` | the check owed since the 41f3462db leg, now with sabotage copies of mixture_infer and hdbscan_infer: `gate verdict EXPECTED MISMATCH SEEN (16 fixtures, 5 GPU columns, exit 0)`, no fixture left EQUAL |
| `x86/check_x86_sabotage.2026-09-15-apple-m4-gp-gmm-sample.txt` | `gate verdict EXPECTED MISMATCH SEEN (27 fixtures, 5 GPU columns, exit 0)`, but the six gmm-sample and gmm-random-init-sample fixtures stayed EQUAL: `gmm_sample_host` reads no GEMM leaf, so the descending-leaf arm cannot move a sample drawn from a saved model (the gmm-sample lane's own host sabotage moved the fit, and its samples inherited that). Closed by a sample sabotage arm in `bindings/mixture_host_scoring.mojo`; see the follow-up below |
| `x86/diff.gp.txt`, `x86/owed.gp.json` | gp and the three Matern lanes against the 166-lane record cut to base, ties and dupes: `summary: IDENTICAL=12`, `summary (infer/model): IDENTICAL=12, OWED=12`, `summary (batch): IDENTICAL=12`, require-columns 4 OK (12 OWED, the new save format's model cells) |
| `x86/diff.gpc.txt` | gpc and gpc-multiclass against their lane's Metal and x86 columns: `summary: IDENTICAL=18`, `summary (infer/model): IDENTICAL=36`, `summary (batch): IDENTICAL=18` |
| `x86/diff.gmm-sample.txt` | gmm-sample and gmm-random-init-sample against their lane's Metal and x86 columns: `summary: IDENTICAL=8, ONE-COLUMN=2`, `summary (infer/model): IDENTICAL=8, N/A=4, ONE-COLUMN=8` (the ONE-COLUMN cells are dupes, which that lane did not run) |
| `x86/diff.gp-normalize-y.txt` | REFUSED on all three fixtures: the reference fit with `normalize_y=True` goes through StandardScaler, and this leg built no preprocessing host binding. Rerun in the follow-up leg with preprocessing built |
| `x86/diff.sabotage.txt` | the gp and gpc lanes under the gp host sabotage build: `summary: DIVERGENT=12, ONE-COLUMN=24`; no train, infer or batch cell left IDENTICAL |
| `x86/test_gpc_surface.txt` | test_gpc_surface PASS on the pod, `test_repeat_batch_save_load_and_host_model` included |
| `x86/test_host_surface.txt` | 135 passed, 1 failed: `test_recordings_and_columns_exist`, because a leg ships only the bench paths it names |
| `x86/binding_sizes.txt` | x86-64: gp_infer 282,816 against gp 376,888; mixture_infer 242,064 against mixture 424,024; hdbscan_infer 273,384 against hdbscan 429,864 bytes |
| `x86/fit_symbols_x86.txt` | mixture: the reference file carries `gmmh_e_step` and `gmmh_m_step`, mixture_infer only `gmmh_score_samples` and `gmm_sample_host`. The Linux gp builds print no matching names in either file, so this listing proves nothing for gp; `fit_symbols_mac.txt` is the gp evidence |

## Follow-up leg (2c7016d24): the sample sabotage arm and gp-normalize-y's CPU column

The stage 3 leg left two gaps: the gmm sample recordings did not move under the host sabotage build,
and gp-normalize-y's reference fit refused for lack of a preprocessing binding. `bindings/mixture_host_scoring.mojo`
now flips the lowest bit of every sampled cell under `-D MOJOLEARN_HOST_SABOTAGE=1` (off in normal builds). On the
M4 first, one core: the six gmm sample recordings through the rebuilt mixture_infer read IDENTICAL, and through
its sabotage build EXPECTED MISMATCH SEEN with all six identity hashes DIFFER. Then one RunPod CPU pod
(68noa7al5r54n2, 160 s billed, $0.0107, delete verified), preprocessing built:

| file | verdict |
|---|---|
| `followup-2c7016d24/check_x86_host.txt` | the 27 gp, gpc and gmm sample recordings through only the shipped bindings: `gate verdict IDENTICAL (27 fixtures, 3 GPU columns, exit 0)` |
| `followup-2c7016d24/check_x86_sabotage.txt` | sabotage copies of gp_infer and mixture_infer: `gate verdict EXPECTED MISMATCH SEEN (27 fixtures, 3 GPU columns, exit 0)`, all 27 identity hashes DIFFER, none EQUAL |
| `followup-2c7016d24/cpu-x86.followup.json`, `diff.gp-normalize-y.txt`, `owed.gp-normalize-y.json` | gp-normalize-y train STABLE on base, ties and dupes; no tracked column carries the lane, so `--require-columns 4 --owed-json` reads OK with 12 parts OWED (train 3, infer and model 6, batch 3). Its x86 infer hashes equal the Metal recording's identity hashes on all three fixtures (base `7c5123dc7a512038`, ties `0e6c262796044cb8`, dupes `3c0b639788527068`) |
| `followup-2c7016d24/diff.gmm-sample.txt` | against the gmm-sample lane's Metal and x86 columns: `summary: IDENTICAL=8, ONE-COLUMN=2`, `summary (infer/model): IDENTICAL=8, N/A=4, ONE-COLUMN=8` |
| `followup-2c7016d24/diff.sabotage.txt` | gp-normalize-y and the gmm sample lanes, CPU against its sabotage column: `summary: DIVERGENT=6, ONE-COLUMN=7`, `summary (infer/model): DIVERGENT=12, N/A=4, ONE-COLUMN=10`; no train, infer or batch cell IDENTICAL |
| `followup-2c7016d24/test_host_surface.txt` | 135 passed, 1 failed (`test_recordings_and_columns_exist`, the leg ships only named bench paths) |

Owed to the release record: NVIDIA and AMD recordings of these 27 models, gp-normalize-y's NVIDIA and AMD
cells, and the model cells of `mojolearn-gp-1`.

