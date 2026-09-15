# Public CPU inference for IsolationForest, GaussianMixture and HDBSCAN approximate_predict (2026-09-15)

Branch `lane/inference-neighbors-density`. Six lanes that had internal CPU reference code now
serve CPU inference from a model saved on a GPU: iforest, iforest-tuned, gmm, gmm-random-init,
hdbscan and hdbscan-leaf.

What changed:

- `IsolationForest.save` and `load` (`mojolearn-iforest-1`). Every scoring call rebuilds the forest
  from the training matrix (DEVIATION 874), so the file holds that matrix and the resolved knobs,
  and the svm host binding's `iforest_run` scores it, which already shipped.
- `GaussianMixture.save` and `load` (`mojolearn-gmm-1`) and `HDBSCAN.save` and `load`
  (`mojolearn-hdbscan-2`, prediction data required) for `mojolearn.hdbscan.approximate_predict`,
  `membership_vector` and `all_points_membership_vectors`. Version 2 adds the tree parents,
  exemplar indices, exemplar offsets and exemplar count the soft clustering calls read, after
  main's HDBSCAN soft clustering merged (41f3462db); no version 1 file left this branch.
- Two INFERENCE-ONLY host bindings that ship in the wheel, `_mojolearn_mixture_infer_host` and
  `_mojolearn_hdbscan_infer_host`, following the neural family's pattern (`routes=None`). The
  scoring and prediction entries moved into `bindings/mixture_host_scoring.mojo` and
  `bindings/hdbscan_host_predict.mojo`, which the reference bindings and the inference bindings
  both register. `mojolearn.host_model(path)` loads a saved model into a host class that binds
  the inference file, as lane/inference-linear-svm serves the scalers.
- `tools/classical_host_gate.py` reads a committed infer cell that holds an `n/a:` reason as N/A.
  The 166-lane record carries `n/a:transductive` on hdbscan and hdbscan-leaf, which predate their
  infer probe; the first check of these recordings read MISMATCH on exactly those 18 cells while
  every recording SHA-256 read EQUAL.

Where it ran: the Apple M4, every core-holding step through `mac_slot.sh` at one core, shared
machine. Metal bindings for mixture and hdbscan built from this tree; the others were copied from
a sibling worktree at the same Mojo sources, and every identity hash that a committed Apple column
carries matched it.

| file | verdict |
|---|---|
| `bench/results/classical_host/2026-09-15-apple-m4-iforest-gmm-hdbscan/` | 16 models saved by the Metal classes: iforest and iforest-tuned on base and ties (each model holds the 20000 x 16 training matrix), gmm, gmm-random-init, hdbscan and hdbscan-leaf on base, ties and dupes; each reloaded on Metal to the same bits |
| `classical_check_host.txt`, `check_apple-m4_host.json` | through a host directory holding ONLY the svm, mixture_infer and hdbscan_infer bindings: `gate verdict IDENTICAL (16 fixtures, 4 GPU columns, exit 0)`; 36 identity hashes EQUAL to committed infer cells (iforest and gmm against the 166-lane record's Apple, NVIDIA and AMD columns; hdbscan against the hdbscan-predict Apple column), 18 N/A, 10 ABSENT |
| `classical_check_sabotage.txt`, `check_apple-m4_sabotage.json` | `-D MOJOLEARN_HOST_SABOTAGE=1`: `gate verdict EXPECTED MISMATCH SEEN`, all 16 identity hashes DIFFER |
| `fit_symbols.txt` | `nm` over the built files: the reference mixture binding carries `gmmh_e_step` and `gmmh_m_step`, the inference one `gmmh_score_samples` only; the reference hdbscan binding carries `hdbh_fit`, `generate_prediction_data` and `boruvka`, the inference one `hdbh_approximate_predict` only. 232,112 against 406,456 bytes and 227,696 against 359,688 |

The x86 CPU columns ran on one RunPod CPU pod (8 vCPU, host bindings built there at x86-64-v3,
commit f1d9feecb; leg `bench/results/runpod_cpu/2026-09-15_171130-inference-neighbors-density`,
untracked; 473 s billed, $0.03, delete verified). The pod's pixi Python has no pip, so its wheel
step built nothing; the installed test wheel ran on the Mac instead.

| file | verdict |
|---|---|
| `cpu-x86.iforest-gmm-hdbscan.json` | reference fits through the x86 host bindings, base, ties and dupes, two repeats |
| `diff.four-columns.base-ties-dupes.txt`, `owed.iforest-gmm.json` | iforest and gmm against the 166-lane record cut to those fixtures: `summary: IDENTICAL=12`, `summary (infer/model): IDENTICAL=12, OWED=12`, `summary (batch): IDENTICAL=12`, require-columns 4 OK (12 OWED, the new save formats' model cells) |
| `diff.hdbscan.txt`, `owed.hdbscan.json` | hdbscan against the hdbscan-predict Apple column with the record's NVIDIA and AMD columns: `summary: IDENTICAL=6`, `summary (infer/model): OWED=12`, `summary (batch): OWED=6`, require-columns 4 OK (18 OWED) |
| `cpu-x86.iforest-gmm-hdbscan.sabotage.json`, `diff.sabotage.txt` | `-D MOJOLEARN_HOST_SABOTAGE=1` svm, mixture and hdbscan builds: `summary: DIVERGENT=18`, `summary (infer/model): DIVERGENT=18, ONE-COLUMN=18`, `summary (batch): DIVERGENT=18`; no train, infer or batch cell stays IDENTICAL |

## After merging main's HDBSCAN soft clustering (41f3462db)

main's identity_break `hdbscan` and `hdbscan-leaf` infer probe now hashes approximate_predict's
labels and probabilities, `membership_vector` on the same 256 rows and
`all_points_membership_vectors`. The membership bindings moved into
`bindings/hdbscan_host_predict.mojo` beside approximate_predict, so the inference-only hdbscan
binding registers all three and still no fit, prediction data generation or tree building
(`fit_symbols.txt`, merged section: `hdbh_approximate_predict`, `hdbh_membership_vector`,
`hdbh_all_points_membership_vectors`, `hdbh_soft_pass`, 287,280 bytes). The two lanes were
re-recorded on Metal with the binding rebuilt from the merged sources: all six identity hashes
equal the membership lane's committed Metal column
(`bench/results/identity_break/2026-09-15_hdbscan-membership-vector/apple-m4.json`: hdbscan base
and dupes `cfe5f8df2a406451`, ties `6ec7f68f0f81ae9b`; hdbscan-leaf base and dupes
`2117320835e91e8a`, ties `47607746aff5c75c`). The iforest and GMM recordings did not change.
The x86 confirmation on the merge commit is in `merged-41f3462db/`.

The isolated installed test wheel, on the Mac, rerun after the merge: a macOS arm64 wheel
(1,146,506 bytes) built from this tree with no Metal set and no MAX runtime, carrying only the
five host bindings this lane uses (core 581,400, estimators 534,192, svm 397,224, mixture_infer
232,112, hdbscan_infer 287,280 bytes), pip-installed into an isolated target and imported from it
(`vendor cpu`).

| file | verdict |
|---|---|
| `installed_wheel_check.txt`, `installed_wheel_check.json` | the 16 recordings from the installed package, the re-recorded hdbscan models included: `gate verdict IDENTICAL (16 fixtures, 0 GPU columns, exit 0)` |
| `../2026-09-15_inference-neighbors-density/installed_wheel_check.txt` | the 54 neighbor and KDE recordings: `gate verdict IDENTICAL (54 fixtures, 0 GPU columns, exit 0)` |
| `installed_refusals.txt` | GaussianMixture, HDBSCAN, RadiusNeighbors and IsolationForest fits refuse; `host_model` on the saved gmm and hdbscan models returns HostGaussianMixture and HostHDBSCAN bound to `_mojolearn_mixture_infer_host` and `_mojolearn_hdbscan_infer_host`, in which `gmm_fit` and `hdbscan_fit` are absent |
| `wheel.txt` | the wheel's `mojolearn/host/` listing and size |

What enters a wheel from this lane: the two inference-only bindings, about 520 KB together on
macOS arm64 (232,112 and 287,280 bytes after the merge; 222,368 and 205,688 bytes on the x86 pod
before it); no reference mixture or hdbscan binding,
so no EM step, Boruvka MST or prediction data generation. The neighbor, KDE and iforest lanes add
no binary; they run on the core, estimators and svm bindings that already shipped.

No GPU box was rented (the release-only GPU rule). Owed to the release record: NVIDIA and AMD
recordings of these 16 models, the NVIDIA and AMD infer cells of hdbscan and hdbscan-leaf, and the
model cells of the four new save formats. The CPU identity gate workflow does not build the two
inference-only families yet, so these recordings sit in `host_surface.INFERENCE_ONLY_RECORDED`
rather than CLASSICAL_RECORDED; that workflow change is owed to its owner.
