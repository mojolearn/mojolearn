# Public CPU inference for IsolationForest, GaussianMixture and HDBSCAN approximate_predict (2026-09-15)

Branch `lane/inference-neighbors-density`. Six lanes that had internal CPU reference code now
serve CPU inference from a model saved on a GPU: iforest, iforest-tuned, gmm, gmm-random-init,
hdbscan and hdbscan-leaf.

What changed:

- `IsolationForest.save` and `load` (`mojolearn-iforest-1`). Every scoring call rebuilds the forest
  from the training matrix (DEVIATION 874), so the file holds that matrix and the resolved knobs,
  and the svm host binding's `iforest_run` scores it, which already shipped.
- `GaussianMixture.save` and `load` (`mojolearn-gmm-1`) and `HDBSCAN.save` and `load`
  (`mojolearn-hdbscan-1`, prediction data required) for `mojolearn.hdbscan.approximate_predict`.
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

The isolated installed test wheel, on the Mac: a macOS arm64 wheel (1,119,513 bytes) built from
this tree with no Metal set and no MAX runtime, carrying only the five host bindings this lane
uses (core 581,400, estimators 534,192, svm 397,224, mixture_infer 232,112, hdbscan_infer 227,696
bytes), pip-installed into an isolated target and imported from it (`vendor cpu`).

| file | verdict |
|---|---|
| `installed_wheel_check.txt`, `installed_wheel_check.json` | the 16 recordings from the installed package: `gate verdict IDENTICAL (16 fixtures, 0 GPU columns, exit 0)` |
| `../2026-09-15_inference-neighbors-density/installed_wheel_check.txt` | the 54 neighbor and KDE recordings: `gate verdict IDENTICAL (54 fixtures, 0 GPU columns, exit 0)` |
| `installed_refusals.txt` | GaussianMixture, HDBSCAN, RadiusNeighbors and IsolationForest fits refuse; `host_model` on the saved gmm and hdbscan models returns HostGaussianMixture and HostHDBSCAN bound to `_mojolearn_mixture_infer_host` and `_mojolearn_hdbscan_infer_host`, in which `gmm_fit` and `hdbscan_fit` are absent |
| `wheel.txt` | the wheel's `mojolearn/host/` listing and size |

What enters a wheel from this lane: the two inference-only bindings, about 460 KB together on
macOS arm64 (205,688 and 222,368 bytes on the x86 pod); no reference mixture or hdbscan binding,
so no EM step, Boruvka MST or prediction data generation. The neighbor, KDE and iforest lanes add
no binary; they run on the core, estimators and svm bindings that already shipped.

No GPU box was rented (the release-only GPU rule). Owed to the release record: NVIDIA and AMD
recordings of these 16 models, the NVIDIA and AMD infer cells of hdbscan and hdbscan-leaf, and the
model cells of the four new save formats. The CPU identity gate workflow does not build the two
inference-only families yet, so these recordings sit in `host_surface.INFERENCE_ONLY_RECORDED`
rather than CLASSICAL_RECORDED; that workflow change is owed to its owner.
