# py-misc: progress (Python front door, misc families)

Branch `lane/py-misc` (worktree ~/mojolearn-wt/py-misc), from lane/apple2-merged 342469dae.
Brief: ~/mojolearn-evidence/py_work_brief.md; audit ~/mojolearn-evidence/python_work_audit.md.
Sub-lanes (own worktrees, merged here when proven): lane/py-misc-metrics (metrics epilogues,
DEVIATION 6106 / row 139), lane/py-misc-msel (model selection), lane/py-misc-prep
(IterativeImputer user estimator, CalibratedClassifierCV).

## cnn: CNNClassifier.fit epoch entry (audit rank 10)

1edeabdb5, 83741a834. `x_cnn_fit_epoch_r` (GPU binding and host twin): each epoch's steps
in one binding call, looping in Mojo over the same `_into` calls the Python step loop made,
with the same arguments in the same order (GPU: the list forms, pair gather; host twin: two
gathers, one optimizer entry per parameter, the same `out_f32` seams). Adam's per-step
hyper rows are still computed in Python (`_adam_hyper`, libm pow) and passed as a float64
table; SGD's first-step flag likewise. Losses come back as float64 of the Float32 the
softmax entry returned; `loss_curve_` is still Python `sum(epoch)/len(epoch)` (unchanged,
Python-version dependent, see unproven). Before arm: `_expansion_cnn._EPOCH_ENTRY = False`
or `MOJOLEARN_XCNN_PY_STEPS=1` (the Python step loop, same build).
Proof script: tools/py_misc/cnn_epoch.py (10 configs: sgd, 2 blocks, nesterov, dampening,
adam, adamw, batch > n, no shuffle, no pool, no conv; digest of losses_, loss_curve_,
weights, optimizer buffers, predict_proba); job: tools/py_misc/cnn_job.sh.

| machine | column | what | before (Python loop) | after (epoch entry) | digests | job |
|---|---|---|---|---|---|---|
| nvc1 pod x86 CPU (no GPU, `sh`) | CPU host twin | 10 configs | | | ALL SAME | sh 19:20Z |
| nvc1 pod x86 CPU | CPU host twin | fit 2000 x (1,28,28), batch 32, 1 epoch | 0.384 s | 0.331 s | same | sh 19:20Z (1 rep, smoke) |

## Unproven / not done
- X over 1 GiB (non-resident path, audit item 2) still steps in Python.
- `loss_curve_` Python `sum` (Python-version dependent bits) left as is.
- Each entry inside the epoch loop still waits on the device (the entries' own synchronize).
