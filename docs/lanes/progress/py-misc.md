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
| nvc1 A40 pod | NVIDIA (cuda) | fit 200k x (1,28,28), conv (8,), batch 32, 1 epoch, median of 3 | 2.192 s | 1.572 s (1.39x) | 469d6166e9698eed both | nvc1-0002 |
| nvc1 A40 pod, Xeon Gold 6342 | CPU host twin | fit 20k x (1,28,28), batch 32, 1 epoch, median of 3 | 8.275 s | 7.161 s (1.16x) | 8d9bb17aa778b5b8 both | nvc1-0002 |
| same job | cuda and cpu | 10 configs (cnn_epoch.py) | | | ALL SAME on both columns; each config's digest equal across columns | nvc1-0002 |
| same job | cuda, cpu | identity lanes x-cnn-trainer, x-cnn-trainer-options, 9 fixtures each | | | before == after per column (IDENTICAL=18 train, infer, batch); after GPU == CPU | nvc1-0002 |

Evidence: ~/mojolearn-evidence/py-misc/cnn1/ and nvc1-0002.log. (An earlier submit, nvc1-0033 on
the first pod, was lost when that pod went down.) No DEVIATION changes: 5717/5718 cover the
resident entries; the epoch entry is plumbing over them, no new device path, so no sabotage arm.

## Unproven / not done
- X over 1 GiB (non-resident path, audit item 2) still steps in Python.
- `loss_curve_` Python `sum` (Python-version dependent bits) left as is.
- Each entry inside the epoch loop still waits on the device (the entries' own synchronize):
  about 250 us per step on the A40. Dropping those waits needs the softmax loss kept on the
  device for the epoch and every host parameter block kept alive past its enqueue (a new
  ordering path, sabotage-proof owed); not done.
- Apple (m2pro) not run yet.
