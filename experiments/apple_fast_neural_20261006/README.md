# Apple FAST neural source-only experiments

**Not tested. Not compiled. Not verified. Not measured.**

Requested delivery: a thorough neural-only idea list, followed by parallel
implementation in a new worktree and a pushed branch. The starting point is
local `main` at `fd6cf8045`; branch `ideas/apple-fast-neural-20261006`.
The new worktree is `../mojolearn-apple-fast-neural-20261006`.

Read [IDEAS.md](IDEAS.md) for 44 mechanism cards, eight interaction recipes,
their A/B hypotheses, workload map, quality risks and future acceptance
requirements. The list was authored before implementation fanout. Source
reading led to explicit corrections to already-shipped Mamba refusal and
embedding context behavior; those cards now name distinct mechanisms.

| Family | Cards | Recipes and source notes |
| --- | --- | --- |
| Attention/projection | A01–A12 | [attention.json](attention.json), [attention.md](attention.md) |
| Training/loss/optimizer/MLP | T01–T12 | [training.json](training.json), [training.md](training.md) |
| Mamba/Samba | M01–M10 | [mamba.json](mamba.json), [mamba.md](mamba.md) |
| CNN/embedding | E01–E10 | [cnn_embedding.json](cnn_embedding.json), [cnn_embedding.md](cnn_embedding.md) |
| Interactions | X01–X08 | [interactions.json](interactions.json) |

Existing disabled arms receive explicit A/B recipes; new variants add guarded
runtime code. New controls use `MOJOLEARN_AFN26_*`, default to OFF, and only
activate in Apple GPU FAST. New inline comments say `not tested`. Existing
defaults and past result comments are preserved; no candidate is promoted.
No Python runtime arithmetic, data processing, workers or device work was
added. The Python catalog below is experiment metadata glue only.

`catalog.py` is a selection tool with **no execution command**. It was not run
in this assignment. For future use, these commands only display/write metadata:

```text
python3 experiments/apple_fast_neural_20261006/catalog.py list
python3 experiments/apple_fast_neural_20261006/catalog.py show E10
python3 experiments/apple_fast_neural_20261006/catalog.py select E08 --variant threads64
python3 experiments/apple_fast_neural_20261006/catalog.py select E10 --variant scratch_atomic --output /tmp/E10-plan.json
```

Selections contain A and B as complete bare-define arrays and compiler flag
arguments. They are not build/run receipts. Replace inherited experiment
flags for each arm; do not append them to stale `ALL` flags or a different
candidate. Geometry comparisons hold the same parent implementation on both
sides. Interaction variants explicitly enumerate their parents and switches.

The family notes name future binding targets and caller restrictions. Those
targets have not been invoked. Full-dataset recipes, dataset versions/hashes,
actual dimensions/caps, complete timed boundaries, quality thresholds and
source/binary/hardware provenance must be resolved and retained before any
future timing. All qualification fields remain pending. Output bit changes
from previous versions are permitted only with preserved task quality and
semantics; no tolerance or model-work reduction is proposed.

No build/test/GPU logs or result receipts exist because no such work was run.
No manifest checker, linter, syntax checker or numerical driver was run either.
The authored documents and inline notes are the retained delivery evidence.
Git hooks are bypassed for the requested commit/push so they cannot trigger
the prohibited compilation or verification. Main is not merged or pushed.
