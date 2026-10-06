# Apple FAST neural source-only experiments

**Not tested. Not compiled. Not verified. Not measured.**

Requested delivery: a thorough neural-only idea list, followed by parallel
implementation in a new worktree and a pushed branch. The starting point is
local `main` at `fd6cf8045`; branch `ideas/apple-fast-neural-20261006`.
The new worktree is `../mojolearn-apple-fast-neural-20261006`.

The [complete experiment and file index](EXPERIMENT_INDEX.md) lists every new
named A/B variant with its exact baseline/candidate defines and implementation
paths, followed by the existing neural A/B experiments found in this worktree.
It separates new source proposals, reused arms, historical recipes and pending
integration/qualification.

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

## Integration status and entry points

The initial delivery connected source controls to runtime kernels and supplied
standalone metadata. It did **not** connect that metadata to the existing
experiment tools. The integration follow-up adds the shared adapter
[`tools/apple_fast_neural_ideas.py`](../../tools/apple_fast_neural_ideas.py)
and source-only entry points in all three existing tools:

```text
python3 tools/performance_ideas.py list --mode fast
python3 tools/performance_ideas.py plan AFN26-E08 --variant threads64 --stage build --vendor apple --output /tmp/AFN26-E08
python3 tools/neural_experiments.py --apple-fast-plan AFN26-E08 --variant threads64
bash tools/afn_ab.sh --experiment-plan AFN26-E08 --variant threads64
bash tools/afn_ab.sh --experiment-list
```

These commands were **not run**. They are authored metadata-only interfaces.
The global IDs are `AFN26-A01` through `AFN26-X08` in their respective ranges;
the prefix avoids colliding with the legacy AMD IDENTICAL `A01` card. Every
interface reads the existing family/interaction JSON as its single source of
A/B defines. Plans include binding scripts, define transport, potential public
driver routes, and explicit missing coverage. New inline integration comments
also say `not tested`.

The legacy runtime-toggle runner's `MOVED` digest rejection is not applied to
these FAST plans: task quality is the acceptance rule, and changed bits alone
are allowed. Its existing numerical execution path remains separate.

| Integration layer | Status |
| --- | --- |
| Runtime switches, new kernel bodies and existing call sites | Programmed on this branch; not compiled or verified |
| Individual A/B controls and interaction combinations | Authored in the shared catalog |
| Main experiment discovery and existing neural/A-B plan entry points | Programmed by the integration follow-up; not run |
| Fully executable, frozen, end-to-end A/B harness for every affected operation | **Still pending**; metadata and prospective legacy command arrays are not this harness |
| Exact full-dataset mapping, route evidence and quality acceptance | **Still pending**, including MLP multistep and CNN/embedding consumers |
| Board admission, default promotion, main merge | Not performed |

`performance_ideas.py execute AFN26-...` refuses before invoking any builder or
driver. Its legacy `check` command reports its old executable-manifest scope
and does not pretend to validate this different, untested source catalog.
Legacy positional `afn_ab.sh` still builds and measures: only the new
`--experiment-plan` / `--experiment-list` entry points are metadata-only.
The plan names those legacy commands as prospective, unqualified routes, not
execution-ready whole-operation qualification. Multiple affected bindings
must eventually be frozen together; separate legacy one-binding runs do not
prove the combined configuration. No evidence or completed work is invented.
