# TargetEncoder device scratch experiment

Base origin/main 973fdf16b. Branch lane/apple-fast-target-scratch.
Define `MOJOLEARN_TARGET_SCRATCH`, opt-in FAST + Apple only.
Build binding: `bindings/build_x_prep.sh`.

TargetEncoder currently calls `_codes` but never consumes its unknown counts.
That adds a serial walk of every row per column (`count_neg_unit`) to both
fit and transform. The candidate skips only that unused computation, keeps
category codes in device scratch, and moves the bucket starts/row indices
from downloaded host arena to device scratch. It explicitly clears bucket
rows so unused tails for omitted categories retain their zero indices.
All encoding arithmetic, bucket ordering, unknown-category semantics and
fitted/output words remain unchanged. This does not revive the old TE_GLOBAL
or TE_ENC parallel-reduction flags.

Run one M3 arm each for target-encoder taxi and istella. No opponent reruns.
Quality in both compiled arms (choose distinct OUT paths):

```
PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=apple \
 ~/board-0834/cache/venv/bin/python tools/target_scratch_quality.py dump OUT.npz
```

Then `... tools/target_scratch_quality.py compare main.npz candidate.npz`.
The checker records all public fitted encodings and transform/cross-fit output
values for continuous/binary/multiclass targets, auto/fixed smoothing, unknown
queries, and explicit categories that omit training values. It independently
checks fixed-smoothing predictions against the mathematical expression. A/B
must be exact because no arithmetic has changed. No local builds/tests run.
