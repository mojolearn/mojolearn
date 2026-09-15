# LANE STATUS: lane/batch-invariance-2 (agent "batch2")

Goal: close the three gaps the `batch` part of tools/identity_break.py declares
it does not test: (1) the backward pass (part `batchgrad`), (2) serving-scale B
at the public surface, (3) ragged and padded sequence batches (a new mask or
lengths argument).

## Done
- Worktree created from origin/main 4eac5719a. Metal identical and host
  bindings copied from the wt-apple-R build of df617c699 (no binding source
  changed between df617c699 and 4eac5719a).

## Running
- nothing

## Next commands
- read the backward contracts, then add `batchgrad` to tools/identity_break.py
