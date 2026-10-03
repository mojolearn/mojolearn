# Apple FAST neural: handoff to the M3 manager (2026-10-03)

Branch `lane/apple-fast-neural` merges the eight lanes `lane/apple-fast-neural-{tier,gemm,optim,mlp,mamba,attn,lm,samba}`
(each cut from main 8897404da, Release 0.8.35). Code only: nothing here was measured, tested or run, and no box was called.
Andrew's order: the peer compiles (the M3 or the external M2), measures and tests.

## What it is
FAST tier for every GPU neural algorithm, Apple only. Every new path is behind
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()`
plus its own `-D MOJOLEARN_AFN_<NAME>` define, default OFF. No CPU paths: no host/CPU/infer file changed, and the CPU
column is excluded from every guard. Plan: docs/apple-fast/PLAN-neural.md. Per lane: docs/apple-fast/notes/neural-<lane>.md
(profile, candidates, compile table) and docs/apple-fast/ab-neural/<lane>.md (one paragraph per define).

## Compile state (compile on your box first)
| Binding | Lanes | State here |
|---|---|---|
| linalg | gemm | FAST x 6 defines + ALL + off, IDENTICAL: rc=0 (lane branch) |
| embedding, x_cnn | mlp | FAST per define + off, IDENTICAL: rc=0 (lane branch) |
| training | optim, mlp, samba | optim and mlp each rc=0 on their own branch; samba UNCOMPILED (ALL rc=1, aliasing fixed, not rebuilt); the merged training binding never compiled |
| transformer | attn | UNCOMPILED (ALL rc=1: flash SIMD splat needs fill=, fixed, not rebuilt) |
| mamba | mamba, samba (Mamba-3 backward) | UNCOMPILED (ALL rc=1, 9 errors fixed, not rebuilt) |
| byte_lm | lm (also edits transformer/checks/transformer_backward.mojo) | UNCOMPILED; build_byte_lm.sh now accepts fast |
| tools | tier | py_compile and bash -n OK; afn_ab.sh never run |
The merged branch as a whole was never compiled; the CPU-column guard edit (11 one-line changes) came after every lane build.

## Same bits (check before any merge to main)
- training: optim moved the resident optimizer step's buffer declarations above its guard (training/estimator.mojo);
  IDENTICAL does the same work but the text differs from main. ID-check training.
- mamba: samba routes IDENTICAL mamba3_prefill_backward through a scratch wrapper (same mamba_zeros calls). ID-check mamba.
- byte_lm: main's ids-upload block re-indented under a guard's else. ID-check byte_lm.

## Running the A/Bs
88 request lines in docs/apple-fast/ab-neural/*.txt (tier 12 baselines identical vs fast, gemm 9, attn 8, mamba 13,
samba 12, lm 8, optim 20, mlp 6), all `CMD lane/apple-fast-neural <tag> bash tools/afn_ab.sh ...`. They are NOT in
docs/apple-fast/ab/ on purpose, so your watcher queues nothing uncompiled; move them (or a subset) there when the builds pass.
`tools/afn_ab.sh <tag> <binding> <board-lane|custom> <shape> <reps> "<defines A>" "<defines B>"` races our arm only, then
judges quality (output diff + tools/neural_fast_quality.py at one seed). Lines needing another lane's work: optim's lm lines
need the FAST byte LM (in this branch). Board: `--modes fast` is allowed for the neural family on Apple only.

## Keep rule
Faster on the M3 and quality within FAST spread (vs IDENTICAL and torch, same weights and batches) -> FAST default, switch
removed, `_OFF` define kept -> main.
