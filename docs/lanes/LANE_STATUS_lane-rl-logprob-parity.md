# Lane status: lane/rl-logprob-parity (the `rlpair` harness part)

Goal: prove "the sampler's log-probabilities equal the trainer's" bit for
bit, as the `rlpair` part of tools/identity_break.py, for the byte LM,
TransformerBlock, Mamba1/2/3Block and SambaStack.

## Done (2026-09-15)
- `rlpair` part in tools/identity_break.py (12 lanes), `summary (rlpair):` in
  `--diff`, `--no-rlpair`, MOJOLEARN_IDENTITY_RLPAIR_SABOTAGE.
- SambaStack.allocate_state / forward(ids, state) / step (python/mojolearn/_samba_impl.py).
- Columns at 8d4fa23af: Apple M4 Metal, RunPod H100 (leg
  bench/results/e1g/2026-09-15_134722-nvidia-h100-rlpair, pod deleted, HTTP 404
  verified; copy in ~/mojolearn-evidence/rlparity/), M4 CPU probe through the
  unmerged lane/cpu-training-samba host set. rlpair IDENTICAL=108, sabotage
  RLPAIR_MOVED on all hashed cells, 21/21 negative controls.
- Evidence: bench/results/identity_break/2026-09-15_rlpair/; docs: IDENTITY_PATHS.md section.

## Running
- nothing rented. CPU identity gate run 34980556800 (workflow_dispatch on 7ae35df48; lane/rl-* pushes do not trigger it). When green: fetch, merge origin/main, light checks, push HEAD:main.

## Owed, with commands
- AMD column: with the next release record (GPU runs only for PyPI releases, 2026-09-15 rule).
- CPU column from main once lane/cpu-training-mamba, -transformer and -samba merge:
  build the host set (bindings/build_{core,training,mamba,transformer,byte_lm}_host.sh, -j 1),
  then `MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --lanes
  byte-lm-host-infer,byte-lm-host-infer-threaded,mamba1,mamba2,mamba3,mamba2-dtlimit,transformer,transformer-window,samba,samba-untied-dropout-accum
  --vendor cpu-apple-m4 --json cpu-apple-m4.json` through a CPU-only package, and
  `--diff` it against apple-m4.json and nvidia-h100-sm_90a.json in the record dir.
