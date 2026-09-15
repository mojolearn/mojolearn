# The rlpair part: sampler log-probabilities against trainer log-probabilities (2026-09-15)

First record of the `rlpair` part of `tools/identity_break.py` (lane
`lane/rl-logprob-parity`, commit 8d4fa23af). The part, its assertions and its
limits are the module docstring's `rlpair` entry and the IDENTITY_PATHS.md
section "The sampler's log-probabilities equal the trainer's". Twelve lanes,
nine fixtures, two fits per cell, every other part (train, infer, model, batch)
run as usual.

| column | box | how | cells |
|---|---|---|---|
| `apple-m4.json` | Apple M4, Metal, this Mac, one core, shared machine | worktree at 8d4fa23af with the Metal identical set built at df617c699 (the only Mojo change between the two is checks/hardware_matrix_check.mojo and checks/kernel_matrix.mojo), `--vendor apple-m4`, 556 s | 108 stable, rlpair STABLE=108 |
| `nvidia-h100-sm_90a.json` | RunPod H100 80GB HBM3, one GPU | leg `bench/results/e1g/2026-09-15_134722-nvidia-h100-rlpair` (kept in ~/mojolearn-evidence/rlparity/), seven bindings built from 8d4fa23af on the box (`nvidia-h100-sm_90a.gate.txt`), 104 s | 108 stable, rlpair STABLE=108 |
| `cpu-apple-m4.probe.json` | Apple M4 CPU, CPU-only package, one core, shared machine | A PROBE, NOT A RECORD: this lane's harness over the python tree and host bindings of the UNMERGED lane/cpu-training-samba (4cc3609e2 plus its uncommitted work, host set `cpu-apple-m4.probe.host-bindings.sha256`), because the Mamba, Transformer and Samba CPU paths are not on main yet. `byte-lm` and `byte-lm-resident` have no CPU trainer and are not run. 1509 s | 90 stable, rlpair STABLE=90 |

No AMD column. The MI325X leg was not rented: Hot Aisle had no single-GPU
stock and the DigitalOcean lock was held when this ran, and Andrew's rule of
2026-09-15 then limited GPU runs to PyPI release records. The AMD column (and a
fresh NVIDIA column) come with the next release record, which will carry the
part without any change here.

## Verdicts

`diff.three-columns.txt`: `summary: IDENTICAL=108`, `summary (infer/model):
IDENTICAL=144, N/A=72`, `summary (batch): IDENTICAL=108`, **`summary (rlpair):
IDENTICAL=108`**; 90 rlpair cells are IDENTICAL x3 (Metal, CUDA, M4 CPU) and
the 18 byte-lm and byte-lm-resident cells IDENTICAL x2 (Metal, CUDA). No MOVED,
RLPAIR_MOVED, DIVERGENT or REFUSED cell. The train, infer and batch hashes of
the two GPU columns also equal the 166-lane record's three columns on these
twelve lanes, so the bindings here are the record's arithmetic.

What that says, per lane, on every fixture: greedy decode through the lane's
state API at batch 5 and batch 1, the trainer's teacher-forced forward at batch
5, batch 1 and the split 2, 3, and continuous batching (row 4 joins at token 3,
row 1 leaves after token 8) give the same ids, logits and log-probabilities
(NLL through `training.cross_entropy(reduction="none")`) bit for bit, and the
same bytes on the Apple GPU, the H100 and the M4 CPU. On the byte-lm lanes the
second sampler was LanguageModelInference on the box's CPU with the trainer's
parameters, on both GPU columns (`rlpair_sides`), so the CPU sampler and the
GPU trainer agreed inside each process too.

## The sabotage, seen to fail

`MOJOLEARN_IDENTITY_RLPAIR_SABOTAGE=1` on `base`, one repeat: every hashed
rlpair cell reads RLPAIR_MOVED on every column (`diff.apple-m4.rlpair-sabotage.txt`
RLPAIR_MOVED=12, `diff.nvidia-h100-sm_90a.rlpair-sabotage.txt` RLPAIR_MOVED=12,
`diff.cpu-apple-m4.probe.rlpair-sabotage.txt` RLPAIR_MOVED=10), each naming
`sampler B=5 vs trainer B=5:seq 0 token 0:nll` with the two values one bit
apart, for example mamba2 `0x3f74221c vs 0x3f74221d` on both the H100 and the
M4 CPU.

The sabotage reaches assertion 1 only, so `rl_controls.py` (run on the M4 Metal
set, `controls.apple-m4.txt`) breaks each other assertion's input once and
requires RLPAIR_MOVED naming that assertion: a bit in the batch-1 sampler, in
the batch-1 trainer, in the 2, 3 split trainer, in the second byte LM sampler,
the joined state's row order and the left state's row order, on mamba2,
transformer, samba and byte-lm-host-infer. 21 of 21 fail as they must and the
unpatched cells return their hashes. The first version of two controls did not
fail (a flip through `reshape` of a non-contiguous view changed a copy, and a
patch of `_rl_cat` applied twice through SambaState's recursion undid itself);
both were control bugs, fixed, and the harness's own sabotage flip now refuses a
non-contiguous array.

## What is not shown

Anything the IDENTITY_PATHS.md section lists as not proven: RNG sampling,
temperature, top-k, top-p, rows at different positions in one batch, padding,
a byte LM KV cache (there is none; its sampler recomputes the prefix), the
backward pass and the optimizer step, long sequences (neither Mamba chunk
boundary is crossed; the window-8 transformer ring is), other lanes, and the
AMD column. The CPU column is a probe on unmerged host bindings and must be
rerun from main once lane/cpu-training-mamba, -transformer and -samba merge.
