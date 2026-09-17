# lane/infer-speed-neural: faster neural inference in IDENTICAL mode, no bit moved

Written 2026-09-17 for a session with no memory. Everything measured here ran
on one rented RunPod RTX 4090 (driver 580.126.20, CUDA column `sm_89`, Mojo
from the repository's pixi lock) and on that pod's AMD EPYC 7532 for the CPU
column. Evidence outside the repository:
`~/mojolearn-evidence/infer-speed-neural/` (pod state, A/B records and logs,
identity JSONs, sweep and gate reports). The small A/B records are committed
under `bench/results/infer_speed_neural_2026-09-17/`.

## What changed, and why no output bit can move

Three changes, each a change of LIFETIME or of host bookkeeping. None respells
an arithmetic seam; every kernel, its launch order and its operands are the
ones the per-call paths already ran.

### 1. `TransformerBlock.decode_session(state)` (DEVIATION 2940)

`bindings/_mojolearn_transformer.mojo` gains a `TransformerDecodeSession`
Python object holding one `DeviceContext`, the nine weights as
`LlamaDeviceWeights`, the caller's cache as `LlamaKVCache`, the rotary table
at `(theta, head_dim, max_tokens)` and the `LlamaDeviceStages` at
`(B, 1, max_tokens, window)`. `step` uploads one token per row, calls
`llama_decoder_layer_forward` at `l = 1` and `pos0 = cached_tokens`, the SAME
certified entry point the per-call `transformer_decode_step` calls, and
downloads the block output, one completion wait per token. `forward` runs `L`
tokens on the resident weights and cache with stages built for that call.

The invariant: the per-call entry builds exactly these structs from exactly
these bytes on every token and destroys them; the session builds them once.
Reusing a stage struct across calls is `training/byte_lm.mojo::ByteTrainer`'s
own pattern (its per-layer `forward` stages serve every training step), and
every stage the block reads it writes first in the same call.

Ownership and refresh, the rule `docs/NEURAL_METAL_DECODE.md` asks for: the
session COPIES the weights and the cache at open. While it is open the
state's `k_cache`/`v_cache` are stale and `TransformerBlock.forward`/`step`
REFUSE that state by name (`_refuse_resident`); `cached_tokens` is kept
current. `sync_state()` and `close()` copy the cache back; `load_state()`
re-uploads the caller's cache (the state half of refresh); the weight half is
`close()` and a new `decode_session`. On the CPU host route
(`TransformerBlockInference`) the binding exports no session and
`decode_session` refuses by name.

### 2. `Mamba1Block.decode_session(state)` (DEVIATION 2941)

The same object on `bindings/_mojolearn_mamba.mojo`: `MambaDeviceWeights`,
`MambaDeviceState` (conv window and `h`), `MambaDeviceStages` at `(B, 1)` and
the resident x buffer; `step` is `mamba_step`, the certified decode entry (the
block at `L = 1`). Same ownership rule, same refusal in `Mamba1Block._call`.

Mamba-2, Mamba-3 and Samba have no session yet (see "Not done").

### 3. Byte LM logits: one wait and a retained scratch (DEVIATION 2942)

`training/byte_lm_logits.mojo::_logits_forward` enqueued five completion waits
between the ids upload, the embedding, each block and the head GEMM, all on one
in-order context; they ordered nothing and are gone, leaving the download's
wait. The call-shaped buffers (ids, the embedding output, the KV cache reset
per block, one stage struct per block, the logits and the head GEMM workspace)
are a `ByteLogitsScratch` that the resident `ByteLMSession` keeps across
`logits` calls of the same `[batch, length]` and rebuilds when the shape
changes; the stateless entry builds one per call and destroys it with its
context, as before. Every buffer is written whole before it is read (the ids by
the upload, `x` by the embedding, the cache from `s = 0`, the stages by the
block, the logits by the head GEMM, whose workspace is scratch the GEMM writes
before reading, as the trainer's own retained workspaces are).

Python: `_byte_lm_host._refuse_ids` settles the common case with two C-speed
scans of the view (`min`, `max`) and keeps its loop only to name the first
offender with the same message. It serves both the CPU class and the GPU
trainer's `logits`.

## Numbers (RTX 4090, IDENTICAL, one process per A/B, arms interleaved)

`tools/bench_neural_decode.py --resident-ab`: a fresh state prefilled to 1024
positions through the per-call `forward`, then 64 decode tokens through the
per-call `step` (arm `percall`) or a `decode_session` (arm `resident`), five
rounds after a warmup with the arm order reversed every round. Every decoded
output and both final state pieces are bytewise equal across arms and rounds
and equal to the fresh full forward over the 1088 positions (the `stepfull`
property), asserted per round. Medians are per token.

| model | d_model | B | per-call ms | resident ms | spread (per-call, resident) | paired ratio |
|---|---|---|---|---|---|---|
| Transformer (16 heads, 4 kv, hd 64, ff 2816) | 1024 | 1 | 9.440 | 0.817 | 1.002, 1.057 | 11.54 |
| Transformer | 1024 | 8 | 12.528 | 0.987 | 1.005, 1.014 | 12.68 |
| Mamba-1 (d_inner 2048) | 1024 | 1 | 31.801 | 0.946 | 1.015, 1.006 | 33.61 |
| Mamba-1 | 1024 | 8 | 26.930 | 1.234 | 1.002, 1.005 | 21.82 |

Records: `bench/results/infer_speed_neural_2026-09-17/*_d1024_b*_p1024_t64.json`.

Byte LM logits, the shipped profile, `~/mojolearn-evidence/infer-speed-neural/
bytelm_ab/probe_{base,cand}.log`, four alternating processes (base, cand, cand,
base), 15 calls each after a warmup, medians:

| call | before ms | after ms | ratio |
|---|---|---|---|
| resident, batch 8, length 32 | 3.05 / 3.04 | 1.63 / 1.66 | 1.85 |
| resident, batch 256, length 32 | 44.94 / 44.90 | 35.40 / 35.89 | 1.26 |
| stateless, batch 8, length 32 | 6.61 / 6.71 | 6.67 / 6.60 | 1.00 |
| stateless, batch 256, length 32 | 48.27 / 48.66 | 47.76 / 47.98 | 1.01 |

The stateless call is unchanged because its time is the per-call context and
the weight uploads through the trainer's own helpers, which this lane does not
touch. At batch 256 the remaining time is `attn.core`, about 14 ms per block
under `MOJOLEARN_TRANSFORMER_TIMING=1`, a kernel matter outside this lane.

## Identity

`tools/identity_break.py` on the twelve neural lanes (`mlp`, `transformer`,
`transformer-window`, `mamba1`, `mamba2`, `mamba2-dtlimit`, `mamba3`, `samba`,
`samba-untied-dropout-accum`, `byte-lm`, `byte-lm-resident`,
`byte-lm-host-infer`), fixtures `base,ties,odd`, `--step-full`, repeats 2,
`--no-rlpair` (the rlpair part needs the training binding on both columns and
is not an inference part). BEFORE is the unmodified tree at `e3213a59a` with
its own bindings; AFTER is this branch's tree with its own bindings, the
transformer, mamba and byte LM `.so` digests differing from BEFORE and every
other binding's equal (`pod_out/so_sha256_{base,cand}_full.txt`). The AFTER
JSONs' `commit` field reads the shipped base commit because the candidate tree
on the pod carried no `COMMIT` file until the sabotage run; the digests are
the witness. Diffed with `--diff`:

| column | cells | train | infer/model | batch | stepfull | MOVED or DIVERGENT |
|---|---|---|---|---|---|---|
| cuda, RTX 4090 | 36 (12 lanes x 3 fixtures) | IDENTICAL=36 | IDENTICAL=51, N/A=21 | IDENTICAL=36 | IDENTICAL=24, N/A=12 | 0 |
| cpu, EPYC 7532 (host route, ten lanes: no `byte-lm`, `byte-lm-resident`, which are GPU objects) | 30 | IDENTICAL=30 | IDENTICAL=39, N/A=21 | IDENTICAL=30 | IDENTICAL=24, N/A=6 | 0 |

The `stepfull` cells (one fresh-state forward against token-by-token decode
with a carried state) read IDENTICAL on both columns for the eight decode
lanes before and after. The resident sessions are additionally checked by the
A/B above (every token bytewise against the per-call arm and the fresh full
forward) and by two probes (`~/mojolearn-evidence/infer-speed-neural/
probe_session.py`, `probe_mamba1.py`): session-only decode, per-call prefill
then session decode, session prefill then per-call decode after `close`, the
full-causal and the window-8 block, all bytewise.

Byte LM on the AFTER build: `tools/byte_lm_gpu_logits_sweep.py` PASS, the
three per-state digests equal the CPU sweep's (`6db55997`, `30a89281`,
`b518e71e`), negative control differs; `tools/byte_lm_host_gate.py --steps
every:16` PASS, 25 compared. Both also PASS on BEFORE. Reports:
`pod_out/bytelm_{sweep,host_gate}_{before,after}.json`.

Files: `~/mojolearn-evidence/infer-speed-neural/pod_out/identity_{cuda,cpu}_
{before,after}.json`, `pod_out/logs/identity_diff_{cuda,cpu}.log`.

## Sabotage controls

Three controls, each seen to fail (`tools/infer_speed_neural_body.sh`
phase `sabotage`, logs under `pod_out/logs/sabotage_*` and
`identity_*sabotage*`):

1. **identity_break under `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`** on the
   cuda column, lanes `transformer`, `transformer-window`, `mamba1`, `mamba2`,
   `mamba3`, fixture `base`, diffed against the AFTER column: every `batch`
   and every `stepfull` cell read BATCH_MOVED (`summary (batch):
   BATCH_MOVED=5`, `summary (stepfull): BATCH_MOVED=5`), the train and infer
   cells IDENTICAL, so the stepfull comparison this lane leans on can fail.
2. **The GPU byte LM binding built under `-D MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE=1`**
   (the define added in `training/byte_lm_logits.mojo`, one output bit
   flipped after the arithmetic, never in a shipped binary), the host binding
   clean: `tools/byte_lm_gpu_logits_sweep.py` exited 1: `FAIL: 816/1680 comparisons equal`, 864 of the 1680 GPU-vs-CPU comparisons differed and the per-state digests no longer equal the CPU sweep's (`pod_out/sabotage_sweep.json`, `verdict: FAIL`).
3. **The host byte LM binding built under `-D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1`**
   (the gate's own DEVIATION 2612 build): `tools/byte_lm_host_gate.py
   --expect-mismatch` exited 0 only because a mismatch was seen: `PASS: 21/25 loss bytes equal (sabotage build, a mismatch was required)`, four loss bytes moved (`pod_out/sabotage_host_gate.json`).

## Commands

Pod, from a checkout at `/root/mojolearn` (the runner is `tools/trees_leg.sh`
with `TREES_LEG_STATE=$HOME/mojolearn-evidence/infer-speed-neural/pod`):

```
INFER_PHASE=setup    ROOT=/root/mojolearn sh tools/infer_speed_neural_body.sh
INFER_PHASE=identity ROOT=<tree> LABEL=<before|after> sh tools/infer_speed_neural_body.sh
INFER_PHASE=bytelm   ROOT=<tree> LABEL=<before|after> sh tools/infer_speed_neural_body.sh
INFER_PHASE=diff     ROOT=<tree> sh tools/infer_speed_neural_body.sh
INFER_PHASE=sabotage ROOT=<tree> sh tools/infer_speed_neural_body.sh
pixi run python tools/bench_neural_decode.py --resident-ab --kind transformer --dm 1024 --batch 8 --prefill 1024 --tokens 64 --rounds 5 --out ab.json
pixi run python tools/bench_neural_decode.py --resident-ab --kind mamba1 --dm 1024 --batch 8 --prefill 1024 --tokens 64 --rounds 5 --out ab.json
```

## Not done, and why

- Main's `hasattr` probe defect on the CPU route (fixed here, see "Merge
  with main") is also the shape of `_mamba_impl.py`'s `mamba3_forward_fresh`
  probe and any other bare `hasattr` on a binding; those were not audited
  in this lane.
- Mamba-2, Mamba-3 and Samba resident sessions. Mamba-3's state is ten
  pieces with a chunk buffer and a pending flag, Samba is a stack of blocks;
  both fit the same object shape and are the next step, not this one.
- The stateless byte LM call keeps its per-call context (DEVIATION 2520's
  design) and its weight uploads through `training/byte_lm.mojo`'s helpers,
  which are training code this lane does not edit.
- The per-call `TransformerBlock.step` and `Mamba1Block.step` are unchanged.
  A cache upload of the used prefix only would move no arithmetic but it
  changes which bytes of the caller's buffer are written back, which the
  identity records hash, so it was not built.
- Apple and AMD columns for the session paths and for the byte LM changes:
  owed at the next release record, as every column is.

## Merge with main, 2026-09-17

Main moved 21 commits past this lane's base (`e3213a59a` to `423f7fa28`).
Two of them (`9da5c4685`, `1b7dc811b`, `docs/TRANSFORMER_SESSION_REUSE.md`)
added a `TransformerSession` to `bindings/_mojolearn_transformer.mojo`, the
per-call route's retained per-model context and workspace, registered as
`transformer_session_{create,close,info,forward}`. The lane's decode session
had registered the same four names. Resolution: the lane's entry points are
now `transformer_decode_session_{create,open,step,forward,export_state,
load_state,info,close}` (definitions, registrations and the Python class);
main's names and semantics are untouched. The two coexist on one block: the
per-call `forward`/`step` go through main's retained context (weights and
cache reread every call), `decode_session` opens its own context and owns
the copies, and `_refuse_resident` applies to both per-call routes because
it lives in `_call_impl`. Mamba had no collision; `mamba1_session_*` stays.
`docs/NEURAL_METAL_DECODE.md` keeps both appended sections, this lane's
first. The merged tree's re-verification on a rented RTX 4090 is recorded
below under "Re-verification after the merge".

One defect found by that run and fixed on this branch (`4378a30fc`): main's
`TransformerBlock._call` probes `hasattr(ext, "transformer_session_forward")`
on every call, and on a CPU-only install the stand-in for the GPU binding
raises `ImportError` BY NAME from `__getattr__`, which `hasattr` does not
swallow, so every `transformer` and `samba` cell of the CPU identity column
REFUSED on the merged tree (first pod, EPYC 7642: `summary: IDENTICAL=18,
ONE-COLUMN=12`). `_transformer_impl._exports` treats `ImportError` as "not
exported", the way `_backend.py`'s own mode read-back does, and the two
decode-session constructors get the same guard. With it the CPU column reads
IDENTICAL on every cell (second pod).

## Re-verification after the merge (RTX 4090, driver 580.159.03, 2026-09-17)

Record: `bench/results/infer_speed_neural_2026-09-17/merged_reverify_2026-09-17.json`;
full outputs in `~/mojolearn-evidence/infer-speed-neural/pod3_out/infer_out/`.
Pod `2qlz5pciaofluz` (rented with `TREES_LEG_CUDA_VERSIONS=13.0`, $0.74/h,
15.1 minutes, reaped and HTTP 404 verified), merged tree `7ccefc445` shipped
by the runner plus the two Python files of `4378a30fc` pushed by hash, all
eleven bindings built there from source. A first pod (`0m4jd39wsn5izm`,
13.2 minutes, same price) came up with driver 570.195.03, which Mojo's GPU
runtime refuses; its GPU column is no evidence and it was reaped.

BEFORE is the lane's own BEFORE column (`e3213a59a`, RTX 4090 driver
580.126.20 and EPYC 7532); MERGED is `identity_{cuda,cpu}_merged.json`, diffed
with `tools/identity_break.py --diff`:

| column | train | infer/model | batch | stepfull | MOVED or DIVERGENT |
|---|---|---|---|---|---|
| cuda (12 lanes x 3 fixtures) | IDENTICAL=36 | IDENTICAL=51, N/A=21 | IDENTICAL=36 | IDENTICAL=24, N/A=12 | 0 |
| cpu (10 lanes x 3, EPYC 7282 against EPYC 7532) | IDENTICAL=30 | IDENTICAL=39, N/A=21 | IDENTICAL=30 | IDENTICAL=24, N/A=6 | 0 |

Byte LM on the merged build: `tools/byte_lm_gpu_logits_sweep.py` PASS
(768 resident logits, 96 stateless, 48 loss bit patterns, 768 next bytes
compared), the three per-state digests equal the CPU sweep's (`6db55997`,
`30a89281`, `b518e71e`), negative control differs; `tools/byte_lm_host_gate.py
--steps every:16` PASS, 25 of 25 equal.

Resident A/B on the merged build (`--dm 1024 --batch 1 --prefill 1024
--tokens 64 --rounds 3`), every output and state piece bytewise equal to the
fresh full forward and across arms and rounds:

| model | per-call ms | resident ms | paired ratio | lane record |
|---|---|---|---|---|
| Transformer B1 | 7.273 | 0.850 | 8.56 | 11.54 (per-call 9.440) |
| Mamba-1 B1 | 33.762 | 0.944 | 35.78 | 33.61 |

The transformer per-call arm is faster than the lane's record because it
now runs through main's retained `TransformerSession` (one context and
workspace per model, weights and cache reread per call); the resident arm is
unchanged within spread, so the ratio fell from 11.5x to 8.6x. Mamba has no
retained per-model context on main and reads as before.

Coexistence, shown not assumed: main's own checks on the merged binding,
`tools/transformer_session_check.py` groups reuse (75 arrays), refusals (15),
lifetime (9), budget (2) and `tools/transformer_session_surface_check.py`
groups state (22), serialization (4), threads (3, worker init), all exit 0
against `transformer_forward`; and a probe on ONE block
(`pod3_out/infer_out/logs/coexist_probe.log`, PASS): per-call prefill and
decode through the retained context, then `decode_session` opened on the
same block while the retained context is alive (its workspace count keeps
growing across the probe, 1 to 6), per-call `step` and `forward` refused
by name while the state is resident, sixteen session tokens bytewise the
per-call tokens, the synced cache bytewise the per-call cache, a second
session on the same block after close, and every tail token bytewise a
fresh full forward. The binding exports both name sets.

Not re-run here: the three sabotage controls (unchanged code paths, seen to
fail on the lane's first pod), and the B=8 A/B rows.

## False claims found in owned docs

None edited away. `docs/NEURAL_METAL_DECODE.md` says a resident decode API
"is a separate improvement"; it is now this lane's, recorded in the CUDA
section added there.
