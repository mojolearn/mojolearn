# Is NCCL's all-reduce bitwise reproducible?

Measured 2026-09-12, RunPod pod `j3uf4hx92kqmwv`, 4x NVIDIA A40. The question
was asked because cross-device bitwise identity for a multi-GPU trainer stands
or falls on the answer, and the usual wisdom -- "NCCL switches between ring and
tree by size and topology, so the reduction order varies and the result is not
reproducible" -- had never been measured here.

## Verdict

**On this stack NCCL's all-reduce IS bitwise reproducible for a fixed
configuration, and is NOT bit-stable across a change in one.** Both halves
matter and neither one is the received wisdom:

- Repetition and process restart are NOT a source of variation. 168
  configurations, each repeated 15 to 25 times inside one process, produced
  exactly **one** distinct result hash each -- zero configurations varied. All
  84 config/dtype/size cells were byte-identical across an independent process
  restart with the same inputs.
- The **configuration** is a source of variation, and NCCL picks most of it for
  you. At 4 ranks, forced `NCCL_ALGO=Tree` and forced `NCCL_ALGO=Ring` differ
  bit-for-bit at **every** size and dtype measured. `NCCL_PROTO` changes the
  Ring result. The channel count changes it again. And which of those NCCL
  chooses depends on message size and topology, so the same code with a
  different bucket size, a different GPU count or a different machine reduces
  to different bits.

So a deterministic all-reduce of our own is NOT needed to make repeated runs
reproducible -- they already are. It is needed only for identity **across**
device counts and topologies, which no NCCL setting can give, because reducing
N buffers in a ring and in a tree is genuinely different arithmetic.

The cheap path exists and is worth taking: pinning `NCCL_ALGO`, `NCCL_PROTO`,
`NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` and the rank-to-device mapping makes
the reduction bitwise repeatable at zero cost on a fixed rank count. The
expensive path (a fixed-order reduction, which is also identical across rank
counts) costs 7-20% at gradient-bucket sizes and about 2x at kilobyte messages.

## The trap, and how it was avoided

A buffer of similar-magnitude values sums to the same float in every order, so
a probe on benign data reports "deterministic" whatever the collective did.
Inputs here are `mantissa * 10^e` with `mantissa` uniform in [-1, 1] and `e`
uniform over **-12 .. +12**, built on the CPU from a seeded generator so the
bytes are identical on every restart and every rank count. The sum is dominated
by cancellation between terms 24 decades apart, which is exactly the regime
where order changes the result.

That is not asserted, it is witnessed. `--mode oracle` gathers the very buffers
the collective is handed onto one device and sums them there in four orders --
rank order, reverse, a rotation, and a pairwise tree. At 4 ranks **all four
orders give four different hashes, at every size and dtype**:

| 4 ranks, float32, 64 MB | hash |
|---|---|
| forward (0,1,2,3) | `7be814f7eec6b279515416949c597b4e` |
| reverse (3,2,1,0) | `d5566f22a13631126b46a7a43172f853` |
| rotate (1,2,3,0) | `df670a050de1498e6796d891e250beb7` |
| pairwise tree | `4253eee6ffa77fa327ac79255e0941f9` |

At 2 ranks the same oracle reports a single hash for all four orders, and the
summarizer flags that as "BENIGN -- verdicts void". That is the correct
reading: `a + b == b + a`, so with two ranks there is only one order and
nothing to be non-deterministic about. **Every 2-rank cell below is therefore
arithmetic, not evidence about NCCL**, and the verdict rests on the 4-rank
cells alone.

## What varies and what does not (4 ranks)

Same input bytes throughout; the only difference between rows is the
configuration. float32 hashes, one row per distinct result:

64 MB per rank (`n = 16777216`), 18 configurations, 4 distinct results:

| result hash | configurations |
|---|---|
| `b05c421e0d9eb287f06a2c468877c30c` | RingSimple, **NCCL's own choice**, 1 channel, 2 channels |
| `41c37aca6180136382ba00a52919f5af` | RingLL |
| `fc0eaf5b930a9c47dd2ab7632a879009` | RingLL128 |
| `3666ac3ee611469b288fec6f31d4bc91` | TreeSimple, TreeLL, TreeLL128 |

1 MB per rank (`n = 262144`), 18 configurations, 5 distinct results:

| result hash | configurations |
|---|---|
| `722d602fced81bb0a4b7ee94884166c2` | RingSimple, **NCCL's own choice**, 2 channels |
| `240e1e8dd420204cfd02db1b9e50b95c` | RingLL |
| `24ffdcdfa17a03f8eeb61141f2b43a3d` | RingLL128 |
| `d70f1e644ce529b651a2341dc0bfdd64` | TreeSimple, TreeLL, TreeLL128 |
| `38ba3e8a60b16c6d6f45917d582b411d` | 1 channel |

1 KB per rank (`n = 256`), 18 configurations, 3 distinct results:

| result hash | configurations |
|---|---|
| `516e18249a6d3dab1f7e866087a1c963` | RingLL, **NCCL's own choice**, 1 channel, 2 channels |
| `86797e4dbff705b93487f5cfaadaec79` | RingSimple, RingLL128 |
| `d8f588694ab73a9d192c348944c90a1b` | TreeSimple, TreeLL, TreeLL128 |

Read across those three tables:

- **Ring and Tree never agree.** Forcing the algorithm alone changes the bits at
  every size and in both dtypes. That is the single fact that proves the
  property depends on a choice NCCL makes for you.
- **NCCL's own choice moves with message size.** Unforced, it matches RingLL at
  1 KB and RingSimple at 1 MB and 64 MB -- it switched protocol as the message
  grew, and the bits switched with it. A trainer that changes its bucket size
  changes its gradients.
- **Protocol matters for Ring, not for Tree.** Simple, LL and LL128 give three
  different Ring results at 1 MB and 64 MB; all three Tree results coincide at
  every size measured.
- **Channel count matters.** One channel and two channels differ at 1 MB
  (`38ba3e8a...` vs `722d602f...`) and agree at 1 KB and 64 MB. The channel
  count is derived from the topology, so the same code on a differently wired
  box is a different reduction.
- bfloat16 behaves the same way: at 1 KB every Ring config gives
  `eff66263363d19b39efcebcf0af021e5` and every Tree config gives
  `1dd10490607a0ed639e815d2668c3f04`.

At 2 ranks all 8 configurations (Ring, Tree, peer-to-peer on, peer-to-peer off,
both restarts) collapse to one hash per size and dtype, e.g.
`0db7ca6d6194766d3cfda5cda538c70b` for float32 at 64 MB. As noted above that is
arithmetic, not a determinism result.

## The price of determinism

`--mode timing`: NCCL's all-reduce against a fixed-order all-reduce -- gather
every rank's buffer to rank 0 over point-to-point (overlapped `irecv`/`isend`,
so the gather is not artificially serialized), sum in rank order on one device,
broadcast. Median of 15 timed passes after 3 warm-ups, both arms on the same
buffers, run one at a time with nothing else on the GPUs.

4 ranks, float32:

| bytes per rank | NCCL (ms) | fixed order (ms) | ratio |
|---|---|---|---|
| 1 KB | 0.485 | 0.952 | 1.97x |
| 64 KB | 0.317 | 0.575 | 1.81x |
| 1 MB | 0.417 | 0.607 | 1.45x |
| 8 MB | 2.452 | 2.757 | 1.12x |
| 64 MB | 18.963 | 20.237 | 1.07x |
| 256 MB | 74.103 | 80.268 | 1.08x |

4 ranks, bfloat16: 2.12x (1 KB), 2.04x (64 KB), 1.46x (1 MB), 1.20x (8 MB),
1.12x (64 MB), 1.12x (256 MB), with NCCL at 35.943 ms for 256 MB.

2 ranks, float32: 1.50x (1 KB) falling to 1.17x (256 MB); bfloat16 1.60x to
1.18x.

The fixed-order result never equals NCCL's bits at 4 ranks (a different order,
as expected) and always equals them at 2 ranks (only one order exists).

The shape of the cost is latency, not bandwidth: at and above 8 MB -- the size
range gradient buckets actually live in -- determinism costs 7-20%, and the
ratio keeps falling as the message grows. Below 1 MB it costs about 2x, because
the extra hop dominates when there is nothing to transfer.

## Exact stack

| | |
|---|---|
| box | RunPod `j3uf4hx92kqmwv`, 4x NVIDIA A40 48 GB, PCIe, **no NVLink** (topology SYS/NODE) |
| driver | 570.195.03 |
| container | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| torch | 2.4.1+cu124, CUDA 12.4, Python 3.11.10 |
| NCCL | **2.20.5+cuda12.4** (torch-bundled, confirmed from the runtime banner; a system `libnccl.so.2.21.5` is present on the image and is not what ran) |
| transport | internal implementation, no `libnccl-net.so` plugin; bootstrap over eth0 |
| collective | `torch.distributed.all_reduce`, `ReduceOp.SUM`, one process per GPU under `torchrun` |

This is a claim about that stack. It is not a claim about NCCL in general, and
specifically not about NVLink/NVLS paths, CollNet, multi-node, or a version
other than 2.20.5.

## Conditions and limits

- **Peer-to-peer at >= 3 ranks hangs on this host.** With default settings the
  first all-reduce never returns at 3 and 4 ranks; it completes at 2. This is
  the usual ACS-enabled PCIe host, not an NCCL bug, and the fix is
  `NCCL_P2P_DISABLE=1`, which every >= 3 rank case here carries (shared-memory
  transport). 4 ranks over peer-to-peer was not measured and is not claimed.
  The 2-rank cases were run both ways and give the same bits, but with one
  summation order that is weak evidence about transport, not strong.
- One node, one GPU model, one NCCL version, one rank-to-device mapping.
- Buffer shapes are fixed per configuration; a trainer whose bucket sizes shift
  between steps is in the "configuration changed" case above, not the
  "repetition" case.
- Reduction is SUM only. Averaging, gradient scaling and the optimizer sit
  outside this measurement.

## What this means for cross-device bitwise identity

1. Repeating a run on the same box with the same code and the same rank count
   already gives bit-identical reductions. Nothing needs building for that.
2. Identity across a *configuration* change on the same rank count is cheap:
   pin `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`
   and the rank order, and the bits stop moving. Worth doing regardless,
   because otherwise a message-size change silently moves them.
3. Identity across **rank counts or topologies** cannot come from configuring
   NCCL, at any price, because the arithmetic itself differs. That needs a
   fixed-order reduction, and the price of one is now a number rather than a
   guess: 7-20% at >= 8 MB, about 2x at kilobyte messages.

## Reproducing

```
tools/nccl_leg.sh rent --gpu "NVIDIA A40" --minutes 100   # TREES_LEG_GPU_COUNT=4
tools/nccl_leg.sh ssh 'cd /root/mojolearn && bash tools/nccl_determinism_leg.sh /root/out/nccl'
python3 tools/nccl_determinism_summarize.py bench/results/nccl_determinism_2026-09-12/hashes.jsonl
```

- `hashes.jsonl` -- every hash, one row per configuration/dtype/size (204 rows)
- `summary.txt` -- the summarizer's output over that file
- `stack.txt` -- versions, topology, `nvidia-smi`
- `logs/` -- one log per configuration, plus `nccl_debug_info.log` (NCCL's own
  tuning table and the tree shapes it built)
