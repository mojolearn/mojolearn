# SEQ_SPEED: how fast our FAST sequence-model path is against native NVIDIA

**Status 2026-08-25: NOTHING HERE HAS BEEN RUN.** Both files this document
describes were written in one session by an agent that was forbidden to
execute anything. There is no measurement in this repository for either
lane. Every number that appears in this file later must carry the box, the
date and the driver commit; until then this is a description of an
instrument, not a result.

Two files:

* `bench/speed/seq_speed_main.mojo` -- our arm. One lane per process, the
  FAST build (the default, **not** `-D MOJOLEARN_NUMERIC_IDENTICAL=1`).
* `tools/speed_torch_seq.py` -- the opponent arm. torch on CUDA, and
  `mamba-ssm`'s fused `selective_scan_cuda` where it installs.

They share a line format so their outputs concatenate into one table, and
they share a shape table: the Python side **parses** the `seq_shape_*`
if-ladders out of the Mojo driver rather than keeping a second copy, the way
`tools/vendor_gemm_price.py` parses `bench/gemm_shapes.mojo`.

---

## 1. What is measured

One **decoder block** or one **Mamba-1 block**, or one submodule of either.

| lane | our entry | file:line |
| --- | --- | --- |
| `transformer` | `llama_decoder_layer_forward` | `transformer/ported/transformers/models/llama/modeling_llama.mojo:3024` |
| `attention` | `llama_attention_forward` | same file `:2494` |
| `mlp` | `llama_mlp_forward` | same file `:2695` |
| `rmsnorm` | `llama_rms_norm` | same file `:1127` |
| `mamba` | `mamba_block_forward` | `mamba/ported/transformers/models/mamba/modeling_mamba.mojo:1269` |
| `selective_scan` | `selective_scan_fn` | `mamba/ported/mamba_ssm/ops/selective_scan_interface.mojo:465` |

**This is not a model and it is not a token.** A served Llama-3-8B is
thirty-two of the `transformer` lane's block plus an embedding, a head, a
sampler and a scheduler. Nothing here measures any of that, and no
tokens-per-second figure may be derived from these milliseconds.

### The shape ladder

Eleven rows, swept whole. No row may be dropped, deferred or retuned because
of what it says about us (`[[no-dataset-cherry-picking]]`).

| row | name | B | L | prior ctx | shape |
| --- | --- | --- | --- | --- | --- |
| 0 | `lane.b2_l4_d32_kv2` | 2 | 4 | 0 | the transformer correctness lane's config |
| 1 | `llama8b.prefill.t1` | 1 | 1 | 0 | d_model 4096, 32 heads, 8 kv, head_dim 128, inter 14336 |
| 2 | `llama8b.prefill.t8` | 1 | 8 | 0 | same |
| 3 | `llama8b.prefill.t128` | 1 | 128 | 0 | same |
| 4 | `llama8b.prefill.t512` | 1 | 512 | 0 | same |
| 5 | `llama8b.decode.t1.ctx512` | 1 | 1 | 512 | same, one query against a 512-token cache |
| 6 | `lane.b2_l4_d8` | 2 | 4 | 0 | the mamba correctness lane's config |
| 7 | `mamba130m.prefill.t1` | 1 | 1 | 0 | d_model 768 (d_inner 1536, dt_rank 48) |
| 8 | `mamba130m.prefill.t8` | 1 | 8 | 0 | same |
| 9 | `mamba130m.prefill.t128` | 1 | 128 | 0 | same |
| 10 | `mamba130m.prefill.t512` | 1 | 512 | 0 | same |

Token counts 1, 8, 128 and 512 are there because the ratio moves enormously
with arithmetic intensity: on one H100 a pinned GEMM cost 1.02x at two rows
and 2.33x at 512. Row 5 is the regime every served token after the first is
in, and it is the only row where attention is bandwidth bound rather than
compute bound.

**There is no mamba decode row and that is a claim, not an omission.**
Mamba's recurrence is O(1) per token whatever the history, so the cost of a
decode step is the cost of row 7 (L = 1). A carried state differs from a
fresh one in its conv window's contents and not in its cost. Recorded as
DEVIATION 1854. If row 7 and a warm-state L = 1 call ever measure
differently, that sentence is wrong and this file must say so.

`MOJOLEARN_SPEED_SIZE=smoke` runs rows 0 and 6 only. Every line carries
`size=`, so a smoke line cannot be mistaken for a measurement.

### What one round is

One call of the lane's entry with a `ctx.synchronize()` (ours) or a
`torch.cuda.synchronize()` (theirs) inside the clock. Setup, allocation,
weight generation, host-to-device upload and rotary-table construction are
**outside** it. An unsynchronized timing measures the enqueue rate and
nothing else.

Anything the entry mutates and also reads is restored **before** each round
and outside the clock: the Llama KV cache's contents and its used length
(DEVIATION 1853), and the Mamba conv window and SSM state. A round that
continued the previous round's state would be a different call each time.

One untimed warm-up on our side (printed as `FSPEED-WARMUP`, never in the
table). **Five** on the torch side, because lazy module init and cuBLAS
handle creation land on the first call and the autotuner can land on the
second. `MOJOLEARN_SPEED_ROUNDS` timed rounds after that, default 10.

---

## 2. The opponents

### Llama

| arm | what it is |
| --- | --- |
| `torch-gpu-fp32` | eager torch: matmul, additive mask, explicit max / exp / sum / divide, matmul. `allow_tf32 = False`. **The apples-to-apples arm.** |
| `torch-gpu-tf32` | the same code with `allow_tf32 = True`. Ten explicit mantissa bits instead of twenty-three. Not the same arithmetic. |
| `torch-gpu-sdpa-math-fp32` | `F.scaled_dot_product_attention` forced to the `MATH` backend |
| `torch-gpu-sdpa-efficient-fp32` | forced to `EFFICIENT_ATTENTION` |
| `torch-gpu-flash-fp32` | forced to `FLASH_ATTENTION`. **Expected to refuse:** FlashAttention on CUDA does not accept FP32. |
| `torch-gpu-flash-bf16` | the fused kernel at bfloat16, which is what anybody actually deploys. A different precision and a different algorithm, named as such. |

**TF32 is the trap that owns this whole slice.** On Ampere and later torch
may satisfy an FP32 matmul with TF32 tensor cores. Measured in this
repository on an H100, the same GEMM ran 44.4 TFLOP/s with `allow_tf32=False`
and 207.5 TFLOP/s with it on, about 5x. Every attention and MLP arm here is
GEMM dominated. A single unlabeled torch number would therefore sit
somewhere in a 5x band for reasons that have nothing to do with either
implementation, so both settings are timed, both are reported, and the
setting is in the arm name. `tools/speed_torch_seq.py` sets the switch
explicitly in both directions rather than trusting a default, because
torch's default has moved between releases.

**The backend is named, never chosen.** An unlabeled
`scaled_dot_product_attention` number is a number for whichever kernel that
particular wheel happened to prefer on that particular shape. Our port is
eager by contract (contract section 6 excludes FlashAttention, SDPA, paged
attention and chunked prefill, because an online softmax's rescale count is
the KV tile count and that is an execution-plan quantity). So the eager arm
is the comparison and the fused arms are the context.

### Mamba

| arm | what it is |
| --- | --- |
| `mamba-ssm-cuda` | `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`, which dispatches to the fused `selective_scan_cuda` extension. **The real opponent.** |
| `torch-ref-scan-gpu` | the pure-PyTorch sequential reference scan, `selective_scan_ref` copied verbatim inside `mamba/corpus/gen_corpus.py`, run on the GPU. A Python loop over the sequence. |
| `torch-ref-scan-gpu-tf32` | the same with TF32 on, which reaches the in_proj / x_proj / out_proj GEMMs and not the scan |

**Say which one ran.** If `mamba-ssm` does not install on the pod, the
`mamba-ssm-cuda` arm emits `FSPEED-REFUSED` with the import error and the
only opponent left is a sequential PyTorch scan, **which is not what anyone
deploys**. A ratio against `torch-ref-scan-gpu` alone is not an answer to
"how fast are we against the native NVIDIA competitor"; it is an answer to
"how fast are we against the slowest correct thing".

The `mamba-ssm-cuda` **block** arm folds the `z` gate into the scan
(`z=gate`), which is what a deployment does. Our port refuses to
(`selective_scan_fn` raises on `z`, `delta_bias` and `delta_softplus`:
DEVIATION 723, seams S12 and S14 are the block's recorded stages). So that
arm does the same math in fewer kernels, and the difference is part of what
is being measured. The `selective_scan` lane calls both sides identically
(no `z`, no `delta_bias`, `delta` already softplused), which is the clean
comparison.

---

## 3. The weights are the same on both sides

Both sides build every tensor from the same hashed generator:

```
value = f32(lo + (hi - lo) * top24(splitmix64(key + i)) * 2^-24)
key   = splitmix64(seed ^ (tensor_id << 32))
seed  = SEQ_SEED_BASE + 0x1000 * row
```

which is `mojolearn.mamba.corpus.hash.v1` and, with different ids and
ranges, `mojolearn.transformer.fixture.hash.v1`. The Mojo side calls the
lanes' own `corpus_tensor` / `fixture_tensor`; the Python side calls the
mamba corpus's own `splitmix64_scalar` / `splitmix64_array`. Neither
re-spells the hash.

That is construction, not proof, so **every tensor also prints a witness**:

```
FSPEED-WEIGHTS lane=<lane> shape=<tag> tensor=<name> n=<count> hash=<16 hex>
```

FNV-1a64 over the tensor's length and then over a fixed strided sample of its
float32 bytes, stride `max(1, n // 4096)`. Both sides spell the same rule.
**If a witness disagrees, the run is void** and no ratio from it means
anything. The sample rather than the whole tensor is DEVIATION 1850: the
Llama-8B `down_proj` alone is 58.7 million floats and FNV-1a64 is
byte-at-a-time by construction, so a whole-tensor witness would cost the
Python side more than the benchmark does. The stride is what makes a
transposition or a permutation visible; a prefix witness would agree with
anything that shares a first page.

Ranges: the transformer weights use `transformer/corpus/gen_corpus.py`'s
`BASE_RANGES` and `ranges_for` and **not**
`transformer_fixture.mojo::fixture_weights`'s, because the corpus scales
`o_proj` and `down_proj` by `fan_in_scale` and the fixture pins them at
values tuned for `d_model = 32`. The seed base is this lane's own
(DEVIATION 1851), so no row here can be mistaken for a corpus case: these are
corpus **shapes** with speed-lane **bits**, and no reference value recorded
for `base_b2_l4_d32_kv2` or `base_b2_l4_d8` applies to them.

**The big rows have a degenerate softmax.** At `d_model = 4096` with
`q_proj` uniform on `[-0.5, 0.5]` the pre-softmax scores run to a few
hundred, so after the max subtraction most weights underflow to exactly zero
and the row is nearly one-hot. Every cell still costs the same, so the timing
is unaffected, and both sides underflow identically, so `FSPEED-AGREE` is
unaffected. No accuracy claim may be read out of these rows.

### FSPEED-AGREE

A speed number for a block that computes something else is worthless, so the
two sides' outputs are compared once per shape:

```
FSPEED-AGREE lane=<lane> max_abs_diff=<float> max_rel_diff=<float> n=<count>
```

Mechanism: the Mojo driver writes its last round's output as raw
little-endian float32 to `$MOJOLEARN_SPEED_DUMP_DIR/seq.<lane>.<tag>.f32.bin`,
and `tools/speed_torch_seq.py --dump-dir` reads it and compares against the
eager FP32 arm. **A report line and not a gate.** Two FP32 implementations of
the same block will not agree bitwise and are not expected to; what this
catches is a harness that timed the wrong thing, which is the failure that
makes a whole benchmark worthless without ever looking wrong. If the dump
directory is unset, both sides say so with `FSPEED-REFUSED lane=... arm=agree`
and the run is weaker for it.

---

## 4. What is NOT comparable, and why

This section is the one to read before quoting a ratio.

**(a) Our block entry validates every weight on the host on every call.
Torch does not.** `llama_refuse_bad_inputs` (`modeling_llama.mojo:2121`) and
`mamba_refuse_bad_inputs` (`modeling_mamba.mojo:1053`) copy every parameter
back to the host and scan it for NaN and infinity, per call, inside the entry
and therefore inside the clock. At the Llama-8B rows that is roughly 872 MB
device to host per call plus a host loop over 218 million floats. **At those
rows the `transformer` lane number is a PCIe measurement with a GEMM
attached.** It is an honest report of what our block entry costs today and it
is not a report of what our kernels cost. Both drivers time the refusal
separately and print it as an `FSPEED-NOTE`, so a reader can subtract it
rather than guess; and the `attention`, `mlp`, `rmsnorm` and
`selective_scan` lanes call submodules that do not refuse, which is why those
lanes exist.

**(b) Our GEMM is the identity lane's kernel in both modes.**
`modeling_llama.mojo:277` imports `gemm.mojo_only.gemm_identical.identical_gemm`
and calls it for every projection. FAST does not swap in MAX's
`linalg.matmul` or anything else; it compiles the same pinned balanced-tree
kernel with the pins removed. So the `attention` and `mlp` ratios are
substantially the ratio between that kernel and cuBLAS, which
`bench/gemm_price_main.mojo` and `tools/vendor_gemm_price.py` already price
directly. Do not read a transformer-shaped number here as new information
about the GEMM.

**(c) Our attention materializes seven score-sized buffers and synchronizes
per head.** `eager_attention_forward` records `attn.scores`, `attn.masked`,
`attn.max`, `attn.exp`, `attn.denom`, `attn.weights` and `attn.ctx` as
separate device buffers, and calls `ctx.synchronize()` inside a
`B * n_heads` loop (`modeling_llama.mojo:2254` and twice more per head). At
32 heads that is at least 32 round trips per call before any arithmetic is
counted. Contract section 6 argues this is the point: a lane whose whole
instrument is the per-stage card may not begin by fusing the stages away.
It is still a cost, and torch's eager arm pays none of it.

**(d) Their module is one library call where ours is thirty.** Torch's
`F.linear` is a cuBLAS call with a tuned tile schedule and a fused epilogue.
Ours is a launch of a kernel that computes a pinned fold order. Comparing
them tells you the cost of the port plus the cost of the design decision;
it does not isolate either.

**(e) The fused arms compute a different function.**
`torch-gpu-sdpa-efficient-fp32` and the flash arms use an online softmax
whose rescale count is a function of the KV tile count. That is a different
arithmetic, not a faster spelling of the same one. `torch-gpu-flash-bf16` is
additionally a different precision. Both are reported because both are what
a user would run; neither is a control for our eager port.

**(f) `torch-gpu-tf32` is not FP32.** Ten explicit mantissa bits. It is here
because it is what a user gets by default on many wheels, not because it is
comparable to a strict FP32 arm.

**(g) On ROCm the arm names still say `cuda`,** because torch's device does.
The same `allow_tf32` switch reaches CDNA3's XF32 matrix mode, so the
`tf32` arm is meaningful there too and is labeled the same way.

**(h) One box, one shape, no serving stack.** No batching across requests,
no paged KV, no CUDA graphs, no compile. `torch.compile` is deliberately not
used: it would fuse across the block and produce a number for a program
neither side wrote.

---

## 5. The line format

Both sides emit exactly these.

```
FSPEED-HEADER family=seq lane=<lane> arm=<arm> mode=<FAST|IDENTICAL> device=<string> rounds=<n> size=<shipped|smoke>
FSPEED-WARMUP lane=<lane> arm=<arm> shape=<tag> ms=<float>
FSPEED lane=<lane> arm=<arm> shape=<tag> round=<i> ms=<float> hash=<16 hex digits or ->
FSPEED-NOTE lane=<lane> arm=<arm> <free text>
FSPEED-REFUSED lane=<lane> arm=<arm> reason=<one line>
FSPEED-WEIGHTS lane=<lane> shape=<tag> tensor=<name> n=<count> hash=<16 hex>
FSPEED-AGREE lane=<lane> max_abs_diff=<float> max_rel_diff=<float> n=<count>
FSPEED-DONE lane=<lane> ...
```

`mode` is read from the comptime numeric-mode constant (`_mode_name()`),
never from the environment and never from the flag that was passed. Three
mislabeled measurements were caught by that witness on 2026-08-23. The torch
side prints `mode=FAST` because a torch arm has no such thing as our
IDENTICAL mode and pretending otherwise would put a wrong label in a column
somebody sorts on.

`hash` is FNV-1a64 over the output bytes through
`core/identity_trace.mojo::fnv1a64_bytes`. Under FAST it **may** move between
rounds; when it does, an `FSPEED-NOTE ... hash moved across rounds:` is
printed and the run continues. That is a report of a non-deterministic arm,
not a failure. The torch side prints `hash=-` when the output exceeds 256 KB,
because FNV-1a64 is sequential by construction and a byte-at-a-time hash of
the 8.4 MB t512 output would cost seconds per round in Python. It prints an
`FSPEED-NOTE ... output witness shape=... hash=...` for every arm regardless,
using the same strided rule as the weight witness, so the big rows still have
a fingerprint. Without it, "the arm produced nothing detectable" would look
exactly like "the arm produced the right thing".

---

## 6. Running it

Our arm, one lane per process:

```
export MOJOLEARN_SPEED_DEVICE=H100-SXM        # there is no device-name
                                              # accessor in this repository;
                                              # torch's get_device_name(0) on
                                              # the other side is the authority
export MOJOLEARN_SPEED_DUMP_DIR=/tmp/seqdump
mkdir -p $MOJOLEARN_SPEED_DUMP_DIR

for lane in transformer attention mlp rmsnorm; do
  MOJOLEARN_SPEED_LANE=$lane pixi run mojo run -I . bench/speed/seq_speed_main.mojo
done
for lane in mamba selective_scan; do
  MOJOLEARN_SPEED_LANE=$lane pixi run mojo run -I . bench/speed/seq_speed_main.mojo
done
```

Theirs:

```
for lane in transformer attention mlp rmsnorm mamba selective_scan; do
  python3 tools/speed_torch_seq.py --lane $lane --dump-dir /tmp/seqdump
done
```

**Prove the build first.** `MOJOLEARN_SPEED_SIZE=smoke` on both sides runs
rows 0 and 6 only and takes seconds. The mamba lane's FAST arm has, per the
lane notes, never been built on some vendors at all, so the first CUDA
invocation of `MOJOLEARN_SPEED_LANE=mamba` should be treated as a build and
budgeted as one.

### pixi tasks

Add to `[feature.bench.tasks]` in `pixi.toml` (the Python arm needs torch,
which lives in `[feature.skgpu.dependencies]`; on a RunPod CUDA image the
system python already has torch and running `tools/speed_torch_seq.py` with
it directly is the shorter path):

```toml
seq-speed-mojo = "mojo run -I . bench/speed/seq_speed_main.mojo"
seq-speed-torch = "python tools/speed_torch_seq.py --lane $MOJOLEARN_SPEED_LANE --dump-dir $MOJOLEARN_SPEED_DUMP_DIR"
```

### Python packages the pod needs

`numpy`, `torch` (CUDA build), `einops` (the corpus's verbatim
`selective_scan_ref` uses `rearrange` and `repeat`). Optional and worth the
attempt: `mamba-ssm`, and `causal-conv1d` beside it. Nothing else.

### Is `mamba-ssm` installable inside a one-hour lease?

Realistically: **only from a prebuilt wheel.** `mamba-ssm` ships CUDA
extension sources and `pip install mamba-ssm` will attempt an `nvcc` build of
`selective_scan_cuda` across several dtype and dstate instantiations. On a
stock RunPod CUDA image that build is long enough to be a serious fraction of
an hour, and it fails outright if the image's CUDA toolkit does not match the
torch build or if there is no `nvcc` at all. The project publishes release
wheels keyed to (torch version, CUDA version, cxx11abi, python version), and
`pip install mamba-ssm --no-build-isolation` picks one up when the tuple
matches exactly.

Therefore, in order:

1. Try `pip install --no-build-isolation causal-conv1d mamba-ssm` and give it
   a hard timeout. If a wheel matches, this is under a minute.
2. If it starts compiling, **kill it**. `MAX_JOBS=<n>` helps but the wheel
   path is the only one that fits the lease.
3. If no wheel matches, run with the `torch-ref-scan-gpu` fallback and say in
   the write-up that the mamba comparison had no native opponent in it.

The honest engineering answer is to bake `mamba-ssm` into a pod image once,
outside a lease, and reuse it. Building it inside the lease you are trying to
measure in is how a rental gets spent on `nvcc`.

### Is flash attention installable?

**It is already there.** `torch.nn.functional.scaled_dot_product_attention`
ships a FlashAttention-2 backend inside torch itself; nothing needs
installing and `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`
selects it. The standalone `flash-attn` package is a separate source build
with the same `nvcc` problem as `mamba-ssm` and this harness does **not**
need it. The only caveat is the one already stated: the flash backend does
not accept FP32, so `torch-gpu-flash-fp32` will refuse and
`torch-gpu-flash-bf16` is the arm that runs.

---

## 7. Known weak points of this harness

Written down now so nobody has to rediscover them.

1. **Nothing has been compiled.** The Mojo driver's most likely failure is a
   signature mismatch on `llama_attention_forward` or `selective_scan_fn`,
   both of which take long positional argument lists.
2. **`MOJOLEARN_SPEED_DEVICE` is passed in.** No file in this repository
   reads a device name from `DeviceContext`, so this driver does not either.
   If MAX exposes one, swapping it in is a two-line change and makes the
   header self-describing.
3. **The KV-cache restore between rounds (DEVIATION 1853) is a guess about
   repack semantics.** The cache is snapshotted as two device buffers and
   copied back, and `kv.s` is reset, which should be exactly equivalent to a
   fresh call. If the round-2 hash moves at row 5 and not at the prefill
   rows, suspect this before suspecting FAST non-determinism.
4. **The submodule lanes read stage buffers filled by one untimed whole-block
   setup call.** A submodule timed on the `_zeros` fill would be timing
   denormals and a softmax over a constant, which is different arithmetic on
   some hardware. The setup call is what makes the inputs realistic; it is
   also one more thing that can be wrong.
5. **`LlamaEager` in the Python arm is a second eager forward** (DEVIATION
   1852), written because `transformer/corpus/gen_corpus.py::block_forward`
   rebuilds the rotary tables in numpy on the host on every call and that
   cannot go inside a timed region. `--crosscheck` compares the two in
   float64 at row 0 and prints the difference. If that cross-check is
   skipped or refuses, the transformer opponent is unvalidated.
6. **The corpus generators set `torch.use_deterministic_algorithms(True)` at
   import time.** `tools/speed_torch_seq.py` turns it back off out loud
   (DEVIATION 1856); left on, deterministic kernels are slower and a CUDA
   matmul raises without `CUBLAS_WORKSPACE_CONFIG`. If a future edit imports
   a corpus and forgets this, every torch number silently becomes a
   deterministic-kernel number.
7. **Host memory at the Llama-8B rows.** Roughly 872 MB of weights are
   generated on the host on each side before upload. The Python side
   generates in chunks for that reason; the Mojo side builds nine
   `List[Float32]` and hands them to `LlamaDeviceWeights`, which copies them
   again through a host buffer. Peak host residency at row 4 is the number to
   watch on a small-RAM pod.
