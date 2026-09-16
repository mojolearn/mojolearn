# LANE STATUS: lane/neural-shape-shrink (2026-09-16)

Cut the Metal cost of the neural identity lanes by shrinking MODEL DEPTH AND
WIDTH rather than data. Branch `lane/neural-shape-shrink`, cut from `main`.

**Nothing here touches the frozen 0.8.6 commit db9047b9f or `release/0.8.6`.**
No box was rented. Every Metal job went through `mac_slot.sh metal`, one at a
time; every CPU check ran one core at `nice -n 19`.

## THE HEADLINE

The launch-overhead theory is **CORRECT**, and it is now proved from the
source rather than inferred from timings. It also turns out to apply to
**one** lane, because every other expensive neural lane is a single block at a
d_model floor and therefore has no depth to remove.

- **Cut: `byte-lm-resident`**, two blocks at d_model 32 down to one block at
  d_model 16. Metal **173.95 s to 77.26 s, 2.25x**, one fixture at two
  repeats, same route both sides. Merged to main as `a7b3b0393`.
- **Floors: `byte-lm`, `samba`, `samba-untied-dropout-accum`,
  `mamba2-dtlimit`** and the sequence-block siblings. Each is named below with
  its mechanism, and the two mechanical refusals were SEEN to fire.

## 1. THE THEORY, SETTLED THREE WAYS

Metal cost in these lanes is per-kernel-launch overhead, not arithmetic.

**(a) The launch count is a function of DEPTH ALONE.** Traced through
`training/byte_lm.mojo` and the modules its block loops call, one byte LM
training step issues

    total = (64 + 23*G) * L + (32 + 5*G)      L = n_layers
    G     = launches per identical_gemm call (1 on Apple; 2 on the
            NVIDIA/AMD ksplit group path)

so at G = 1 it is `87*L + 37`: **211 launches at L = 2, 124 at L = 1**. Per
block that is 24 forward, 45 backward and 18 more that are easy to miss,
because `_unpack_block` and `_pack_block` issue nine `_copy_into` each and
`_copy_into` is a real kernel (`training/checks/train_loop.mojo:1302`), not a
copy. The decisive part: **no launch on the shipped path sits inside a loop
over sequence length, d_model or head count.** Only the grid dimensions scale.

That is why the 2026-09-16 fixture shrink bought nothing here. It cut steps
and sequence length, which removes arithmetic and not one launch.

**(b) Sequence length is inert, measured.** `mamba2-dtlimit`'s block at
d_model 32, per call:

| L | forward | prefill+state | step | backward | sum |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.288 s | 0.283 s | 0.328 s | 0.417 s | 1.316 s |
| 16 | 0.304 s | 0.308 s | 0.323 s | 0.420 s | 1.355 s |

Doubling the sequence costs about 3%. There is no arithmetic in this lane to
remove.

**(c) Depth is not inert.** The same model, Metal, per step:

| byte LM shape | stateless step | resident step | logits |
|---|---:|---:|---:|
| 2 blocks, d32, ff64 (shipped) | 1.89 s | 1.380 s | 0.85 s |
| 1 block, d32, ff64 | 1.09 s | 0.849 s | 0.44 s |
| 2 blocks, d16 | 1.60 s | | 0.77 s |
| 2 blocks, ff32 | 1.73 s | | 0.71 s |
| 1 block, d16, ff32 | 0.90 s | 0.644 s | 0.24 s |

Halving the blocks removes about 42% of a step. Halving d_model alone removes
about 15%, and halving the MLP width removes about 8%. Depth is the lever,
exactly as the launch formula says.

## 2. WHAT WAS CUT

### `byte-lm-resident`, 2.25x on Metal

`ByteLanguageModelConfig(n_layers=1, d_model=16, n_heads=2, n_kv=1,
head_dim=8, intermediate=32)`. `batch` and `length` are untouched, so the
token stream is the same bytes.

| route | before | after |
|---|---:|---:|
| Metal, one fixture, two repeats | **173.95 s** | **77.26 s** |
| launches per training step (G=1) | 211 | 124 |

**Why this lane and not `byte-lm`.** This lane's claim is the SESSION, not a
shape. It asserts that the resident export equals the stateless path's
gradient byte for byte at whatever shape both are built at (`_same_bytes` in
the lane body), so the claim is shape-independent and survives the cut.

### The gate, seen both ways

The rule was mechanical. Confirm the arm fires at the CURRENT size first, so a
firing arm afterwards is not an artifact; then cut, and require the arm to
fire again.

| | train | infer | model | batch | rlpair |
|---|---|---|---|---|---|
| production, new size | STABLE | STABLE | STABLE | STABLE | STABLE |
| sabotage, current size | **DIVERGENT** | | | | |
| sabotage, new size | **DIVERGENT** | | | | |

Both DIVERGENT cells read `parts differ: loss,params,grads,logits; agree: -`.
CPU host route, one core, `MOJOLEARN_NUMERIC_MODE=identical` set explicitly.

**The sabotage twin was nearly built inert, and that is worth recording.** The
first twin was built with `-D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1`, which is the
INFERENCE head-fold arm (`docs/BYTE_LM_CPU_INFERENCE.md`). The define this
binding actually names as its negative control is
`-D MOJOLEARN_HOST_SABOTAGE=1` (`bindings/_mojolearn_byte_lm_host.mojo:32`),
which reaches the training step and the logits through `gemm_oracle`. A twin
built with the wrong define computes right answers, and a gate run against it
would have passed on nothing.

### A regression this found and fixed

Four part helpers constructed `ByteLanguageModelConfig()` outright, correct
only while every byte LM lane ran the shipped profile. At the new shape the
rlpair part read **REFUSED**, `parameters must be float32 [34944]`. That is a
live check turning into a refusal, not a passing gate.

`_byte_lm_shape(ml, e)` now reads the shape off the fit. The inference classes
and the host trainer expose `shape`; the GPU trainer carries it in
`state_dict()['model_shape']`, which it writes only when the shape is not the
shipped profile. Harness only, no library API moved, so the inventory gate is
untouched.

Proof it moved nothing else: `byte-lm`, `byte-lm-host-infer`,
`byte-lm-host-infer-threaded` and `byte-lm-host-train` read **IDENTICAL x2**
against `main` across train, infer, model, batch and rlpair after the change.

## 3. THE FLOORS, WITH MECHANISMS

### `byte-lm`. The shipped profile, and its only device column.

`mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1` is the
published model's shape. `training/byte_lm.mojo:92` pins it as a comptime,
`training/BYTE_LM_GRADIENT_ORACLE.md` defines the oracle against exactly
"two independent Llama decoder blocks", and `tools/byte_lm_shape.py`,
`tools/verify_linux_surface_qualification.py` and
`tools/test_byte_lm_shape.py` all carry the string. The `byte-lm-host-*` lanes
run the HOST binding even on a GPU box, so `byte-lm` is the only lane that
hashes the shipped profile's DEVICE arithmetic. Moving it would leave the
published shape with no device column. **Floor.**

### `mamba2-dtlimit` and the sequence-block siblings. One block, at the floor.

Each of `mamba1`, `mamba2`, `mamba3`, `mamba2-dtlimit`, `transformer` and
`transformer-window` is a SINGLE block, so there is no depth axis at all. For
the Mamba blocks d_model 32 is a hard floor, **SEEN to refuse**:

    mojolearn Mamba2Block: d_model must be a multiple of 32 so that
    nheads = 2*d_model/64 is whole (profile constants headdim 64,
    expand 2 -- Mamba2Dims.of carries the same refusal); got 16

`d_state`, `headdim`, `ngroups` and `chunk` are profile constants and
CHUNK_SIZE is part of the arithmetic (DEVIATION 783), never a tuning knob. So
`mamba2-dtlimit`, the sharpest test of the theory, has no axis to move: its
cost is real per-launch overhead and nothing in this lane's toolkit removes it.

`transformer` admits free shapes, so it was measured rather than assumed, and
it does not pay. Per-call sum of forward, prefill, step and backward:

| transformer shape | sum |
|---|---:|
| baseline d32 nh2 hd16 it64 | 1.121 s |
| d32 **nh1** hd32 it64 | 1.287 s (worse) |
| d16 nh1 hd16 it32 | 1.090 s (1.03x) |
| d16 **nh2** hd8 it32 | 1.418 s (worse) |

A 3% gain does not justify moving a hash and re-recording the lane on three
vendors. `transformer-window` behaves the same way (0.913 s to 0.834 s).
**Floor by measurement.**

`mamba1` does admit d_model 16 (forward 0.158 s to 0.127 s, backward 0.278 s
to 0.232 s, about 1.2x), but the six sequence-block lanes deliberately share
one `_seq(X, 2, 16, dm)` shape at the smallest legal d_model so their cells are
comparable. Desynchronizing one lane of the family for 20% of a 429 s lane is
not worth it. **Left alone.**

### `samba` and `samba-untied-dropout-accum`. The stack IS the claim.

`SambaConfig(layers=("mamba3", "attention"))` is one Mamba-3 layer and one
attention layer. Depth is a real lever here, measured: 1.656 s for the pair
against 0.968 s for the Mamba-3 alone and 1.015 s for the attention alone, and
6.906 s against 4.344 s and 4.164 s for the accumulation lane. But a stack of
one kind is not a stack of two kinds, and the heterogeneous composition is the
thing these lanes exist to prove. **Floor, and it binds.**

Width gives nothing anyway. d_model 32 is the floor and it was **SEEN to
refuse** at 16:

    mojolearn.SambaConfig: a mamba3 layer needs d_model to be a multiple
    of 32 (Mamba3Block)

and the head count and MLP width are flat (baseline 1.015 s, `n_heads=1`
1.080 s, `n_heads=1` with `intermediate=32` 1.017 s).

`samba-untied-dropout-accum` additionally keeps the two floors the previous
lane found, 32 rows because `accumulation_is_aligned(512, 4)` is the A=4 claim
and the third step because it is the first that evaluates the cosine arm of
the warmup schedule.

> **THAT SECOND HALF WAS NOT TRUE WHEN IT WAS WRITTEN** (2026-09-16,
> lane/shrink-floors). The lane had already been cut to ONE step, so the third
> step was not kept and the cosine arm was never evaluated: measured, the cell
> could not tell `WarmupCosineLR` from `WarmupLinearLR` or from `ConstantLR`
> (`docs/lanes/LANE_STATUS_shrink-blindness-audit.md` section 3). This is the
> third document in a chain that lost a floor written down in
> `docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md` section 1f. The lane
> is back at three steps, and the floor is now `@floor(steps=(3, ...))` on the
> lane itself with `tools/fixture_floors.py` refusing a cut past it, so no
> document has to be right about this again.

## 4. THE REVISED APPLE COLUMN

The projection stood at **8.32 h**. `byte-lm-resident` is measured at 173.95 s
per fixture before and 77.26 s after, so over nine fixtures it removes
96.7 x 9 = **870 s, about 0.24 h**.

| | hours |
|---|---:|
| before this lane | 8.32 |
| **after this lane** | **8.08** |

**Say the size of this plainly.** It is about fifteen minutes off an eight
hour column. The theory was right and the one lane it could be applied to gave
a clean 2.25x, but most of the Apple column's neural cost sits in single-block
lanes at a d_model floor, where this lever does not exist. Anyone hoping to
make the Apple column cheap should read section 1 and conclude that the
remaining target is the PROBE COUNT (about 24 device round trips per cell) or
the per-launch cost itself, not the models.

## 5. WHAT THIS DOES NOT CLAIM

- The Metal numbers are one fixture at two repeats, not a column. The
  per-fixture unit is the previous lane's, cross-checked four ways there.
- `LANE_REVISIONS` carries `byte-lm-resident` at `shape-l1-d16-ff32-1`, so
  columns recorded at the old shape read OWED to the next record rather than
  DIVERGENT against different bytes.
- `verify_reference/table.json` is **NOT** regenerated here. The staleness
  guard drops a lane whose fixture moved past its reference and names it, and
  the release regenerates the table.
- The launch counts are static, traced from the source on a shipped IDENTICAL
  build. `G` is shape and column dependent, and the fused attention path is a
  runtime decision; head_dim 8 refuses it, so the byte LM runs the eager arm,
  whose count scales with `b * n_heads`.

## Rules this lane ran under

One core, `nice -n 19`, one process at a time, own worktree, never the shared
checkout. Host bindings built into scratch directories with fresh inodes. No
box rented. Every Metal job took the shared Metal lock through
`mac_slot.sh metal` and released it.

## Resume

    cd /Users/andrewhendel/CascadeProjects/mojolearn      # SHARED: never build or commit here
    git worktree add -b <branch> <scratch>/wt main
    cd <scratch>/wt

Metal route (the 0.8.6 wheel supplies the bindings, this tree supplies the
harness; do NOT set PYTHONPATH, and do NOT set MOJOLEARN_HOST_DIR):

    python3 -m venv <scratch>/venv086
    <scratch>/venv086/bin/pip install ~/mojolearn-evidence/release-0.8.6/macos-wheel/*.whl numpy
    bash ~/mojolearn-evidence/release-0.8.6/scripts/mac_slot.sh metal \
      <scratch>/venv086/bin/python tools/identity_break.py --lanes <lane> \
      --fixtures base --repeats 2

CPU gate route, production and the twin (note the define):

    MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_HOST_OUTDIR=<scratch>/hostbuild \
      nice -n 19 sh bindings/build_core_host.sh          # and training
    MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_BYTE_LM_HOST_OUTDIR=<scratch>/hostbuild \
      nice -n 19 sh bindings/build_byte_lm_host.sh
    # the twin: -D MOJOLEARN_HOST_SABOTAGE=1 for EVERY family, byte_lm included.
    # MOJOLEARN_BYTE_LM_HOST_SABOTAGE is the inference head-fold arm and leaves
    # the training step computing right answers.
    # Each build refuses an output directory that already exists.

## Pods

None rented on this branch.
