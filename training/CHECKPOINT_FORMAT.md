# The checkpoint file. `mojolearn.identical.train.ckpt.file.v1`

Written 2026-09-03 by the checkpoint-file lane.
**DEVIATIONS 2050 through 2069 are this lane's, PROPOSED and not yet in the
orchestrator's ledger.** 1550-1589 belong to the training-loop lane and
1581-1589 are reserved by `TRAINING_LOOP_PLAN.md` for the NVIDIA leg, so this
lane took a fresh block rather than borrowing from a reservation.

## STATUS

**WRITTEN, NOT RUN. NOTHING IN THIS FILE, IN `training/checkpoint.mojo` OR IN
`training/checks/checkpoint_check.mojo` HAS EVER BEEN COMPILED OR EXECUTED.**
No compiler has read any of it. No byte has ever been written to disk by it.
No file has ever been loaded back. **Every "refuses", "round trips" and
"equals" below is a PREDICTION**, and the exact commands that would falsify
them are in section 10.

This file does NOT close `TRAINING_LOOP_PLAN.md` section 12 item 8. That item
is closed by a RUN, on two vendors, and the run is owed.

---

## 0. Why this file exists

`TRAINING_LOOP_PLAN.md:49` states the gap in one line:

> No checkpoint file format, so clause (d) tests resume WITHIN one process.

Section 4.2's N6 control says the same thing at length, and section 7 item 6
turns it into a disclaimer: **"No claim about resuming from a file."** The
training loop already reaches `h_all = 463245ce6c97e68d` on an Apple M4 and an
AMD MI325X over eight steps, which is a real result about a run. It is not yet
a result about a CHECKPOINT, because nothing the loop produces survives the
process that produced it.

The thing worth having is *train here, resume there*: eight steps on an M4,
four of them on an MI325X and four more on an H100, and the same digest. That
requires training state to become a file, and it requires the file to be
**the same file** on all three machines. This spec is that file.

**The equality IS the test.** A checkpoint written on Apple and a checkpoint
written on AMD from the same training state must be byte-for-byte identical --
not "equal after parsing", not "equal to within the digest", identical under
`cmp`. Any field that could differ between two machines running the same
configuration is a field that must not be in this file. Section 3 is the list
of what is therefore excluded, and it is the specification.

`OPT_SAB_RESUME_REINIT`'s real form -- a checkpoint that drops the momentum
flags on the way to disk -- is unreachable from `resume_training`, which never
leaves memory. It is reachable from here, and
`training/checks/checkpoint_check.mojo` clause (iii) is what reaches it.

---

## 1. What the training state IS

Ground truth is `training/checks/train_loop.mojo`. A step's carried state, and
nothing else, is:

| what | where it lives | shape | dtype |
|---|---|---|---|
| parameters | `TrainBuffers.param` | `n_total` | fp32 |
| Adam first moment | `TrainBuffers.m_state` | `n_total` | fp32 |
| Adam second moment | `TrainBuffers.v_state` | `n_total` | fp32 |
| momentum-initialized flags | `TrainBuffers.buf_initialized` | `J` | bool |
| step counter | `train_step`'s `t` | 1 | int, ONE-BASED |
| the layout | `TrainBuffers.offsets` | `J + 1` | int |

Everything else in `TrainBuffers` is scratch. `grad` is recomputed from the
batch every step; the twenty-odd cross-entropy, gemm-workspace and
embedding-run buffers are intermediates; there is no KV cache between steps
(DEVIATION 1554, a fresh `LlamaKVCache` at `pos0 = 0` every step); the eleven
per-tensor weight buffers are a VIEW refreshed from the flat one at the top of
every step (DEVIATION 1553). **The flat buffer is the state.** A checkpoint of
the per-tensor view would be a checkpoint of something the optimizer never
wrote, which is failure mode V3 in the plan's table.

There is also no RNG stream to carry. `train_batch_ids(seed, step_index)` is a
pure function of two integers (plan section 6), so `(seed, t)` reproduces every
future batch exactly. **That stops being true the moment anything stochastic
enters** -- dropout, a sampler, a shuffled loader -- and adding any of them
needs this paragraph rewritten, not extended.

At the v1 shape, `J = 11` and `n_total = 13376`.

---

## 2. The byte layout

**Every multi-byte field is LITTLE-ENDIAN and is written BYTE BY BYTE from an
integer, never by copying host memory.** That distinction is the whole reason
the byte order is a property of the FORMAT and not of the machine: a `memcpy`
of a `List[Float32]` is little-endian only because every host this repository
runs on happens to be, and "happens to be" is not a specification.

**Every float is its raw IEEE-754 binary32 bit pattern.** Never a decimal
text representation, anywhere, for any reason. `String(Float32)` does not round
trip in this toolchain (`[[mojo-string-float-roundtrip]]`), and a checkpoint
that went through a decimal round trip could report agreement across a real
difference -- the worst failure an instrument of this kind can have. That rule
covers the hyperparameters in the descriptor as much as the payload:
`lr` is stored as `0x3A83126F`-style bits and not as `"0.001"`.

The file is four sections and a trailer, in this order, with no padding
between them and no alignment requirement beyond what the layout already
gives.

    [ header 64 ][ descriptor 64 ][ layout 32*J ][ payload 12*n_total ][ trailer 16 ]

    file_bytes = 64 + 64 + 32*J + 12*n_total + 16

At `J = 11, n_total = 13376` that is **161008 bytes exactly**, and a file of
any other length for that shape is refused before a single float is read.

### 2.1 Header, 64 bytes, at offset 0

| off | size | type | field | value |
|---|---|---|---|---|
| 0 | 8 | bytes | `magic` | `4D 4C 43 4B 50 54 30 31`, ASCII `MLCKPT01` |
| 8 | 4 | u32 | `format_version` | 1 |
| 12 | 4 | u32 | `header_bytes` | 64 |
| 16 | 4 | u32 | `descriptor_bytes` | 64 |
| 20 | 4 | u32 | `layout_bytes` | `32 * j_count` |
| 24 | 4 | u32 | `payload_bytes` | `n_arrays * n_total * 4` |
| 28 | 4 | u32 | `trailer_bytes` | 16 |
| 32 | 4 | u32 | `j_count` | number of parameter tensors, `>= 1` |
| 36 | 4 | u32 | `n_total` | elements in each state array, `>= 1` |
| 40 | 4 | u32 | `n_arrays` | 3 -- param, m, v, in that order |
| 44 | 4 | u32 | `elem_bits` | 32 |
| 48 | 4 | u32 | `t` | optimizer steps COMPLETED; the next step is `t + 1` |
| 52 | 4 | u32 | `reserved0` | 0 |
| 56 | 4 | u32 | `reserved1` | 0 |
| 60 | 4 | u32 | `reserved2` | 0 |

The four size fields are redundant with `j_count` and `n_total` on purpose.
They are what lets a reader compute the expected file length and check it
BEFORE it indexes anything, which is the difference between refusing a
truncated file and reading past the end of one.

`t` is in the header and not the descriptor because it is carried STATE and not
configuration. It is **ONE-BASED and counts completed steps**:
`identical_optimizer_step` was last called with `t = <this field>`, and a
resume calls it next with `t + 1`. A file written before any step has `t = 0`.
Getting this off by one moves `beta^t`, which the optimizer contract's two
spellings agree on through `t = 6` and first disagree at `t = 7` -- so an
off-by-one here is INVISIBLE in any run shorter than seven steps and then
appears at step seven of an eight-step run. That is why clause (iii)'s `t`
control exists and why `N = 8` is the number to run it at.

### 2.2 Descriptor, 64 bytes, at offset 64

The run's configuration. **It is not carried state and it is not in the
content hash** (section 4), and it is here for exactly one job: so that a
loader can REFUSE a resume under a configuration different from the one that
produced the state.

`TRAINING_LOOP_PLAN.md` section 7 item 4 states the hole this closes:
*"comparing two digests produced under different `lr` is meaningless and the
harness cannot detect it. The card's header line records the configuration and
a comparison must check it by eye."* A file can be checked by a machine.

| off | size | type | field |
|---|---|---|---|
| 64 | 8 | u64 | `seed` |
| 72 | 4 | u32 | `opt_kind`, `optimizer_oracle`'s `OPT_SGD 0 / OPT_ADAM 1 / OPT_ADAMW 2` |
| 76 | 4 | u32 | `lr` bits |
| 80 | 4 | u32 | `beta1` bits |
| 84 | 4 | u32 | `beta2` bits |
| 88 | 4 | u32 | `eps` bits |
| 92 | 4 | u32 | `weight_decay` bits |
| 96 | 4 | u32 | `momentum` bits |
| 100 | 4 | u32 | `dampening` bits |
| 104 | 4 | u32 | `nesterov`, 0 or 1 and NO other value |
| 108 | 4 | u32 | `max_norm` bits |
| 112 | 4 | u32 | `steps_planned`, the run's `N`; 0 means unrecorded |
| 116 | 4 | u32 | `arm`, `train_loop`'s `ARM_*` id |
| 120 | 4 | u32 | `reserved0` = 0 |
| 124 | 4 | u32 | `reserved1` = 0 |

`arm` is here because a checkpoint written under the `ulp` control and one
written clean are two different states that a `cmp` would correctly report as
different and a tired operator would call a cross-vendor divergence. The field
turns that into a refusal with a name.

**THE DESCRIPTOR IS PART OF THE BYTE-EQUALITY CLAIM, AND THAT IS A HAZARD
WORTH STATING.** Two legs whose `MOJOLEARN_TRAIN_SEED` differs produce
different files, and correctly so. Two legs must be launched with the same
configuration for the comparison to mean anything at all -- which was already
true of the digest, silently. Here it is visible in the first 128 bytes.

### 2.3 Layout table, 32 bytes per tensor, `j_count` entries, at offset 128

Entries are in ASCENDING `param_id` and `param_id` IS the index. Plan section
2 and DEVIATION 1550: **this order is part of the checkpoint hash
specification and changing it invalidates every digest this profile has ever
produced.** The table is written into the file so that a loader can check the
order it was told to expect against the order that was written, rather than
assuming they agree.

| off within entry | size | type | field |
|---|---|---|---|
| 0 | 4 | u32 | `param_id`, must equal the entry's index `j` |
| 4 | 4 | u32 | `offset`, `offsets[j]` |
| 8 | 4 | u32 | `count`, `offsets[j+1] - offsets[j]`, `>= 1` |
| 12 | 1 | u8 | `buf_initialized[j]`, 0 or 1 and NO other value |
| 13 | 3 | u8 | reserved, all zero |
| 16 | 16 | ascii | `name`, NUL padded, from `[A-Za-z0-9._-]` |

The name is in the file because "self-describing" has to mean something a
human with `xxd` can act on. `embed`, `norm1_w`, `w_q`, `w_k`, `w_v`, `w_o`,
`norm2_w`, `w_gate`, `w_up`, `w_down`, `lm_head` all fit in fifteen bytes with
room to spare, and a refusal that says *"tensor 4 is named `w_o` in the file
and `w_v` in the caller's layout"* is a refusal somebody can fix. A refusal
that says *"shape mismatch"* is not.

`buf_initialized` is the optimizer's per-tensor momentum flag. **Under AdamW it
is never read** -- `identical_optimizer_step` consults it only on the SGD
branch -- so under the v1 configuration this byte is carried and is bit-inert,
and clause (iii)'s flag control is REPORTED and not asserted for exactly that
reason (`[[reached-but-inert]]`). It is written anyway because dropping the
flags on the way to disk is `OPT_SAB_RESUME_REINIT`'s named failure and a
format that has nowhere to put them makes that failure unfixable rather than
untested.

### 2.4 Payload, `n_arrays * n_total * 4` bytes, at offset `128 + 32*J`

Three fp32 arrays, back to back, no headers between them, no padding:

    param[0 .. n_total-1]  then  m_state[0 .. n_total-1]  then  v_state[0 .. n_total-1]

Each element is four bytes, little-endian, its raw IEEE-754 binary32 bit
pattern, **including every NaN payload and both signed zeros**. `-0.0` is
`00 00 00 80` and `+0.0` is `00 00 00 00` and the format keeps them apart,
because `m` starting at exactly `+0.0` is what makes `OPT_SAB_MOMENT_LERP`
bit-inert on a first step and a format that normalized the sign would erase a
distinction the optimizer contract depends on.

Parameters are in flat `param_id` order over one contiguous range, so
`param[offsets[j] .. offsets[j+1]-1]` is tensor `j`, exactly as
`train_offsets()` says.

### 2.5 Trailer, 16 bytes, at the end

| off from end | size | type | field |
|---|---|---|---|
| -16 | 8 | u64 | `h_all`, the content hash of section 4 |
| -8 | 8 | u64 | `h_file`, FNV-1a64 over bytes `[0, file_bytes - 8)` |

`h_file` covers everything before itself, `h_all` included. `h_all` covers the
payload and NOTHING else. Two hashes, because they answer two different
questions: `h_file` asks *did this file arrive intact*, and `h_all` asks *is
this the same training state*. A file can be intact and hold different state,
and a file can hold the right state and have been truncated in transit, and
conflating the two is how a corruption gets read as a divergence.

---

## 3. Vendor neutrality, which is the point

**The file must not encode anything about the device that wrote it.** Not a
device pointer, not a vendor tag, not a driver version, not an SM or CU count,
not a block size, not a plan `choose_gemm_plan` picked, not a workspace size,
not an allocation address, not a buffer CAPACITY, not a wall clock, not a
hostname, not a process id, not the output path, not the numeric mode banner,
not the sabotage-arm names the build carries.

Two structural facts enforce that rather than a convention doing it:

1. **`training/checkpoint.mojo` does not import `max.gpu.host`.** It has no
   `DeviceContext` and no `DeviceBuffer` in any signature. It cannot name a
   device because it cannot see one. Everything reaching it is a host
   `List[Float32]`, a `List[Int]`, a `List[Bool]` and a handful of integers.
   The download from device to host happens in the CALLER, and the caller is
   the gate. This is not an aesthetic choice; a serializer that took a
   `DeviceContext` would be one refactor away from putting `ctx`'s properties
   in the header.
2. **Every count in the file is `n_total` or a tensor extent, never
   `len(buf)`.** `core/identity_trace.mojo` rule 3 and the plan's section 3:
   a buffer allocated with slack and hashed to its capacity folds
   uninitialized memory into the digest, which differs run to run on ONE
   machine and would make the instrument report divergence everywhere. There
   is no capacity anywhere in this format. The counts come from the layout
   table and the layout table comes from `train_offsets()`.

The file is a pure function of

    (param bits, m bits, v bits, offsets, names, buf_initialized, t, descriptor)

and of nothing else. Every one of those is identical on two machines that ran
the same configuration for the same number of steps, **so byte-for-byte
equality of two files is exactly the claim, and `cmp -l apple.bin amd.bin` is
the whole verdict.**

`[[always-gpu-agnostic]]`: there is no `if apple` in `training/checkpoint.mojo`
and there is nowhere one could go. The file contains no arithmetic on floats
at all -- it moves bit patterns -- so there is no rounding decision in it to
diverge on.

---

## 4. The content hash

**FNV-1a64. Offset basis `0xCBF29CE484222325`, prime `0x100000001B3`, folded
ONE BYTE AT A TIME over the LITTLE-ENDIAN bytes of the elements IN INDEX
ORDER.** It is `core/identity_trace.mojo::fnv1a64_bytes`, CALLED and not
restated, for the reason that file gives: a word-at-a-time variant is faster
and is a DIFFERENT FUNCTION, `tools/identity_trace_diff.py` recomputes this
from a `.bin` dump, and the writer and the reader have to be the same spelling.
A second hash function in one repository is a second thing to get wrong.

    h_all  = h(h(h(FNV_OFFSET, param, n), m_state, n), v_state, n)

which, because the payload is those three arrays contiguously and in that
order, is **one continuous fold over the payload region** -- a single call over
`payload_bytes` bytes starting at `128 + 32*J`. The two spellings are the same
number by construction and `checkpoint.mojo` uses the single-fold one.

    h_file = h(FNV_OFFSET, file_bytes[0 .. file_bytes-8))

`h_all` is DELIBERATELY the same number as `TrainDigest.h_all` from
`train_loop.mojo::digest_of_lists`. That is what makes a checkpoint file and a
trace record comparable: the sidecar `MOJOLEARN_TRAIN_CKPT` already prints
`h_all` as sixteen hex digits, and the same sixteen digits are now bytes
`-16 .. -9` of the checkpoint file.

**The coincidence is asserted and not assumed.** `digest_of_lists` folds the
raw host memory behind a `List[Float32]`; this format folds bytes it built
explicitly from `bitcast[DType.uint32]`. Those two byte sequences are the same
on a little-endian host and are NOT the same on a big-endian one. Every machine
in this repository's matrix -- Apple arm64, x86-64, AMD -- is little-endian, so
they agree today; `checkpoint_check.mojo` clause (i) compares the two numbers
rather than trusting the argument, and on a big-endian host it would fail
loudly instead of producing a file whose stated hash was wrong.

**What is NOT in `h_all`, and this list is the specification.** The header, the
descriptor and the layout table. That means `h_all` does not read `t`, the
seed, the learning rate, the arm, the tensor names, the offsets, or the file's
own length -- which is the plan's section 3 exclusion list, kept intact. A
checkpoint's content hash is a function of the STATE BITS and of nothing else,
so that a state saved at step 4 of an eight-step run and the same state
reconstructed some other way hash the same.

`h_file` is the opposite: it covers every byte the format defines except
itself, so a changed `t`, a changed learning rate or a changed name DOES move
it. That is the field a transfer is checked with.

---

## 5. What is refused, and why

**REFUSE, DO NOT GUESS.** Every check below raises, none of them warns, none of
them falls back, and none of them repairs. Each raise begins with a stable
uppercase tag so a gate can assert on the CATEGORY of a refusal rather than on
the fact that something raised -- a refusal for the wrong reason is not a pass.

| tag | fires when | why not guess |
|---|---|---|
| `CKPT_REFUSE_MAGIC` | bytes 0..7 are not `MLCKPT01` | a `.trace`, a `.bin` dump, a truncated download or a text sidecar handed to the loader by mistake would otherwise be parsed as a header and produce a shape from noise |
| `CKPT_REFUSE_VERSION` | `format_version != 1` | a v2 file read by a v1 reader is field-shifted garbage that still has plausible float values in it. There is no migration path and there is not meant to be one |
| `CKPT_REFUSE_HEADER` | a fixed field is wrong (`header_bytes`, `descriptor_bytes`, `n_arrays`, `elem_bits`, `trailer_bytes`), a size field disagrees with `j_count`/`n_total`, `j_count < 1`, `n_total < 1`, or any reserved field is nonzero | the reserved-must-be-zero rule is what makes a v1 reader safe against a writer that quietly started using one |
| `CKPT_REFUSE_TRUNCATED` | the file is shorter than the header, or shorter than `64+64+32*J+12*n+16` | there is no atomic write here. A crash mid-write leaves a short file and the loader must say so, not read the payload it has and zero the rest |
| `CKPT_REFUSE_TRAILING` | the file is LONGER than the computed length | extra bytes mean the writer and the reader disagree about the format. Ignoring a tail is how two versions coexist silently |
| `CKPT_REFUSE_FILE_HASH` | recomputed `h_file` differs from the stored one | the transfer or the disk changed a byte. Naming this separately from the content hash is what separates "corrupted in transit" from "different training state" |
| `CKPT_REFUSE_LAYOUT` | `param_id != j`, `offsets[0] != 0`, offsets not contiguous and ascending, a count `< 1`, the counts do not sum to `n_total`, `buf_initialized` is not 0 or 1, a reserved byte is nonzero, or a name is not NUL-padded printable `[A-Za-z0-9._-]` | a wrong offset gives plausible, in-bounds, wrong numbers that are IDENTICAL on all three vendors, so **no cross-vendor comparison can see it** (plan V7). This is one of the two places that can |
| `CKPT_REFUSE_SHAPE` | the file's `j_count`, `n_total`, per-tensor `offset`, per-tensor `count` or per-tensor `name` differs from the layout the CALLER passed in | this is the "refuse BY NAME" requirement. The message names the `param_id`, the tensor's name on both sides, and the two numbers |
| `CKPT_REFUSE_CONTENT_HASH` | recomputed `h_all` over the payload differs from the stored one | the only failure that survives a correct `h_file`, so it can only be reached by an edit that repaired the file hash. It is checked anyway, because a content hash that no arm can fire is decoration |
| `CKPT_REFUSE_STATE` | `save_checkpoint` is handed inconsistent state: `param`/`m`/`v` of different lengths, `len(offsets) != J+1`, `len(names) != J`, `len(buf_initialized) != J`, a name too long or outside the charset, `t < 0`, or an empty path | refusing at SAVE time is cheaper than discovering it at LOAD time on another continent |

The order is deliberate: length before magic-independent parsing, magic before
version, version before sizes, sizes before the file hash, the file hash before
the layout, the layout before the payload. **Nothing indexes into the buffer
until the length check has passed**, which is the difference between a refusal
and a read past the end.

`CKPT_REFUSE_SHAPE` is checked against the caller's expectation and NOT against
a compiled-in constant, so `checkpoint.mojo` stays free of the v1 model shape.
`n_total = 13376` and `J = 11` appear nowhere in it. A second architecture --
`TRAINING_LOOP_PLAN.md` section 12 item 9 -- needs no change to this file.

---

## 6. What this does NOT cover

**This is not a model file.** It cannot be loaded by anything except this
repository's own training loop, it names no architecture, and it is not an
interchange format. It has no relationship to safetensors, GGUF or a PyTorch
`state_dict` and is not meant to acquire one.

1. **No tensor SHAPES, only counts.** The layout table records that tensor 10
   is called `lm_head` and holds 2048 elements. It does not record `[64, 32]`.
   **A transposed weight of the same element count loads without complaint**,
   and `[V, d_model]` versus `[d_model, V]` is exactly the kind of defect that
   is identical on all three vendors. Clause (f) of `train_step_check.mojo`
   is what catches that class, and this format does not replace it. Recording
   a rank and extents is the obvious v2 field and is OWED.
2. **No dtype variety.** FP32 only, `elem_bits = 32`, refused otherwise. No
   bf16, fp16, tf32 or float64 -- and no device float64 exists in this
   repository's matrix anyway (`[[mojolearn-hardware-limits]]`).
3. **No RNG state**, because there is none: batches are a pure function of
   `(seed, step_index)`. **A stochastic data path makes this format
   incomplete, not merely inconvenient**, and adding dropout or a shuffled
   loader requires a v2 with a stream field.
4. **No KV cache, no activations, no gradients.** All three are per-step and
   two of them do not exist between steps by design (DEVIATION 1554).
5. **No multiple parameter groups.** One optimizer configuration per file.
6. **No optimizer state beyond `m`, `v` and the per-tensor flags.** An
   optimizer with a third moment, an EMA of the weights, or a per-tensor
   trust ratio does not fit and must not be squeezed in.
7. **No atomicity.** No temp file, no rename, no lock. A crash mid-write
   leaves a short file that the loader REFUSES as truncated. That is the
   intended behavior and not a gap to paper over -- a half-written checkpoint
   that loaded would be the worst outcome available.
8. **No integrity against malice.** FNV-1a64 is not a MAC. Anyone who can edit
   the payload can recompute both hashes. The hashes catch a flipped bit, a
   short transfer and a wrong file; they catch nothing that is trying.
9. **No compression, no streaming, no partial reads.** The whole file is read
   into host memory in one go. At 161 KB that is free; at a real model size it
   is a design that would have to change.
10. **No migration.** A v2 reader is not required to read a v1 file and a v1
    reader REFUSES a v2 file. The version field exists to make the refusal
    early and named, not to enable a compatibility shim.
11. **No claim that the state is CORRECT.** `TRAINING_LOOP_PLAN.md` section
    1.1, restated because it is the sentence most likely to be skipped: two of
    the twelve stages of a step have never been certified against their
    oracle. **Three machines writing byte-identical checkpoints of the same
    wrong gradient agree perfectly.** A matching file shows agreement and
    never correctness.
12. **No performance number.** Saving downloads three `n_total` buffers to the
    host and folds every byte twice. Any timing taken from a run that writes
    checkpoints is fiction, exactly as plan section 7 item 7 says of a traced
    run.
13. **Big-endian hosts are undefined territory.** The format is little-endian
    by definition and would still be written correctly on a big-endian host,
    but `h_all` would stop coinciding with `digest_of_lists`' fold of host
    memory. Clause (i) asserts the coincidence and would fail there. No such
    host is in the matrix and none is planned.

---

## 7. The implementation

`training/checkpoint.mojo`, and it is deliberately small.

    struct Checkpoint            the whole state, on the HOST
    struct CheckpointHashes      (h_all, h_file, bytes) returned by a save
    save_checkpoint(path, ck) -> CheckpointHashes
    load_checkpoint(path, expect_offsets, expect_names) -> Checkpoint
    read_checkpoint_unchecked(path) -> Checkpoint
    compare_checkpoint_files(path_a, path_b) -> String

`load_checkpoint` is the entry point everything should use: it takes the
layout the caller expects and refuses a file that does not match it.
`read_checkpoint_unchecked` skips ONLY that comparison -- magic, version,
sizes, length, both hashes and the internal layout consistency are still
enforced -- and exists so a tool can inspect a file whose shape it does not
know. **It is not a "load anyway" escape hatch and must not be used to get past
a `CKPT_REFUSE_SHAPE`.**

`compare_checkpoint_files` returns `""` when two files are byte-for-byte
identical and otherwise a report naming the first differing offset, the section
that offset falls in, and both files' `h_all` and `h_file`. That is the
function the orchestrator points at two vendors' checkpoints; a bare `cmp` is
the same verdict with less of an address.

### 7.1 The file-I/O idiom, and why this one

**The repository has exactly one binary-file idiom in Mojo and this file uses
it rather than inventing a second.** `core/identity_trace.mojo:311` writes a
`.bin` dump with

    with open(path, "w") as fh:
        fh.write_bytes(Span(bytes))

and `checks/identity_trace_check.mojo:376` reads one back with

    with open(bin_path, "r") as fh:
        var body = fh.read_bytes()

then re-folds it and compares against the recorded FNV -- which is the
repository's own evidence that this pair round-trips arbitrary bytes, NUL bytes
included, through a mode string of `"w"` and `"r"` rather than `"wb"` and
`"rb"`. `bench/speed/seq_speed_main.mojo:639` and
`ensemble/bench/rf_bench.mojo:440` are the same two spellings again. There is
no `os.write`, no buffered writer and no seek in this toolchain's surface as
this repository uses it, so **the whole file is built in one `List[UInt8]` and
written once, and read back in one call.** At 161 KB that is the right shape
anyway; at a real model size it is the first thing that would have to change,
and it is named in section 6 item 9 rather than hidden.

### 7.2 The Mojo traps, each one checked

* **`String` is not indexable.** `s[i]` and `s[0:10]` are refused. The magic
  number is therefore a table of eight `UInt8` VALUES and never a string
  literal that gets indexed, and the ASCII names are read back through
  `t[byte=i]` against a charset table -- `core/identity_trace.mojo::_hex16`'s
  proven spelling. `chr()` is not used: it appears nowhere in this repository
  and this lane cannot compile, so an unproven builtin is a risk with no
  upside.
* **`&+` is bitwise AND, not wrapping add.** It produced wrong hashes in this
  repository twice on 2026-08-25 alone. **There is not one `&+` in
  `checkpoint.mojo`.** Every byte assembly uses `|` and `<<`, every arithmetic
  is a plain `+` or `*` on `Int`, and the hash is `fnv1a64_bytes`, which is
  called and not restated.
* **Integer widening sign-extends.** Every field in this format is UNSIGNED
  and every byte read is a `UInt8` widened to `UInt32` or `Int`, which
  zero-extends. **If any of these were `Int8` the top byte of every field with
  its high bit set would read back as `0xFFFFFF..`**, and `n_total` would be
  enormous and the length check would refuse a good file. Unsignedness here is
  load-bearing, not stylistic.
* **`String(float)` does not round trip.** Section 2 -- no float is ever
  formatted into this file, including the hyperparameters.
* **A buffer is freed at LAST USE.** `checkpoint.mojo` holds no
  `DeviceBuffer`, so the device form of the trap cannot arise in it. The HOST
  form can: `fnv1a64_bytes` takes an `UnsafePointer` into a `List[UInt8]`, and
  every call site keeps the owning list alive past the fold with `_ = buf`.
  **The gate does hold device buffers**, downloads three of them per
  checkpoint, and keeps every owner alive past `ctx.synchronize()` with
  `_ = x^` in the same shape `run_training` uses.
* **`fnv1a64_bytes` wants a MUTABLE origin.** DEVIATION 1578: a borrowed
  `List` yields an immutable one and does not unify. `train_loop.mojo` pays
  for this with a `.copy()` per digest. This file avoids the copy by folding
  buffers that are already local `var`s or `mut` arguments, and the one helper
  that folds a range takes `mut` with a comment saying it does not write --
  because `mut` on a reader says something false and the honest fix, an
  immutable-origin overload in `core/identity_trace.mojo`, is OWED and would
  delete a copy at every call site in the tree.

---

## 8. The gate

`training/checks/checkpoint_check.mojo`, five clauses, each naming what it
would catch, because a clause that cannot name its failure is decoration.

**(i) ROUND TRIP, BIT FOR BIT.** Save a fully populated `Checkpoint`, load it,
compare all `3 * n_total` cells by BIT PATTERN and every scalar field. Then
save the loaded copy again and compare the two FILES byte for byte.

The planted values are **distinct per cell AND distinct per array**:
`param[i] = f32_from_bits(0x3F800000 + i)`, `m[i] = f32_from_bits(0xBF800000 +
i)`, `v[i] = f32_from_bits(0x40000000 + i)`. `[[uniform-test-data-hides-
permutation]]` applied to a serializer: **a uniform fill would pass this with
the three arrays written in the wrong order, with `m` and `v` swapped, and with
every offset wrong.** The clause additionally plants `-0.0` and a NaN with a
nonzero payload at known indices, because both are values a format that went
anywhere near a decimal representation would quietly destroy.

Also asserted here: the file's `h_all` equals
`train_loop.mojo::digest_of_lists(param, m, v, offsets).h_all`. That is
section 4's coincidence, checked rather than argued.

And the third `_hex16`. `core/identity_trace.mojo`, `train_loop.mojo` and
`checkpoint.mojo` each carry a private sixteen-digit hex spelling, because
each is private to its module. `train_step_check.mojo` clause (e) already
asserts the first pair on a fixed value; clause (i) asserts the third against
it, so a digest printed by any of the three means the same thing.

**(ii) THE REFUSALS, TEN ARMS.** Each arm takes a file that has just been
demonstrated to load CLEAN, corrupts exactly one thing, and asserts a raise
carrying the RIGHT TAG.

The clean-load precondition is not optional. `optimizer_check.mojo`'s clause
(f) learned this the direct way: if the uncorrupted input already refuses, every
arm below "refuses" whatever it holds and the clause gates nothing. This gate
runs that guard first and RAISES rather than reporting a pass if it trips.

    1   magic         byte 0 flipped                  -> CKPT_REFUSE_MAGIC
    2   version       format_version := 2             -> CKPT_REFUSE_VERSION
    3a  shape         a VALID file with J-1 tensors   -> CKPT_REFUSE_SHAPE
    3b  shape         w_q and w_k counts swapped,
                      n_total unchanged, file
                      internally consistent           -> CKPT_REFUSE_SHAPE
    4   truncated     last 8 bytes dropped            -> CKPT_REFUSE_TRUNCATED
    5   content hash  one payload bit flipped AND
                      h_file repaired                 -> CKPT_REFUSE_CONTENT_HASH
    6   file hash     the same bit, nothing repaired  -> CKPT_REFUSE_FILE_HASH
    7   header        a reserved header field set     -> CKPT_REFUSE_HEADER
    8   trailing      8 extra bytes appended          -> CKPT_REFUSE_TRAILING
    9   layout        entry 1 calls itself param_id 0 -> CKPT_REFUSE_LAYOUT
    10  state         m one element short, at SAVE    -> CKPT_REFUSE_STATE

**Every `CKPT_REFUSE_*` tag section 5 defines fires at least once**, so none
of them is code that has never decided anything.

**Arm 5 repairs the file hash on purpose.** Flipping a payload bit and stopping
there fires `CKPT_REFUSE_FILE_HASH`, which is a correct refusal for the wrong
reason and would leave `h_all`'s check as unreached code
(`[[reached-but-inert]]`). Arm 6 flips the same bit WITHOUT repairing, so both
hashes are demonstrated live and the two findings -- damaged file, different
state -- are shown to be distinguishable.

**Arm 3b is the one no cross-vendor comparison could ever catch.** Two tensors
with swapped counts give a file that is internally consistent, hashes
correctly, and holds plausible in-bounds numbers at wrong offsets --
identically on all three vendors (plan V7).

**Arm 10 is the only refusal that fires before a byte reaches disk.**

**(iii) TRAIN, SAVE, LOAD, RESUME -- the clause this whole lane exists for.**

    whole:  N steps in one process                         -> h_all_whole
    split:  `first` steps, SAVE TO A FILE, then in FRESH
            buffers LOAD THAT FILE, upload param/m/v,
            restore t and the flags, run N - first more    -> h_all_split

and `h_all_whole == h_all_split` is the assertion. This is
`train_step_check.mojo` clause (d) with a filesystem in the middle of it, and
it is the first thing in this repository that tests a resume the plan's
section 4.2 N6 explicitly says is untested.

**Its negative controls, because a resume clause is the easiest place here to
produce a meaningless green.** A resume that ignored the file entirely and
reinitialized from the seed would pass a badly written version of this clause.

    R1  resume with m and v ZEROED       h_all MUST differ
    R2  resume with t restarted at 1     h_all MUST differ, and it must
                                         differ at N = 8 specifically,
                                         because the beta^t spellings agree
                                         through t = 6
    R3  resume with buf_initialized
        all reset to False               EXPECTED BIT-INERT under AdamW and
                                         REPORTED, NOT ASSERTED

**R3 is labelled inert BEFORE it runs, not after it fails to fire.** The flags
are read only on `identical_optimizer_step`'s SGD branch, so under the v1
AdamW configuration dropping them cannot move a bit. An arm whose predicted
answer is "no bits move" is worth having only when it is labelled that way in
advance -- `optimizer_check`'s `OPT_SAB_SCALARS_PER_ELEMENT` made the same
call. R3 becomes a real control the day an SGD configuration is added, and it
is the one that would catch `OPT_SAB_RESUME_REINIT` for real.

**(iv) THE TWO-FILE COMPARISON.** `compare_checkpoint_files` must return `""`
for two files written from equal state, and for a file with one flipped byte
must return a report NAMING THE RIGHT OFFSET and the right section. A
comparator that reported a difference everywhere, or nowhere, would be useless
in exactly the situation it is for.

**(v) DETERMINISM AND NEUTRALITY, as far as a program can check them.** The
same state saved twice in one process must produce byte-identical files; the
file length must be exactly `64 + 64 + 32*J + 12*n_total + 16`; and a save
performed after unrelated device work must equal one performed before it.
**The real neutrality argument is structural and section 3 states it**; this
clause only catches a serializer that picked up a clock, an address or a
counter, which no cross-vendor comparison could distinguish from a divergence.

### 8.1 What the gate does NOT prove

It runs on ONE device. `[[one-box-verdict-is-not-three]]`: a green gate says
the format round-trips and refuses correctly on the box that ran it. **The
cross-vendor claim needs two files from two vendors and the gate cannot
manufacture the second one.** Section 9 is how that run is done and it is
OWED.

---

## 9. The cross-vendor legs, and how to run one

Per leg, on a machine that is NOT the one that built the other leg
(`[[verify-on-a-box-that-did-not-build-it]]`):

    MOJOLEARN_TRAIN_STEPS=8 \
    MOJOLEARN_TRAIN_CKPT_FILE=/tmp/<box>.ckptbin \
      pixi run check-train-checkpoint

then bring both files to one place and

    cmp -l /tmp/apple.ckptbin /tmp/amd.ckptbin && echo IDENTICAL

    MOJOLEARN_CKPT_COMPARE_A=/tmp/apple.ckptbin \
    MOJOLEARN_CKPT_COMPARE_B=/tmp/amd.ckptbin \
      pixi run check-train-checkpoint

`cmp` is the verdict. The compare mode gives the address: the first differing
offset, which section it lands in, and both `h_all`s -- so a difference in the
descriptor (two legs launched with different configuration) is distinguished
from a difference in the payload (a real divergence) without anybody squinting
at hex.

**A leg that runs the write mode and never has its file compared is not a
result**, and neither is a comparison of two files written on the same box.

The stronger leg, once two boxes are up: write at `first = 4` on box A, carry
the file to box B, resume there for four more, and compare the final `h_all`
against a whole eight-step run on either box. **That is "train here, resume
there" and it is the sentence this lane exists to be able to write.** It is
not written yet.

---

## 10. STATUS, OWED, and the exact commands

**WRITTEN, NOT RUN**, all three files, on every column:

| file | Apple | NVIDIA | AMD |
|---|---|---|---|
| `training/CHECKPOINT_FORMAT.md` | -- | -- | -- |
| `training/checkpoint.mojo` | NOT COMPILED | NOT COMPILED | NOT COMPILED |
| `training/checks/checkpoint_check.mojo` | NOT COMPILED | NOT COMPILED | NOT COMPILED |

**RUN OWED.** The orchestrator runs these; this lane runs nothing
(`[[subagents-no-local-tests]]`). In order, cheapest first:

    # 1. does it compile at all
    pixi run mojo build -I . training/checks/checkpoint_check.mojo -o /tmp/ckptchk

    # 2. every clause. (i), (ii), (iv) and (v) are host work; (iii) needs a
    #    DeviceContext, and N = 8 is what reaches t = 7
    MOJOLEARN_TRAIN_STEPS=8 pixi run check-train-checkpoint

    # 3. write a leg's file, on each box
    MOJOLEARN_TRAIN_STEPS=8 MOJOLEARN_TRAIN_CKPT_FILE=/tmp/apple.ckptbin \
      pixi run check-train-checkpoint

    # 4. compare two legs' files, on one box
    cmp -l /tmp/apple.ckptbin /tmp/amd.ckptbin
    MOJOLEARN_CKPT_COMPARE_A=/tmp/apple.ckptbin \
    MOJOLEARN_CKPT_COMPARE_B=/tmp/amd.ckptbin \
      pixi run check-train-checkpoint

The task is `check-train-checkpoint`, added to `pixi.toml` 2026-09-03
alongside `check-train-step` and `check-train-loop`, which
`TRAINING_LOOP_PLAN.md` section 12 item 7 had been owed since 2026-08-25.
**None of the three has been invoked once.** A task is a path with a shorter
name; it is not evidence.

**OWED, and none of it is optional.**

1. **RUN IT.** Nothing here has been compiled. Every "refuses" is a
   prediction.
2. **RUN IT ON TWO VENDORS AND THEN THREE.** One box writing a file it can
   read back is a serializer test. Two boxes writing the same bytes is the
   claim. `[[one-box-verdict-is-not-three]]`.
3. **The genuine "train here, resume there" leg**, section 9's last
   paragraph: save on box A, resume on box B. It needs two leases overlapping
   or a file carried between them, and it is the only thing that closes
   `TRAINING_LOOP_PLAN.md` section 12 item 8 as written.
4. **Wire `train_loop.mojo::main` to write the file**, so that the loop
   itself and not only the gate can produce a checkpoint. This lane did NOT
   touch `train_loop.mojo`: it is a shared checkout
   (`[[mojotrees-shared-checkout-parents]]`) and the training-loop lane owns
   that file. The shape is one call after `run_training` returns, guarded by
   `MOJOLEARN_TRAIN_CKPT_FILE`, alongside the existing
   `write_ckpt_file(MOJOLEARN_TRAIN_CKPT, ...)` sidecar.
5. **A resume ENTRY POINT in `train_loop.mojo`.** `TrainBuffers.__init__`
   always initializes from the seed and `run_training` always starts at
   `t = 1`, so the gate drives its own step loop to resume. That is a FOURTH
   spelling of the loop next to `run_training`, `resume_training` and clause
   (a)'s, and the plan's owed item 11 already says three spellings of one
   thing is three chances to get it wrong. A
   `run_training_from(ctx, cfg, ck, ...)` in `train_loop.mojo` is the real
   fix and it is that lane's call.
6. **Tensor rank and extents in the layout table**, section 6 item 1. A
   transposed weight of the same element count is accepted today.
7. ~~**A `pixi.toml` task.**~~ Added 2026-09-03 -- `check-train-step`,
   `check-train-loop`, `check-train-checkpoint` -- and **never invoked.**
8. **An immutable-origin overload of `fnv1a64_bytes`** in
   `core/identity_trace.mojo`, DEVIATION 1578's real fix, which would delete a
   `.copy()` in `digest_of_lists` and a `mut` in this file's `_fold_bytes`.
9. **A second shape.** `checkpoint.mojo` compiles in no model constant and
   should serve `TRAINING_LOOP_PLAN.md` item 9 unchanged. **That is a
   prediction and it has not been tried.**
