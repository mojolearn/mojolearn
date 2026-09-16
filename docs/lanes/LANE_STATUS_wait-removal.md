# lane/wait-removal

**THE RESULT, IN ONE TABLE.** The same check, the same two commits, three
vendors, eight alternating timed repeats per vendor.

| column | launches before | launches after | waits before | waits after | wall clock, mean |
| --- | ---: | ---: | ---: | ---: | --- |
| Apple M4 (Metal) | 1241 | 1241 | **1887** | **765** | 15.772 s -> 13.211 s, **8 of 8 separated** |
| NVIDIA L40S (CUDA) | 1241 | 1241 | **1887** | **765** | 2.127 s -> 2.083 s, overlapping |
| AMD MI300X gfx942 (HIP) | 1241 | 1241 | **1887** | **765** | 1.078 s -> 1.091 s, overlapping |

Two things follow and the second is the one worth carrying.

1. **THE WAIT COUNT IS A PROPERTY OF OUR CODE, NOT OF THE VENDOR.** 1887
   before and 765 after, to the unit, on Metal, CUDA and HIP, from three
   independently built binaries on three machines. The launch count is 1241
   in all six cells because this lane removed no kernel. A removed wait is
   removed everywhere.
2. **THE PRICE OF A WAIT IS NOT.** Dividing each column's mean time
   difference by the 1122 waits removed gives about **2283 us per wait on
   Apple**, about **38 us on NVIDIA** and a negative number on AMD. Only the
   Apple column separates. So `LAUNCH_BOUND_LANE_CENSUS.md`'s ranking of this
   work was right about WHERE the waits are and right that Apple pays for
   them, and this lane's own contribution is that the same 1122 waits are
   free on the other two columns. **Nobody should expect a CUDA or HIP
   speedup from removing a synchronize, and this lane does not claim one.**

**IDENTITY, CROSS-VENDOR, WHICH IS THE CLAIM THE LIBRARY ACTUALLY MAKES.**
The 37-stage identity card is one sha256 in all four rented cells:

```
a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8
  NVIDIA L40S  before      NVIDIA L40S  after
  AMD MI300X   before      AMD MI300X   after
```

`diff` of the files is empty in every pairing. Two vendors, two arms, one
hash. And the check's own bitwise gate reads `clause (a): PASS, 17 cases,
37/37 stages bit-identical to the oracle on all 412172 cells` in all six
instrumented arms.

## 0. What was changed, in one table

| file | waits before | waits after | what went |
| --- | ---: | ---: | --- |
| `core/device_scan.mojo` | 13 | 5 | 8 allocation and mid-scan waits |
| `gemm/checks/gemm_identical.mojo` | 29 | 26 | 3 workspace-allocation waits |
| `transformer/impl/llama/modeling_llama.mojo` | 32 | 20 | 12 waits in front of a trace record |
| `transformer/checks/transformer_backward.mojo` | 37 | 23 | 14 waits in front of a trace record |
| **total static call sites** | **111** | **74** | **37** |

No kernel body, no grid, no block size, no accumulation order and no
allocation size was touched. **Nothing in this diff can move a bit**, and
section 4 is the gate that says so rather than the argument.

That claim is mechanically checkable rather than asserted. Take the whole
diff, drop the file headers, drop every line that is a comment, and drop
every line that is a `ctx.synchronize()` or its `step_count_sync()`
companion. **What is left is two docstring sentences.**

    git diff 438fb4529..HEAD | grep -E "^[+-]" | grep -vE "^(\+\+\+|---)" \
      | grep -vE "^[+-]\s*#" | grep -vE "ctx\.synchronize\(\)|step_count_sync\(\)"

    -> 12 lines, all of them prose in two docstrings of core/device_scan.mojo

    removed `ctx.synchronize()` lines: 37
    added   `ctx.synchronize()` lines:  0

Printing the surviving lines rather than their count is deliberate: a grep
that returns nothing because the pattern was wrong is indistinguishable from
a grep that returns nothing because the code is clean.

## 1. The census of the step path's waits, by cause

Every `ctx.synchronize()` reachable from one instrumented byte LM / decoder
step, sorted into the four categories, from the source at `438fb4529`.

### (a) the host must read a device value to decide CONTROL FLOW

These are the only structurally required waits on the path. All of them sit
immediately before a host read whose value picks a branch or a refusal.

| file:line (at 438fb4529) | what it feeds |
| --- | --- |
| `core/device_scan.mojo:209` | `_fold_partials` -> `device_first_nonfinite`'s return -> refuse or not |
| `core/device_scan.mojo:247` | the same for `device_first_negative` |
| `core/device_scan.mojo:271` | `device_classify_nonfinite` picks "NaN in" or "infinity in" |
| `core/device_scan.mojo:308` | `DeviceScanScratch._finish`'s fold |
| `transformer/impl/llama/modeling_llama.mojo:2583` | `_refuse_nonfinite_device` reads the offending element to name it |
| `transformer/impl/llama/fused_attention.mojo:5546` | `_read_flag` reads the attention corner flag |
| `training/checks/loss.mojo:1313` region | `device_first_nonfinite` over the logits |
| `training/checks/optimizer.mojo:1256` region | the optimizer's four scans |

**KEPT, every one.** What CAN be done to this category is to make the scan
cheaper, not to delete the wait: each scan is already one kernel plus one
2 KB download, and after this lane it is one wait rather than four. The
remaining structural cost is one host round trip per scanned buffer, and
moving it to one flag per STEP is section 7 item 1.

### (b) the host reads a value only to REPORT it

| file:line | what |
| --- | --- |
| `training/byte_lm.mojo:938` | `download_f32(ce_loss, 1)`, four bytes, one round trip, every step, for a loss the caller usually prints |
| `transformer/impl/llama/modeling_llama.mojo:747, :756` | `_download`, the export and stateless paths only |
| `core/identity_trace.mojo:356` | the trace hash, and a traced run is not a measurement by that file's own rule 4 |

**NOT TAKEN HERE.** `byte_lm.mojo:938` is the prior lane's ranked item 3 and
deferring it changes a public interface (the step would return the previous
step's loss). That is a decision about the API, not about a wait, and it
belongs to whoever owns the API. It is one round trip per step against the
2000 this path spends.

### (c) buffer lifetime or aliasing, a wait standing in for ordering

| file:line | why it is real |
| --- | --- |
| `gemm/checks/gemm_identical.mojo:6499` | `_ = ws` is the workspace's LAST USE, so the runtime may release it there. The kernel may still be reading it. **Sabotaged in section 5.** |
| `gemm/checks/gemm_identical.mojo:4131`, `:6116` | the same for `_ = gws` |
| `transformer/impl/llama/modeling_llama.mojo:733`, `:845` | `_upload` and `_plant_bits` keep a staging host buffer alive past the copy |
| `transformer/impl/llama/fused_attention.mojo` `_launch_bwd_*` trailing waits | the attention scratch stashes, `_ = y_st^` and friends |

**KEPT, every one.** `[[mojo-buffer-freed-at-last-use]]` is a real rule in
this repository and these waits enforce it. The right fix is a
caller-owned workspace, not a deleted wait; section 7 item 2.

### (d) defensive, no reason discoverable -- ALL REMOVED

Three shapes, 37 static sites.

**(d1) A wait between an `enqueue_create_buffer` and the launch that
consumes it.** `core/device_scan.mojo:191`, `:229`; `gemm_identical.mojo:6496`,
`:4125`, `:6107`. The allocation and the kernel are submitted to ONE in-order
`DeviceContext` in program order, and the host never touches the buffer. The
repository already ships this without the wait in `core/gram_splitk.mojo:1029-1030`
and in `gemm_identical._kpack_run`.

**(d2) A wait between a kernel and a copy of its output, or between a host
allocation and its copy.** `core/device_scan.mojo:201`, `:205`, `:239`,
`:243`, `:267`, `:303`. Same in-order argument; the copy cannot start before
the kernel it follows. `arima/impl/batched_arima.mojo:194-196` has shipped
exactly this shape, host allocation followed straight by its copy and ONE
wait, since before `device_scan.mojo` existed.

**(d3) A wait whose next statement is a trace record.** Twelve sites in
`modeling_llama.mojo` and fourteen in `transformer_backward.mojo` (through
`_rec`, a one-line wrapper). `IdentityTrace.record_device`
(`core/identity_trace.mojo:315`) returns at once when tracing is off, and
when it is on it enqueues its OWN copy behind the kernel and drains AFTER
that copy. Neither branch reads the device before a drain the function
performs itself.

**This is the only one of the three shapes that was costing real time.**
Every (d3) site had a kernel pending behind it. Every (d1) and (d2) site had
at most a buffer allocation.

## 2. The refusal paths, which the brief ranked first

`fwd.refuse_call` recorded 34 launches and **136** waits, `grad.refuse_scan`
17 and **68**: four waits per kernel, the worst ratio in the census, and both
are validation rather than arithmetic.

The cause is not `_refuse_nonfinite_device`'s two waits at
`modeling_llama.mojo:2579` and `:2583`. Those run ONLY when a non-finite has
already been found; on a healthy step they execute zero times. The four
waits per launch are all inside `core/device_scan.device_first_nonfinite`,
which ran

    alloc, WAIT, kernel, WAIT, host alloc, WAIT, copy, WAIT, fold

for ONE kernel. It now runs

    alloc, kernel, host alloc, copy, WAIT, fold

**The check is not weakened.** The kernel, its predicate
(`(bits & 0x7FFFFFFF) >= 0x7F800000`, by bits), its geometry (256 threads,
at most 512 blocks), the block-level halving fold, the host minimum over the
partials and the message text are all untouched. The same input produces the
same index and the same refusal at the same flat index. **The early-exit
behavior is unchanged too**: `device_first_nonfinite` still returns before
any wait when `n <= 0`, and `_refuse_nonfinite_device` still returns without
allocating or copying anything when the scan reports no hit.

Measured, instrumented backward check, 17 cases:

| site | waits before | waits after |
| --- | ---: | ---: |
| `fwd.refuse_call` | 136 | **34** |
| `grad.refuse_scan` | 68 | **17** |

One wait per scan, down from four.

## 3. The GEMM's two waits per launch

`identical_gemm` (`gemm/checks/gemm_identical.mojo:6451`) is the host-visible
entry point every projection, every MLP matmul and every weight gradient
goes through. It ran

    nws = workspace_max_floats(m, n, k)
    ws  = enqueue_create_buffer(nws)
    WAIT                                  <- removed
    identical_gemm_into(...)              <- one launch, asynchronous
    WAIT                                  <- kept
    _ = ws

Exactly 2.0 waits per launch, 782 of the census's 1887, and those two lines
were both of them.

**The first is gone.** It waited for a buffer allocation that the very next
submission on the same in-order context consumes. The host never reads `ws`.

**The second stays and section 5 shows why.** `_ = ws` is the workspace's
last use, so the runtime is free to release it there, and the kernel may
still be writing it. The function's own docstring says this and it is right.

Measured: the `gemm.*` overlay fell from 782 waits to 391, exactly one per
launch. The partition fell by 595, because 204 further `identical_gemm`
calls per run sit inside `bwd.mlp_through_oproj` and carry no `gemm.*` tag;
the census's 391 was therefore a floor on this change, not its size.

**What was NOT taken.** The remaining 595 waits are the workspace lifetime.
They can be removed entirely by hoisting the workspace to the caller and
calling `identical_gemm_into`, which the docstring already offers and which
`llama_decoder_layer_forward`'s stage struct is the natural home for. That
is a change to about forty call sites and it is not this lane's, because it
is where a wrong workspace size stops being a wait and starts being a wrong
answer. Section 7 item 2.

## 4. Identity evidence, on rented NVIDIA and AMD

### 4.1 The cross-vendor card, which is the claim the library makes

`transformer_backward_check` writes a 37-stage identity card when
`MOJOLEARN_IDENTITY_TRACE` names a file, one sha256 per backward stage.
Four cards were taken, two vendors by two arms, each by the binary whose
sha256 is printed beside it in `BINARY_DIGESTS.txt` and taken in the same
shell that ran it.

| cell | GPU | arm | card sha256 |
| --- | --- | --- | --- |
| `WRLEG-658f279e2374-nvidia` | NVIDIA L40S | before | `a4166b19a0af95ec...` |
| `WRLEG-658f279e2374-nvidia` | NVIDIA L40S | after | `a4166b19a0af95ec...` |
| `WRLEG-658f279e2374-amd` | AMD MI300X gfx942 | before | `a4166b19a0af95ec...` |
| `WRLEG-658f279e2374-amd` | AMD MI300X gfx942 | after | `a4166b19a0af95ec...` |

`diff` between any two of the four files is EMPTY, not merely equal in
digest. **One card, two vendors, two arms.** The before-against-after
pairing says this lane moved no bit; the NVIDIA-against-AMD pairing says the
library's cross-vendor claim still holds with 1122 waits gone.

### 4.2 The bitwise gate against the host oracle, six arms

Every arm on both boxes, instrumented and plain, printed

```
clause (a): PASS, 17 cases, 37/37 stages bit-identical to the oracle
            on all 412172 cells
```

Seventeen fixture cases span `B` 1 to 3, `L` 1 to 16, `d_model` 32 and 48,
`inter` 64 and 300, `n_kv_heads` 1 and 2, and an odd head dim of 24.

### 4.3 What this replaces, and what was thrown away

This lane was redirected off Apple partway through. **What was discarded is
one `tools/identity_break.py --fixtures base` pair on the byte-lm and
transformer lanes**, which had both arms' bindings built
(`_mojolearn_byte_lm.so` and `_mojolearn_transformer.so`, four `.so` files in
all) and was sitting in the Metal FIFO when the redirect arrived. The ticket
was released rather than run. That comparison would have been Apple against
itself; section 4.1 is NVIDIA against AMD, which is strictly the stronger
statement, so nothing that mattered was lost. The four bindings were built
and are not needed.

**The base commit differs between the two legs and it does not matter.** The
NVIDIA leg shipped `438fb4529` and `438fb4529 + this lane`; the AMD leg, run
after a rebase, shipped `0fb6cf985` (origin/main, 120 commits later) and
`0fb6cf985 + this lane`. Those 120 commits touch **zero** files under
`transformer/`, `gemm/` or `core/`:

    git diff --name-only 438fb4529..0fb6cf985 -- 'transformer/**' 'gemm/**' 'core/**'
    -> 0 files

which is why the two legs' cards are the same bytes. Had that command
printed anything, the cross-vendor diff in 4.1 would have been worthless and
the NVIDIA leg would have had to be re-run at the new base.

## 5. The sabotage, seen to fail on both rented vendors

A check that has never been observed to fail is not evidence. **The wait this
lane KEPT** in `core/device_scan.device_first_nonfinite`, the one between the
device-to-host copy and `_fold_partials`, was deleted on purpose in a third
tree built beside the other two on each box.

**The failure was predicted from the source before either box existed.** The
host folds a pinned buffer the copy has not yet written. `_fold_partials`
takes the MINIMUM over the partials against `NONFINITE_NONE = 2147483647`, so
whatever that memory happens to hold is almost certainly smaller, the fold
returns an index on a buffer that holds no non-finite at all, and the refusal
path fires on a finite value.

| box | arm | rc | `clause (a): PASS` | what it said |
| --- | --- | ---: | ---: | --- |
| NVIDIA L40S | SABOTAGE rep 1 | **1** | **0** | `offset, len and elem_size are out of range` at `modeling_llama.mojo:2575` |
| NVIDIA L40S | SABOTAGE rep 2 | **1** | **0** | the same |
| NVIDIA L40S | clean rep 1, rep 2 | 0 | 1 | PASS |
| AMD MI300X | SABOTAGE rep 1 | **1** | **0** | the same message, same line |
| AMD MI300X | SABOTAGE rep 2 | **1** | **0** | the same |
| AMD MI300X | clean rep 1, rep 2 | 0 | 1 | PASS |

**Four sabotage runs, four hard failures, zero reaching `clause (a)`. Four
clean runs beside them on the same two boxes, four passes.** `:2575` is
`buf.create_sub_buffer(idx, 1)` inside `_refuse_nonfinite_device`, reached
with the bogus `idx` the unsynchronized fold returned.

**The symptom is vendor-shaped and the failure is not.** The same sabotage
run earlier on the M4 said `llama: infinity in input_layernorm.weight at flat
index 0 REFUSED (row 39)`, because there the un-copied pinned buffer read as
zeros and zero folds to index 0, which is in range. On CUDA and HIP it held
something large and the sub-buffer refused first. Three vendors, three
deterministic failures, two different messages, one cause. Anyone reading
only one box's message would have called it a bounds bug.

**And the wait that was NOT load-bearing was sabotaged too.** Deleting
`identical_gemm`'s trailing wait before `_ = ws` (category (c), section 1)
PASSED twice on the M4. At the shapes this gate runs, `choose_gemm_plan`
picks a TILE or TUNED plan and `identical_gemm_workspace_floats` is zero for
those, so the workspace is allocated, handed over as a pointer and never
dereferenced. **That is a race that happens not to fire, which is exactly why
the wait was kept rather than removed.** A shape whose `contract_partition`
gives `P >= 4` takes a split plan and materializes `m * n * P` partials in
that workspace, and then the same deletion writes the answer from freed
memory. Section 7 item 2 removes it by giving the buffer an owner, never by
deleting the wait.

## 6. Timing, with the prediction stated first

### 6.1 What to expect, computed before any box was rented

A wait costs about 4.15 ms on Metal when the queue has pending work and
**12.7 us when it does not** (`LANE_STATUS_lane-metal-launch-overhead.md`
section 1, `sync_only`). The value of a removed wait is therefore not 4.15 ms
flat; it is 4.15 ms only if something was enqueued since the previous wait.
Classifying the 1122 removed waits by that test:

| removed | what was pending behind it | count | each (Metal) | total |
| --- | --- | ---: | ---: | ---: |
| (d3) trace-record waits | a kernel | **374** | 4.15 ms | 1552 ms |
| (d2) the scan's post-kernel wait | a kernel | **51** | 4.15 ms | 212 ms |
| (d1) workspace-allocation waits | an allocation | 595 | 12.7 us | 8 ms |
| (d2) the scan's other waits | nothing | 102 | 12.7 us | 1 ms |
| | | **1122** | | **1773 ms** |

**Predicted: about 1.8 s on Metal, and nothing measurable on CUDA or HIP**,
because neither of those prices a host round trip anywhere near 4 ms. Both
halves of that prediction were recorded before the runs below. The flat model
would have said `1122 * 4.15 ms = 4.7 s` on every column, which is 2.6x too
high on Apple and infinitely too high on the other two.

### 6.2 The counts, measured on three vendors

Built with `-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1`
and run under `MOJOLEARN_TRANSFORMER_TIMING=1`. The partition is
`fwd.* + grad.* + bwd.*`; `gemm.*` re-reports the same intervals by GEMM kind
and is shown separately.

| column | arm | launches | waits | gemm launches | gemm waits |
| --- | --- | ---: | ---: | ---: | ---: |
| Apple M4 | before | 1241 | 1887 | 391 | 782 |
| Apple M4 | after | 1241 | **765** | 391 | **391** |
| NVIDIA L40S | before | 1241 | 1887 | 391 | 782 |
| NVIDIA L40S | after | 1241 | **765** | 391 | **391** |
| AMD MI300X | before | 1241 | 1887 | 391 | 782 |
| AMD MI300X | after | 1241 | **765** | 391 | **391** |

**Six cells, three vendors, three separately compiled pairs of binaries, and
every number agrees to the unit.** The BEFORE rows also reproduce
`LAUNCH_BOUND_LANE_CENSUS.md` section 0.3's 1241 and 1887 exactly, which were
taken on a different day on the Mac, so all three columns are measuring what
the census measured.

Per site, first commit only, and every line is accounted for:

| site | before | after | why |
| --- | ---: | ---: | --- |
| `fwd.refuse_call` | 136 | 34 | four waits per scan became one |
| `grad.refuse_scan` | 68 | 17 | the same |
| seven `fwd.*_proj` | 238 | 119 | one `identical_gemm` allocation wait each |
| sixteen `grad.*_d[AB]`, `grad.norm*_dW` | 544 | 272 | the same |
| `bwd.mlp_through_oproj` | 527 | 323 | 204 further `identical_gemm` calls |
| **subtotal** | **1887** | **1139** | 748 removed by commit 1 |
| the 26 trace-record sites, commit 2 | 1139 | **765** | 374 more |

The census's `gemm.*` overlay counts 391 of those GEMM calls; 204 more sit
inside `bwd.mlp_through_oproj` with no `gemm.*` tag, so the overlay's 782 was
a floor on this change, not its size.

### 6.3 Wall clock, eight alternating repeats per vendor

Un-instrumented binaries, the shipped shape. Both arms warmed once and the
warm-ups discarded, then BEFORE and AFTER alternating inside one hold so the
two arms see the same machine minute by minute. Seconds.

| | Apple M4 | | NVIDIA L40S | | AMD MI300X | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat | before | after | before | after | before | after |
| 1 | 14.461 | 12.360 | 2.158 | 2.058 | 1.070 | 1.079 |
| 2 | 14.398 | 13.109 | 2.125 | 2.174 | 1.042 | 1.107 |
| 3 | 16.323 | 13.033 | 2.109 | 2.009 | 1.060 | 1.085 |
| 4 | 15.253 | 13.600 | 2.190 | 2.139 | 1.067 | 1.163 |
| 5 | 16.333 | 12.643 | 2.095 | 2.161 | 1.135 | 1.137 |
| 6 | 15.622 | 13.476 | 2.209 | 2.046 | 1.071 | 1.045 |
| 7 | 18.002 | 13.763 | 2.049 | 2.058 | 1.071 | 1.043 |
| 8 | 15.785 | 13.701 | 2.078 | 2.022 | 1.108 | 1.070 |
| **mean** | 15.772 | **13.211** | 2.127 | 2.083 | 1.078 | 1.091 |
| **median** | 15.704 | 13.293 | 2.117 | 2.058 | 1.071 | 1.082 |
| sd | 1.164 | 0.515 | 0.055 | 0.065 | 0.029 | 0.042 |
| **samples separated?** | **YES, 8 of 8** | | no | | no | |
| mean delta | **-2.561 s (-16.2%)** | | -0.043 s (-2.0%) | | +0.013 s (+1.2%) | |

**Apple.** Every AFTER repeat is faster than every BEFORE repeat: the slowest
AFTER, 13.763 s, is below the fastest BEFORE, 14.398 s. For complete
separation of two samples of eight the exact two-sided permutation
probability is `2 / C(16,8) = 1.55e-4`. Predicted 1.77 s, measured 2.04 s min
to min and 2.56 s mean to mean, so the prediction is low by 15 to 45 percent,
which is the right direction: the 4.15 ms constant was measured on a quiet
box and this one sat at load average 14.

**NVIDIA and AMD.** Neither separates. The NVIDIA arms differ by 0.043 s with
a 0.06 s spread and the AMD arms differ by 0.013 s in the WRONG direction.
Dividing by the 1122 removed waits implies about 38 us per wait on NVIDIA and
a negative number on AMD, against 2283 us on Apple. **A `cudaDeviceSynchronize`
or `hipDeviceSynchronize` is cheap and a Metal submit-and-wait is not, and
that, not the wait count, is what differs between the columns.** This lane
claims no NVIDIA or AMD speedup and none should be read into it.

**THE APPLE NUMBERS WERE TAKEN BEFORE THE REDIRECT TO RENTED BOXES AND WERE
NOT RE-TAKEN.** They are kept because they are the only column where this
change is worth wall-clock time at all, and dropping the one measurement that
shows a cost would misrepresent the work in the flattering direction. They
are one Mac against itself on a contended box.

**A retraction avoided, and how.** An earlier four-repeat Apple pair of the
same binaries read BEFORE 13.491, 15.570, 25.232, 18.997 against AFTER
14.607, 14.228, 17.683, 16.108: overlapping, with the AFTER minimum SLOWER.
That pair warmed only the BEFORE binary, so AFTER's first repeat paid a cold
binary's one-time cost, and one BEFORE repeat hit a 25 s contention spike.
Both arms are warmed in the table above. The bad pair is printed here rather
than dropped, because dropping it is how a lane talks itself into a number.

### 6.4 The boxes, and what they cost

| leg | GPU | host | image | wall | teardown |
| --- | --- | --- | --- | --- | --- |
| `WRLEG-658f279e2374-nvidia` | NVIDIA L40S 46 GB, driver 580.159.03 | EPYC 9354, 128 cores | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | ~19 min | `DELETE -> HTTP 204`, `GET -> HTTP 404` |
| `WRLEG-658f279e2374-amd` | AMD Instinct MI300X, gfx942 | EPYC 9474F, 192 cores | `rocm/dev-ubuntu-22.04:6.4.1-complete` | ~8 min | `DELETE -> HTTP 204`, `GET -> HTTP 404` |

Both verified gone, and a subsequent `GET /v1/pods` on the whole account
returned **zero pods**. About $1.10 of the $10 authorized, including two
failed AMD creates described in section 6.5.

Each phase uploaded its results to R2 the moment it finished, under
`legs/wait-removal-2026-09-16/WRLEG-658f279e2374-<vendor>-<phase>.tgz`, four
per leg, every one `HTTP 200`. The token `WRLEG-658f279e2374` is written into
`box.txt`, `TOKEN.txt`, `timing.txt` and every counts header, so a later
reader cannot be served a previous leg's object and believe it.

### 6.5 Two failed AMD creates, recorded because they cost money

The first MI300X create succeeded and then died on the ready timeout: a plain
`rocm/dev-ubuntu-22.04` image carries no sshd and no RunPod start script, so
ssh never answered and `attn_pod.sh` deleted the box after ten minutes, about
$0.42 for nothing. `tools/gemm_remote_leg.sh` already solves this by passing
`tools/runpod_ssh_bootstrap.sh` as the container start command; a scratch copy
of `attn_pod.sh` now does the same. The second create then reached the arming
step and refused because that scratch copy computed `HERE` from its own path
and could not find `tools/runpod_guard.sh`; **the guard refusing rather than
proceeding unarmed is the guard working**, and the box was deleted in under a
minute. Neither change is in this lane's diff; the scratch runner is
`scratchpad/leg/attn_pod_amd.sh` and the two fixes it carries are worth
folding into `tools/attn_pod.sh` by whoever owns it.

## 7. What is still owed, ranked

1. **The refusal scan's remaining round trip, about 51 per run here.** Each
   scanned buffer still costs one host round trip. The shape that removes
   them is one DEVICE flag per step: every scan writes into one `Int32` with
   an atomic minimum, and the host reads that single flag ONCE at the end of
   the step. The refusal message needs the flat index and the offending
   value, and both can be carried in the flag plus one 4 B read taken ONLY
   when the flag is set, so the healthy path pays one round trip per step
   instead of one per buffer. This does not weaken the check and it moves no
   arithmetic.
2. **The GEMM workspace's owner, 595 waits.** `identical_gemm_into` already
   exists and is asynchronous. Giving `LlamaStages` and the backward stage
   struct a workspace field sized by `identical_gemm_workspace_max_floats`
   over the shapes the block uses, and calling `identical_gemm_into`,
   deletes every one of them. Section 5.2 shows the wait is not currently
   load-bearing at these shapes and WOULD be at a split-plan shape, so this
   must be done by giving the buffer an owner and not by deleting the wait.
3. **`training/byte_lm.mojo:938`**, the four-byte loss download, one round
   trip per step. Interface decision, not a wait decision.
4. **The elementwise waits that are NOT in front of a trace record.** The
   (d3) rule caught 26 sites. `fused_attention.mojo` has about twenty more
   waits whose next statement is `_attn_tick` or another launch, and
   `transformer_backward.mojo` has several whose next statement is
   `identical_gemm(`. Each needs reading individually, which is why none was
   taken mechanically here.
5. **NOT TAKEN AND NOT TO BE TAKEN UNDER THIS LANE: arithmetic fusion.**
   Contracting a multiply and an add into an FMA moves bits. Nothing in this
   diff fuses two float operations, and nothing in it should.

## 8. House rules honored

**Verification ran on rented NVIDIA and AMD, not on the shared Mac.** No
Metal lock was taken after the redirect; the one queue ticket this lane held
was released without ever acquiring the lock, and the job that held the lock
at the time (another agent's, pid 73183) was not touched. The Apple numbers
in section 6.3 were taken before the redirect and were not re-taken.

Both boxes were created with an armed lease and a dead-man before any work,
terminated explicitly, and verified gone twice over: `DELETE -> HTTP 204`,
`GET -> HTTP 404`, and then `GET /v1/pods` for the whole account returning
zero pods. Every phase uploaded to R2 as it finished, with a distinctive
token in every record.

Own worktree, rebased onto `origin/main` at `0fb6cf985` before the AMD leg
shipped. The shared checkout was never edited, nothing was stashed, `git add
-A` was never used, and the sabotage was applied to a COPY of the tree on the
box (`/root/src-sab`) rather than to any tracked file, so no restore was
needed. `git rev-parse --abbrev-ref HEAD` was read before every commit.
Nothing was pushed and nothing was merged.

Every binary digest in this file was taken by `sha256sum` in the same shell
that executed the binary, immediately before the exec, and is recorded beside
its result. The two arms of every comparison were built minutes apart on ONE
box from ONE toolchain, so the only thing that moved between them is this
lane's two commits.

Raw logs, both legs, every card and every digest:
`~/mojolearn-evidence/wait-removal-2026-09-16/`. R2:
`legs/wait-removal-2026-09-16/WRLEG-658f279e2374-*`.
