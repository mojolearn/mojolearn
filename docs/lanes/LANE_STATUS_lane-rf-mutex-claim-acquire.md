# lane/rf-mutex-claim-acquire

**2026-09-16. The repair for the MI300X random forest nondeterminism, with the memory-model
argument written out, applied to every site that shares the primitive. Cut from `main`
9257fd6de. NOT MERGED. Andrew wants this reconciled with Codex's work on the same defect
before anything reaches `main`.**

The defect itself, its history, the twelve MI300X legs and the trace evidence live in
`lane/rf-score-weighted-nondeterminism` (`docs/lanes/LANE_STATUS_lane-rf-score-weighted-
nondeterminism.md` and `docs/lanes/LANE_STATUS_MECHANISM.md` on that branch). This file
answers the five questions the addendum asked, names the repair, and carries the evidence
for it.

> **2026-09-16 12:47. MAIN IS BROKEN ON APPLE.** `01c228a72` merged the
> acquire FENCE into `core/device_mutex.mojo` at 12:07. Measured on main's own
> binary at `c173a6b4f`, a 3-tree random forest fit fails at Metal pipeline
> creation; the same commit with the fence compiled out fits fine. See "MAIN IS
> BROKEN ON APPLE AS OF `c173a6b4f`". The spelling this branch carries is the
> fix.

> **2026-09-16, second pass. THE REPAIR NOW EMITS, AND IT IS NOT THE
> SPELLING THIS PASS FIRST CHOSE.** The original repair read
> `_ = Atomic.load[ordering = Ordering.ACQUIRE](mutex)`, and an atomic load
> whose result is discarded is deleted by the compiler. That was
> re-established in one command before anything was built on it. The obvious
> replacement, a standalone `fence[ordering = Ordering.ACQUIRE]()`, emits on
> all three columns and is the textbook translation of cuVS's `atomicCAS`
> plus `__threadfence()`, and IT CANNOT RUN ON APPLE. It compiles, it
> disassembles, and it then crashes the Apple GPU machine code generator at
> pipeline creation on every launch. The shipped repair is therefore a
> post-claim acquire LOAD whose value is consumed by the exit condition of a
> one-iteration loop. It emits on all three columns, it is in the shipped
> artifact, and it runs on Apple. What is still NOT demonstrated is that the
> hole is CLOSED on MI300X. Nothing in this lane has run on AMD.

## The repair, in one paragraph

Every cross-block mutex in this repository spun on an ACQUIRE load and then took the lock
with a weak RELAXED compare-exchange. Those are two separate reads of the mutex, and only
the first carried an acquire. The repair adds one post-claim ACQUIRE LOAD of the mutex,
written so that its value is CONSUMED. Nothing else changes.

    while Atomic.load[ordering = Ordering.ACQUIRE](mutex) != Int32(1):
        pass

That load reads the value the claim itself wrote. A compare-exchange is a read-modify-write,
so its write sits in the release sequence headed by whichever release store the claim
consumed, and an acquire load that reads a value in a release sequence synchronizes with
the release that heads it (C++ [atomics.order], the release-sequence rule). The edge is made
with the release the lock was ACTUALLY taken against.

THE LOOP RUNS EXACTLY ONE ITERATION and is not a wait. This thread's own claim wrote the
value being compared against, a thread cannot read a value earlier in the modification order
than one it wrote itself, and only the holder writes anything else. The loop exists because
the exit condition is what consumes the loaded value; `_ =` does not, and the compiler
deletes the load.

THE CONSTANT IS PER SITE, not always 1. The kNN consumer claims by exchanging `-2` for `-1`,
so its loop waits for `-1`. Each of the six loops was checked against the third argument of
its own `compare_exchange`; a loop waiting for the wrong value would never exit.

| site | claim | repaired |
|---|---|---|
| `ensemble/decisiontree/batched_levelalgo/split.mojo` `_publish_to_global` | 1 | yes, with the control define |
| `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` block publish | 1 | yes |
| `neighbors/impl/detail/fused_l2_knn.mojo` consumer and producer | 2 | yes |
| `neighbors/mutex_probe_main.mojo` consumer and producer | 2 | yes |

Every one of the six is the post-claim acquire loop as of this pass, and each waits for its
own claim's value. The rf publish kernels were read out of the shipped artifact and carry
TWO acquire loads each where the stock arm carries one.

`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` compiles the pre-repair claim in the forest binding.
It exists only so the A/B leg has a control that must be SEEN to move. It is not a default
and not a matrix row.

## 1. Is the memory-model reasoning correct? YES

The claim under test is that a relaxed compare-exchange taking the lock, preceded by a
separate acquire load, does not establish the synchronizes-with edge that an acquire
read-modify-write would. That is correct, and it follows from the definition rather than
from hardware folklore.

An acquire operation synchronizes with the release operation whose value it reads (or a
value in that release's release sequence). The spin load is an acquire, so it synchronizes
with whichever release wrote the zero it read. The compare-exchange is relaxed, so it
synchronizes with nothing, whatever value it reads. The two reads need not see the same
release. Concretely, with holders A, B and contender C on one node's mutex:

1. A releases (store-release of 0, call it R1).
2. C's acquire spin load reads R1's zero. C now synchronizes with A, not with anyone later.
3. B's claim consumes R1's zero first. C's claim on that zero fails and C spins again, OR
   C's claim is simply late.
4. B loads `split[node]`, merges, stores `split[node]`, releases (R2).
5. C's relaxed claim consumes R2's zero and succeeds.

C holds the lock without any acquire that reads R2. B's plain store of `split[node]` is
sequenced before R2, but R2 does not happen-before anything C does, so B's store and C's
plain load of `split[node]` are unordered. In the C++ model that is a data race with
undefined behavior. In hardware terms C may be served a copy of the slot that predates B's
store. C then merges its own candidate into that stale copy and writes the whole struct
back, and B's candidate is gone.

The only caveat to the addendum's wording is the sentence "an RMW must read the latest
value in coherence order". A read-modify-write reads the last value in the modification
order immediately before its own write, which is the property that puts it in the release
sequence. That is exactly why a plain acquire load AFTER the claim repairs it (section 3).

Why MI300X and not the other columns. Each of the MI300X's eight XCDs has its own L2, and
a plain load on one XCD can be served from a line that XCD cached before another XCD's
holder wrote the slot back. The acquire load at the spin invalidates that XCD's caches, but
it does so BEFORE B's store, and nothing invalidates them again between the spin and the
claim. A part with one coherent L2 has the same formal hole with a far smaller window,
which is consistent with the NVIDIA and Apple columns never having shown a move without
proving they cannot.

## 2. Does the fix close the hole, or narrow it? It CLOSES it

The failing step is "C holds the lock with no acquire that reads a value written by or
after R2". After the repair, C performs an acquire load after its claim succeeds. That load
reads the value C's own claim wrote. A compare-exchange is a read-modify-write, and the
release sequence headed by R2 is the maximal contiguous run of read-modify-writes that
follow R2 in the modification order of the mutex, so C's claim is in R2's release sequence.
An acquire that reads any value in a release sequence synchronizes with the release that
heads it. R2 therefore synchronizes with C's post-claim load, and B's store of `split[node]`
happens-before C's load of it. There is no remaining path by which C can hold the lock
without having synchronized with the release it claimed against. The window is not
narrower; it does not exist.

THE ARGUMENT WAS NEVER THE PROBLEM. THE SPELLING WAS, TWICE. The reasoning above was
already written on 2026-09-16 for `_ = Atomic.load[ACQUIRE](mutex)`, and it was correct;
that line simply emitted nothing. An acquire fence reaches the same edge by a different
rule, C++ [atomics.fences], with the relaxed claim as the operation sequenced before the
fence that reads the release; that was correct too, and it cannot run on Apple. What is
claimed here is claimed about the loop, which is in the artifact, was read out of it, and
has launched on Metal.

Two things the repair does NOT rely on. It does not need the spin's acquire load, which
stays only because removing it would change more text than the defect requires. It does
not need any vendor-specific instruction. The acquire load is the same one the spin already
used on every column, so it legalizes on Apple, where the compiler rejects an acquire
compare-exchange by name.

## 3. Is there a better-shaped fix? The three alternatives, evaluated

**Acquire success ordering on the compare-exchange.** Formally equivalent to the repair,
and the direct translation of the reference's `atomicCAS` plus `__threadfence`. RE-CHECKED
AGAINST THE SHIPPED TOOLCHAIN on 2026-09-16 rather than inherited, because if it had gone
stale it would have been the cleanest fix of all, needing no extra operation. It has not
gone stale. The compiler still says, by name:

    oss/modular/mojo/stdlib/std/atomic/atomic.mojo:758:10: error: Apple GPU does not
    support `acquire` atomic ordering
      ... "pop.atomic.cmpxchg" ... success_ordering = #pop<atomic_ordering acquire> ...

All four of DEVIATION 106's Apple refusals were re-compiled rather than inherited, and all
four still fire, with these messages:

| spelling | verdict on the shipped toolchain |
|---|---|
| `Atomic.compare_exchange[success_ordering = ACQUIRE]` | `error: Apple GPU does not support `acquire` atomic ordering` |
| `Atomic.compare_exchange[weak=False]` | `error: Apple GPU only supports `weak` compare-exchange; AIR exposes no strong compare-exchange primitive` |
| `std.gpu.intrinsics.threadfence()` | `note: constraint failed: threadfence is only implemented on NVIDIA GPUs` (`intrinsics.mojo:790:5`) |
| `Atomic.fetch_add[ordering = ACQUIRE]` | `error: Apple GPU does not support `acquire` atomic ordering`, on `pop.atomic.rmw` |

THE LAST ROW WAS NOT PREVIOUSLY RECORDED, and it closes off a whole family. There is no
acquire read-modify-write of any operation on Apple, so "an RMW that carries its own
acquire", the textbook shape, is not available at all. What IS available, and what
DEVIATION 106 did not know, is the standalone `std.atomic.fence`, which is a different
symbol from `std.gpu.intrinsics.threadfence` and legalizes everywhere. That is what the
repair now uses, with ONE spelling for every column.

**A standalone acquire FENCE.** `std.atomic.fence` is a different symbol from
`std.gpu.intrinsics.threadfence`, it legalizes on Metal, sm_80, sm_90a and gfx942, and it is
the direct translation of cuVS's `atomicCAS` plus `__threadfence()`. It was CHOSEN FIRST in
this pass, on the strength of section digests and disassembly showing it in the artifact. It
does not run on Apple. See "THE FENCE COMPILES ON APPLE AND CANNOT RUN THERE". It costs one
instruction where the loop costs three, so if Apple ever lowers it, it is the better
spelling, and DEVIATION 106 records the failure so the check is cheap to redo.

**A post-claim acquire LOAD whose value is consumed.** THE SHIPPED REPAIR. A loop,
`while Atomic.load[ordering = Ordering.ACQUIRE](mutex) != Int32(<what the claim wrote>): pass`,
terminates in exactly one iteration, emits on all three columns because its exit condition
depends on the loaded value, and launches on Apple. The cost against the fence is one extra
`global_load_dword ... sc0 sc1` and a branch on gfx942, and one extra
`ld.acquire.sys.global.b32` and a branch on sm_80. Its one hazard is the constant: it must
be the value THIS claim writes, which is `-1` at the kNN consumer and `1` everywhere else,
and a loop waiting for the wrong value would never exit. All six sites were checked against
the third argument of their own `compare_exchange`, and two had to be corrected.

**Make the `split[node]` store itself a release, or the loads acquires.** Wrong side of the
edge. B's stores are already ordered before B's release store of the mutex by the release.
The missing half is on the acquirer, and it is missing for the mutex, not for the slot.
Field-wise acquire loads of the seven fields would have to synchronize with field-wise
release stores of the same fields, seven atomics each way per publish, and `Split` holds a
Float64 threshold and an Int64 count for which 64-bit atomics are a compile error on Apple
(`ensemble/checks/atomic_width_probe.mojo`). Heavier, less portable, no more correct.

**Remove the mutex.** One block per (node, sampled column) could write its candidate to a
per-node scratch of `n_sampled_cols` slots, and a second kernel per round could reduce them
in column order. Because sampled columns are a permutation, `update` over distinct colids
is a plain maximum on a total order, so the result would be identical to a correct mutex
merge. The costs are one extra launch per round per tree batch, a scratch of
`max_nodes * n_sampled_cols` splits, and a redesign of a seam the reference implementation
does not have. The Apple column already pays eight hours to launch overhead, so adding
launches to fix an ordering bug that one load repairs is the wrong trade. Kept as the
fallback if the leg shows the repaired claim still moves.

## 4. Is the diagnosis right on four traced divergences? YES, and not because of the count

Four is thin for a rate, and it would be thin for choosing between two mechanisms that
predict different distributions. It is not thin for this conclusion, because the argument
is structural. With every candidate merged correctly, the merged result is the maximum of
the candidates under a total order on `(gain, colid, ...)`, which does not depend on
arrival order at all. Any fit whose merged split differs from another fit's, over
bit-identical histograms and an identical column sample (which leg 11 checked at the exact
diverging round), has by definition dropped a candidate or read a partial state. Three of
the four traces show a lower colid winning an equal-gain tie that `update` awards to the
higher colid, and the fourth shows a strictly lower gain winning. Both are impossible under
a correct merge and both are exactly what a stale read of `split[node]` followed by a
write-back produces. One trace would have sufficed for "a candidate was lost". Four in the
same direction, and zero in the other, is what one expects from a mechanism that can only
ever erase.

What the count cannot yet settle is that THIS hole is the only one. That is what the A/B
leg is for. The control arm has to move at the rate the earlier legs measured, and the
repaired arm has to be stable at a size where a null is strong.

## 5. Other spellings of the same hole

Every `compare_exchange` in the repository was listed (`git grep -ln compare_exchange`);
the four files above are the complete set, and every claim in them now carries the
post-claim acquire. The gbdt single-pass reorder (`gbdt/gpu_util/kernel/
reorder_single_pass.mojo`) uses acquire loads and release stores without a claim, and it
reads its payload out of the atomic word itself, so the acquire load IS the read of the
released value and the edge is sound. The plain store of `leaf_total_zeros` there is read
by a later launch, which is ordered by the queue. No other spin-and-claim exists.

The DEVIATION 2502 word store of the purity flag outside the lock
(`builder_kernels_impl.mojo`) writes the value every merge writes and regression never
marks, so it is inert for this defect; noted so nobody re-derives it.

## Evidence, Apple (Metal) and the binaries, 2026-09-16

Run on the shared M4 under `mac_slot.sh` (one core, `nice -n 19`,
`MOJOLEARN_COMPILE_JOBS=1`, one Metal job at a time), from the worktree at
`lane/rf-mutex-claim-acquire` `2e9d7dc65`. Logs and JSONs under
`/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/460f8328-3642-45a3-a14f-579fc7e7363f/scratchpad`.

### Step 0. The inert-repair finding, re-established in one command

Before anything was built on it. Two builds of `_mojolearn_rf.so` from
`5b6ed67b3`, the shipped default and `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1`,
under `bash -x` so the define was SEEN on the compiler command line:

    pixi run mojo build -j 1 --emit shared-lib --target-cpu apple-m1
      -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1 ...

| section | fixed | stock |
|---|---|---|
| `__TEXT,__text` 715756 B | `a3d2c126...` | `a3d2c126...` |
| `__TEXT,__const` 506703 B | `9918e5d1...` | `9918e5d1...` |
| `__DATA_CONST,__const` 64 B | `48f7fde3...` | `48f7fde3...` |
| every section concatenated | `d0dab2d7...` | `d0dab2d7...` |

Confirmed. The two arms are one program.

TWO CORRECTIONS TO THE INSTRUMENT, both of which matter for anyone repeating
this.

DIGEST SECTIONS, NOT SEGMENTS. The `__TEXT` SEGMENT starts at file offset 0
and contains the Mach header and the load commands, so it carries the
`mktemp` install name and its digest differs between two builds of identical
source. Digesting the `__TEXT` segment gives a false DIFFER. The table above
is sections.

A SECTION DIGEST IS ALSO PATH DEPENDENT. Two builds of byte-identical source
from two DIFFERENT directories differ in `__TEXT,__text` and `__TEXT,__const`;
from the same directory they are identical. So a section comparison is only
meaningful when both arms were built in the same place. EVERY arm in this file
was built in the one worktree
`.../67533bb6-.../scratchpad/wt-rf-fix` by the same script, except the two
`origin/main` arms, which were both built in the one throwaway worktree
`.../460f8328-.../scratchpad/wt-main-check`. The loop arm's all-sections digest
`dcead5dda2f33020...` was produced twice in that directory, once from a scratch
edit and once from the committed source, and matched. The disassembly evidence
does not depend on this at all.

### The spelling: four of them, compared in the emitted kernel IR

`mojo build --emit asm` writes a per-kernel sidecar next to the host assembly,
`.ll` for Metal, `.ptx` for NVIDIA and `.amdgcn` for AMD. That is the artifact
the driver is handed, so it settles what emits without waiting for a binding
build. A scratch kernel in the shape of `_publish_to_global` (spin on an
acquire load, weak relaxed claim, post-claim spelling, plain load and store of
a payload, release store) was built four ways for each of Metal, sm_80 and
gfx942.

| spelling | Metal AIR | sm_80 PTX | gfx942 GCN |
|---|---|---|---|
| stock, no post-claim read | `143b374c` | `0d18d453` | `39c09595` |
| `_ = Atomic.load[ACQUIRE](mutex)` | `143b374c` | `0d18d453` | `39c09595` |
| `fence[ordering = ACQUIRE]()` | `bd2e4b8c` | `900904a0` | `b45373eb` |
| verification loop on the loaded value | `94a94043` | `5b5c1f66` | `99f79767` |

The discarded load is byte-identical to the unrepaired text on ALL THREE
columns, not only on Apple. That is the finding, at the level where it is not
an inference from a binary size.

`fence[ordering = Ordering.ACQUIRE]()` was chosen. Section 3 says why the other
two lose.

### What each spelling compiles to, on each column

Metal AIR, the fence's whole diff against stock, one line:

    27:                                               ; preds = %24
    +  fence acquire
       %28 = bitcast ptr addrspace(1) %6 to ptr addrspace(1)
       %29 = load float, ptr addrspace(1) %28, align 4

sm_80 PTX, stock beside the two emitting spellings:

    STOCK                            FENCE                          LOOP
    atom.relaxed.sys.global.cas.b32  atom.relaxed.sys.global.cas.b32   (same)
    setp.ne.b32 %p3, %r3, 0;         setp.ne.b32 %p3, %r3, 0;          (same)
    @%p3 bra $L__BB0_2;              @%p3 bra $L__BB0_2;               (same)
                                     fence.acq_rel.sys;             $L__BB0_4:
                                                                    ld.acquire.sys.global.b32 %r4,[%rd1];
                                                                    setp.ne.b32 %p4, %r4, 1;
                                                                    @%p4 bra $L__BB0_4;
    ld.global.b32 %r4, [%rd2];       ld.global.b32 %r4, [%rd2];     ld.global.b32 %r5, [%rd2];

gfx942 GCN, the column the traces came from:

    STOCK                            FENCE                          LOOP
    global_atomic_cmpswap ... sc0 sc1  (same)                        (same)
    s_waitcnt vmcnt(0)                 (same)                        (same)
    ... claim-failed branch ...        (same)                        (same)
                                     buffer_inv sc0 sc1           global_load_dword v1,v0,s[0:1] sc0 sc1
                                                                  s_waitcnt vmcnt(0)
                                                                  buffer_inv sc0 sc1
                                                                  v_cmp_ne_u32_e32 vcc, 1, v1
                                                                  s_cbranch_vccnz .LBB0_5
    global_load_dword v1,v0,s[2:3]   global_load_dword v1,v0,s[2:3] global_load_dword v1,v0,s[2:3]

`buffer_inv sc0 sc1` is the cache invalidate. On the STOCK arm there is NOTHING
between the compare-exchange that takes the lock and the plain
`global_load_dword` of the payload, so that load may be served from the
claiming XCD's own L2. That is the defect, in the instruction stream of the
part that showed it. Both repairs put an invalidate there; the loop pays one
extra global load and a branch to get it, and the fence does not, which is why
the fence was chosen first.

gfx942 and sm_80 are COMPILER OUTPUT from cross-compiling with
`--target-accelerator` on the Mac. They are not a run. Nothing in this lane has
executed on AMD or NVIDIA.

### MAIN IS BROKEN ON APPLE AS OF `c173a6b4f`

Written 2026-09-16 12:47. `01c228a72` merged lane/rf-mutex-reconcile at 12:07
and put the shared claim in `core/device_mutex.mojo`, whose line 122 is
`fence[ordering = Ordering.ACQUIRE]()` under `comptime if not CLAIM_STOCK`.
`claim_device_mutex` is called from `ensemble/decisiontree/batched_levelalgo/
split.mojo`, `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` and
`neighbors/impl/detail/fused_l2_knn.mojo`, so random forest, extratrees and
the fused kNN path all carry it on Apple.

MEASURED ON MAIN'S OWN BINARY, not inferred from this branch. A throwaway
worktree at `origin/main c173a6b4f`, `bindings/build_rf.sh` and
`bindings/build.sh`, then the smallest fit that exists, 512 rows, 6 columns,
3 trees, depth 6, under the Metal lock:

| main `c173a6b4f`, one worktree, one machine | result |
|---|---|
| default build (fence compiled IN) | **fit FAILS**, `Failed to create compute pipeline state (GPU machine code generation): Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED` |
| `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` (fence compiled OUT) | **fit OK**, train accuracy 0.996 |

Same commit, same worktree, same machine, minutes apart. The only difference
is the fence. The attribution is closed on main's own artifact, not only on a
probe.

THE FIX IS THE SPELLING THIS BRANCH CARRIES, applied to
`core/device_mutex.mojo` instead of to six sites. One trap for whoever does
it: the loop must compare against `claim_value`, the value THAT claim writes,
not a literal `1`. `claim_device_mutex` serves both the `0 -> 1` and the
`-2 -> -1` claims, so a literal would hang the kNN consumer forever. Two of
this branch's six sites were wrong on the first pass for exactly that reason.

    while Atomic.load[ordering = Ordering.ACQUIRE](mutex) != claim_value:
        pass

### THE FENCE COMPILES ON APPLE AND CANNOT RUN THERE

This is the finding of the pass, and it cost the fence spelling.

`fence[ordering = Ordering.ACQUIRE]()` compiles for Metal, emits `fence acquire`
into the AIR, and `xcrun metal-objdump -d` disassembles it. Every launch then
fails at PIPELINE CREATION:

    Failed to create compute pipeline state (GPU machine code generation):
    Compilation failed due to an interrupted connection:
    XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.

That is the AIR-to-ISA compiler service dying, not a compile diagnostic, which
is why nothing upstream of a launch catches it.

WHAT MAKES THIS A BINARY FACT AND NOT AN INFRASTRUCTURE STORY. The first A/B
ran the fence arm FIRST and every one of its 81 cells refused while the stock
arm, run second, refused none. That confounds the binary with the position, so
the run was repeated with the ORDER REVERSED.

| run | position 1 | position 2 |
|---|---|---|
| first | fence, **81 REFUSED** | stock, 80 STABLE 1 MOVED, 0 refused |
| reversed | stock, 80 STABLE 1 MOVED, 0 refused | fence, **81 REFUSED** |

The refusal follows the BINARY, not the position, in both directions.

It was then reproduced in isolation, in about six seconds, by three
single-kernel executables built from one source behind three defines, each
creating ONE pipeline and launching it, run in two rounds:

    round 1  fence   -> FAILED rc=1 :: Failed to create compute pipeline state ...
    round 1  stock   -> OK rc=0 :: claim probe done
    round 1  verify  -> OK rc=0 :: claim probe done
    round 2  fence   -> FAILED rc=1 :: Failed to create compute pipeline state ...
    round 2  stock   -> OK rc=0 :: claim probe done
    round 2  verify  -> OK rc=0 :: claim probe done

REPLICATED SOLO AFTER A CONTENTION WARNING. Another session reported that
`tools/gemm_remote_leg.sh` generated an Apple reference card WITHOUT taking
the Metal lock at 12:27 and 12:29. Neither fell inside the runs above: the
reversed A/B finished 12:23:26 and the first probe 12:24:15. The probe was
nonetheless rerun alone under the lock at 12:38, three rounds with the order
rotated inside each round, and the result is unchanged: `fence` FAILED 6 of 6,
`stock` and the loop passed 6 of 6 each. Concurrent Metal is not the cause.

So the fence is out, the loop is in, and the general lesson is the rule
`neighbors/mutex_probe_main.mojo` already wrote down and this pass forgot:
on Apple, support is established BY ENQUEUE, never by host compile. Section
digests and disassembly proved the fence was IN the artifact. They could not
have told anyone it would not launch, and nothing short of running it would
have.

### Section digests of the shipped binding, before and after

`_mojolearn_rf.so`, identical tier, Apple, one core, all sections concatenated.

| arm | all sections | runs on Apple |
|---|---|---|
| stock (`-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1`) | `996d1c15f679092b...` | yes |
| the original `_ =` repair | `996d1c15f679092b...` | yes, being the same program |
| acquire fence | `3700cd9899bf7b10...` | **NO, pipeline creation fails** |
| acquire loop (SHIPPED) | `dcead5dda2f33020...` | yes |

Rows one and two are the same digest, which is the step-0 result and also the
pair the leg guard is now proved to reject. The loop arm's digest was produced
twice, once from a scratch edit and once from the committed source, and came
out identical both times.

### The loop is in the shipped artifact, at the claim site

Carved out of `_mojolearn_rf.so` by `MTLB` header and disassembled with
`xcrun metal-objdump -d`. The three `ensemble_decisiontree_batched_*` publish
kernels carry TWO acquire loads each; the stock arm's carry one.

    1398:                                        ; preds = %1400, %1398, %1394
      %1399 = load atomic i32, i32 addrspace(1)* %758 acquire, align 4
      %.not174 = icmp eq i32 %1399, 0
      br i1 %.not174, label %1400, label %1398        ; the SPIN

    1400:                                        ; preds = %1398
      store i32 0, i32* %10, align 4
      %1401 = call i32 @air.atomic.global.cmpxchg.weak.i32(
                  {} addrspace(1)* %759, {}* nonnull %11, i32 1, i32 0, i32 0, i32 2, i1 true)
      %1402 = icmp eq i32 %1401, 0
      br i1 %1402, label %1403, label %1398        ; the CLAIM

    1403:                                        ; preds = %1403, %1400
      %1404 = load atomic i32, i32 addrspace(1)* %758 acquire, align 4
      %.not175 = icmp eq i32 %1404, 1
      br i1 %.not175, label %1405, label %1403     ; THE REPAIR

    1405:                                        ; preds = %1403
      %.elt176 = getelementptr inbounds { i32, float, i32, float, i64, i64, i32, i32 },
                  ... addrspace(1)* %13, i64 %37, i32 1
      %.unpack177 = load float, float addrspace(1)* %.elt176, align 4

`{ i32, float, i32, float, i64, i64, i32, i32 }` is `Split`. The repair sits
between the claim and the plain loads of `split[node]`'s fields, which is the
whole point, and `%1404` is on the same address `%758` the spin and the claim
used.

### A contamination this session caused, recorded because it is still live

A first attempt at the A/B installed an arm by copying it over the SHARED
worktree's `python/mojolearn/identical/_mojolearn_rf.so`, then died on a
missing `PYTHONPATH` before running anything. It left the STOCK arm
(`996d1c15...`) installed there at 11:29. Another lane's Metal run, queued
behind the lock and started at about 11:37
(`tools/identity_break.py --lanes rf-clf,rf-reg,rf-reg-poisson,rf-reg-gamma-ig,par-forest
--fixtures base,ties,wide --json .../json/rf_sabotage.json`), therefore imported the
PRE-REPAIR binary, not this branch's default.

IT WAS NOT INTERRUPTED, and it must not be: it is an owed run and killing it
would hand its whole cost to the next attempt. What its output is NOT is a
measurement of this branch. Anything it reports about the rf lanes is a
measurement of `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1`.

The A/B driver was rewritten so this cannot recur. Each arm now gets a PRIVATE
copy of the `mojolearn` package under the scratchpad with only its own
`_mojolearn_rf.so` in it, and `PYTHONPATH` points at that copy. The shared
worktree's `python/mojolearn` is never written.

### Two Apple run-to-run movers from the previous pass, unexplained

Recorded and NOT chased, and they are noted here because they contradict this
file's earlier sentence that the Apple column has "never shown a move".

  - `rf-reg-gamma-ig/wide` MOVED on the stock arm: two fits of the same seed
    gave `fbdbb762bf85ec73` and `f76e788124ef87d3`.
  - `par-forest/denormal_ftz` REFUSED on the fence arm's predecessor, from the
    lane's own check: `fit_forest predict_proba and plain predict_proba
    differ: 1236 bytes of 16384`.

Both were taken while the Mac was at load 20 to 25. Replicated at six repeats
on a quiet machine, both arms, all four cells were STABLE and IDENTICAL across
arms. Neither reproduced. NOTHING IS ATTRIBUTED TO THE SPELLING: one event
landed on each arm and, as the section digests above prove, those two arms
were the same program, so the spelling could not have been the variable. The
rate is not established, one event in two repeats then zero in six. They stay
here as unexplained.

### The A/B leg guard, rewritten so it can fail

`tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh` decided its two arms
were independent by comparing the whole-file sha256 of `_mojolearn_rf.so`,
AFTER both arms had been timed. Two things were wrong with that.

  - The comparison could never fire. A file digest differs for reasons that
    are not code, so its `DIGEST COLLISION ... results VOID` branch was dead.
  - It ran last. A gate that runs after the arms cannot stop a leg from
    spending the box on one program timed twice.

Now: both arms are BUILT first and each is copied aside so the second build
cannot overwrite the first; their CODE AND CONSTANT sections are compared by
`tools/rf_nondeterminism/section_digest.py`; and `run_arm` is not called at all
unless that comparison says DIFFER. `section_digest.py` reads ELF `.text` and
`.rodata` or Mach-O `__TEXT,__text` and `__TEXT,__const` with its own header
reader, so a pod without binutils is not a reason for it to go quiet, and it
REFUSES with exit 2 rather than printing a digest when it does not understand
the file or a section is missing. Exit 1 means IDENTICAL and exit 2 means
refused, deliberately different numbers, so a broken instrument can never be
read as a passing comparison.

THE VOID BRANCH WAS MADE TO FIRE, on the pair that is genuinely one program:
the two step-0 arms. The gate block was extracted verbatim out of the template
and run against three pairs.

| pair | gate says |
|---|---|
| the two step-0 arms (`_ =` repair vs its control) | `SECTION DIGEST COLLISION -- .text and .rodata are IDENTICAL, the two arms are ONE PROGRAM, results VOID, nothing timed`, `ARMS_INDEPENDENT=0` |
| the fence arm vs its control | `sections DIFFER: stock_prerepair=996d1c15 claimfix=3700cd98`, `ARMS_INDEPENDENT=1` |
| a text file handed to it as a `.so` | `SECTION DIGEST REFUSED, results VOID, nothing timed`, `ARMS_INDEPENDENT=0` |

The first row is the one that matters. The old guard PASSED that same pair,
because their file digests are `cbb686c6` and `5c6257b8` and differ. The leg
would have run.

### Binaries built, all identical tier, Apple, one core

| file | spelling | all-sections sha256 | launches on Apple |
|---|---|---|---|
| `_mojolearn_rf.so` | `_ =` acquire load (step 0, inert) | `996d1c15f679092b...` | yes |
| `_mojolearn_rf.so` | `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` | `996d1c15f679092b...` | yes |
| `_mojolearn_rf.so` | acquire fence | `3700cd9899bf7b10...` | **no** |
| `_mojolearn_rf.so` | acquire loop (SHIPPED) | `dcead5dda2f33020...` | yes |
| `_mojolearn_trees.so` | acquire loop | built | not separately run |
| `_mojolearn.so` | acquire loop (kNN sites) | built | not separately run |
| `_mojolearn_metrics.so` | not a claim site, built so `rf-score-weighted` can run | | |
| `neighbors/mutex_probe_main.mojo` | acquire loop | compiles | NOT RUN |
| `claim_probe` exe | stock / fence / loop, one kernel each | | stock yes, fence no, loop yes |

Rows one and two are the same digest. That is the step-0 result and also the
pair the leg guard is now proved to reject.

### CPU host column: the repaired lines are not on it

Not run, because it would be a pass that cannot fail, and the source says so.

  - The rf CPU route is `bindings/_mojolearn_rf_host.mojo`, whose header says
    "HOST ONLY. No DeviceContext, no kernel launch, no GPU", and whose fit is
    `ensemble/host/rf_oracle.mojo::rf_host_fit`.
  - `grep -c "Atomic\|compare_exchange\|mutex"` over that closure
    (`_mojolearn_rf_host.mojo`, `ensemble/host/rf_oracle.mojo`,
    `core/forest_host_predict.mojo`, `bindings/forest_export_binding.mojo`,
    `bindings/forest_host_groves_binding.mojo`) returns 0 for every file. The
    same grep on `ensemble/decisiontree/batched_levelalgo/split.mojo` returns
    31, so the grep has teeth.
  - `bindings/_mojolearn_trees_host.mojo` is 0 as well and does not import
    `batched_levelalgo`.
  - It could not have been selected here anyway: `python/mojolearn/_backend.py`
    loads the host set only when `_CPU_ONLY is not None`, "so a box with a GPU
    never serves host arithmetic under a GPU label".

A digest experiment on the host binding was attempted and is NOT reported as
evidence: its contrast arm failed. Three builds to one fixed `-o` gave one
sha256 with and without the define, but the same experiment on the GPU binding
ALSO gave one sha256, and the GPU binding is the arm that had to move. Two
causes were found and both are recorded so the next session does not repeat
them: the Bash tool runs zsh, where an unquoted `$flags` does not word-split,
so the `-D` never reached argv on the first attempt; and after that was fixed
the builds completed in seconds, which are compiler-cache hits. The byte
comparison of `__TEXT` above replaces it and needs no such control.

### kNN: the two fused sites are unreachable from the public surface

Not run, for the same reason, and again from the source.

  - `bindings/_mojolearn.mojo:234` is the only `knn_search(` call site and it
    passes `KNN_METHOD_AUTO`. `python/mojolearn/neighbors.py:487` says the arm
    "is NOT a parameter of this" surface.
  - Under `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`,
    `neighbors/impl/detail/knn_brute_force.mojo:1482` sets `want_fused = False`
    unconditionally for AUTO (DEVIATION 509), and the launch at `:1560` is
    guarded by `and want_fused`.
  - Even under an explicit `KNN_METHOD_FUSED`, `fused_l2_knn.mojo:563` takes a
    `gdx == 1` early path that never touches the mutex.

So `neighbors/impl/detail/fused_l2_knn.mojo`'s two claim sites cannot be
exercised by any kNN lane in the identical tier on any column, and a kNN
identity comparison would have been a third check that cannot fail. DEVIATION
106's sentence that the kNN producer and consumer "carry the same post-claim
acquire load" is true of the text and says nothing about the shipped path.

### Still owed

  - **FIX MAIN.** `c173a6b4f` cannot fit a forest on Apple. This branch's six
    per-site loops are now the WRONG SHAPE for main, which has since moved the
    claim into `core/device_mutex.mojo` (`01c228a72`). What main needs is the
    loop applied ONCE there, comparing against `claim_value`. This branch is
    the evidence and the spelling, not the patch.
  - THE APPLE IDENTITY A/B FOR THE LOOP IS NOT FINISHED. Its loop arm was
    cancelled partway when the fence-on-main question took priority. The stock
    arm has been run three times (80 STABLE 1 MOVED each time) but the loop arm
    has no completed record, so this branch has NO cell-for-cell identity
    result for the shipped spelling. It must be run before the spelling is
    trusted to leave output unmoved.

  - THE MI300X A/B. Nothing in this lane has RUN on AMD. Everything above is
    compiler output plus an Apple identity comparison, and neither can say the
    hole is closed on the part that showed it. The leg
    (`tools/rf_nondeterminism/rf_claimfix_ab_body.template.sh`) is ready, its
    guard now compares sections before either arm is timed, and the guard has
    been shown to reject a pair of identical arms. Blocked on Hot Aisle stock;
    leg-1 under `~/mojolearn-evidence/rf-mutex-claim-acquire/leg-1/` records
    `exit=3`, no box created and nothing spent.
  - The control arm has to be SEEN to move at the rate the earlier legs
    measured, 13/300 to 16/300 on the wide fixture at 16 columns. If it does
    not move, the leg says nothing about the repair.
  - A powered run on `rf-reg-gamma-ig/wide`, the shape with the most mutex
    traffic per node, to put a rate on the two unexplained Apple movers.
  - `neighbors/mutex_probe_main.mojo` has never been RUN with the repair in it.
    It compiles, and its two claim sites now carry the loop with the consumer
    waiting for `-1`. Its two sabotage arms are the only thing in the tree that
    tests this handoff protocol directly, and running it on Metal is the
    cheapest non-AMD evidence that the repair does not break the protocol it is
    meant to strengthen. It matters more than usual now, because the consumer's
    loop constant differs from every other site's and nothing has executed that
    path.
  - `_mojolearn_trees.so` and `_mojolearn.so` were rebuilt with the loop and
    were NOT run. Only the rf binding has been through identity with it.
