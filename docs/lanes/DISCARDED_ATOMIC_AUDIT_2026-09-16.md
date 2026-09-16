# Discarded atomics: a repository audit, 2026-09-16

`lane/discarded-atomic-audit`. Evidence outside the repo in
`/Users/andrewhendel/mojolearn-evidence/discarded-atomic-audit/`.

## Why this lane exists

The random forest mutex repair was first written as
`_ = Atomic.load[ordering = Ordering.ACQUIRE](p)`, and that spelling turned out
to compile to **zero instructions**. Nobody had checked whether the same shape
appeared anywhere else. This is that check.

## The answer, first

**No site in the repository has the defect.** Every atomic operation on
`origin/main` at `01c228a72` either uses its value or is a read-modify-write
whose memory side effect is the point. There are zero pure atomic loads whose
value is discarded, zero assigned to a variable that is never read, and zero
whose value feeds a computation the compiler can fold away.

The one site that was written that way, `claim_device_mutex`, already landed
today spelled as `fence[ordering = Ordering.ACQUIRE]()`, and that spelling is
confirmed below to emit on all three GPU columns and to move the Apple
metallib.

Three things came out of the measurement that were not the question asked, and
they are recorded at the end: a section-digest comparison is **path dependent**,
`__TEXT,__text` moves for a reason unrelated to the code under test, and on
Apple a sequentially consistent `Atomic.fetch_add` is emitted as a **relaxed**
one.

## What emits and what does not

A probe of matched WITH/WITHOUT kernel pairs, differing by exactly one line,
compiled for each column and compared instruction for instruction.
`emit_probe.mojo` (GPU), `emit_probe_host.mojo` (CPU). Assembly is the
per-kernel sidecar `mojo build --emit asm` writes: `.ptx` for NVIDIA,
`.amdgcn` for AMD, `.ll` (AIR) for Metal, and the host `.s` for the CPU.

| spelling | NVIDIA sm_90a | AMD gfx942 | Apple M4 | CPU arm64 |
|---|---|---|---|---|
| `_ = Atomic.load[ACQUIRE](p)` | **nothing** | **nothing** | **nothing** | **nothing** |
| `var w = Atomic.load[ACQUIRE](p)`, `w` never read | **nothing** | **nothing** | **nothing** | **nothing** |
| `Atomic.load[ACQUIRE](p)` into a computation the compiler folds dead | **nothing** | **nothing** | **nothing** | **nothing** |
| `if Atomic.load[ACQUIRE](p) == k` (value used) | `ld.acquire.sys.global.b32` | `global_load_dword sc0 sc1` + `buffer_inv sc0 sc1` | `load atomic ... acquire` | `ldapr` |
| `Atomic.store[RELEASE](p, v)` | `st.release.sys.global.b32` | `buffer_wbl2 sc0 sc1` + `global_store_dword sc0 sc1` | `store atomic ... release` | `stlr` |
| the same store written plainly | `st.global.b32` | `global_store_dword` | `store i32` | `str` |
| `_ = Atomic.fetch_add[RELAXED](p, v)` | `atom.relaxed.sys.global.add.u32` | `global_atomic_add sc1` | `air.atomic.global.add.s.i32` | `ldadd` |
| `_ = Atomic.fetch_add(p, v)` (default, seq_cst) | `fence.sc.sys` + `atom.acquire...add` | `buffer_wbl2` + atomic + `buffer_inv` | `air.atomic.global.add.s.i32`, **identical to relaxed** | `ldaddal` |
| `_ = Atomic.min(p, v)` | `fence.sc.sys` + `atom.acquire...min.s32` | a `global_atomic_cmpswap` **loop** | `air.atomic.global.min.s.i32` | `ldsminal` |
| `Atomic.min(p, v)` written bare, no `_ =` | identical to `_ =` | identical | identical | not measured |
| `_ = Atomic.compare_exchange[...]`, result discarded | `atom.acquire.sys.global.cas.b32` | `global_atomic_cmpswap sc1` | `air.atomic.global.cmpxchg.weak.i32` | `casa` |
| `fence[ordering = Ordering.ACQUIRE]()` | `fence.acq_rel.sys` | `s_waitcnt vmcnt(0)` + `buffer_inv sc0 sc1` | `fence acquire` | not measured |
| `fence[ordering = Ordering.RELEASE]()` | `fence.acq_rel.sys` | `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0)` | `fence release` | not measured |

The rule is the same on every column, including the CPU: **a pure atomic load
whose value nothing consumes is deleted whatever ordering it carries, and every
read-modify-write survives whether or not its result is used.** On arm64 the
deletion is total enough that the compiler then merges the two payload stores
around it into one 64-bit store.

Two spellings are refused outright rather than silently dropped, and both are
compile errors, not defects:

* `Atomic.compare_exchange[success_ordering = Ordering.ACQUIRE]` is rejected on
  the Apple GPU by name (`Apple GPU does not support 'acquire' atomic
  ordering`). The same call compiles for the host CPU, so the restriction is
  the GPU backend's, not the platform's.

### The comparison was seen to fail before any "identical" was believed

Six of the twelve pairs moved, on every column, in the same run that reported
the other three as producing nothing. An "identical" here is not a harness that
cannot fail.

## Every atomic operation in the repository

166 code sites (comment lines excluded), from
`git grep -nE 'Atomic\.|fence\[' -- '*.mojo'` at `01c228a72`, all of them
inside GPU kernels. No `Atomic[...]` instance anywhere, and `__threadfence` and
`volatile` appear only inside comments quoting the reference C++. `Atomic` and `Ordering` and `fence` are
the only names imported from `std.atomic` anywhere in the tree.

| shape | count | result is | verdict |
|---|---|---|---|
| `Atomic.fetch_add` | 126 | discarded at 115 sites, used as a cursor or slot index at 11 | EMITS. The write is the point; the discarded return is free |
| `Atomic.min` | 15 | discarded | EMITS |
| `Atomic.store[RELEASE]` | 11 | none | EMITS, and is distinguishable from a plain store on every column |
| `Atomic.load` | 6 | used at all six | EMITS |
| `Atomic.max` | 2 | discarded | EMITS |
| `Atomic.compare_exchange` | 1 | used | EMITS |
| `fence[ACQUIRE]` | 1 | none | EMITS |
| **pure load with the value discarded** | **0** | | **the defect does not appear** |

### The six loads, one by one

| site | spelling | value used for | emits |
|---|---|---|---|
| `core/device_mutex.mojo:110` | `Atomic.load[ACQUIRE]` | the spin test `!= available` | yes |
| `core/device_mutex.mojo:107` | `Atomic.load[RELAXED]` | the same test under `-D MOJOLEARN_MUTEX_SPIN_RELAXED=1` | yes; it is a measurement arm, never a default |
| `extratrees/impl/decisiontree/batched_levelalgo/split.mojo:528` | `Atomic.load[RELAXED]` | a spin counter in the `SPLIT_SAB_NO_LOCK` sabotage arm | yes |
| `gbdt/gpu_util/kernel/reorder_single_pass.mojo:315` | `Atomic.load[ACQUIRE]` | the decoupled-lookback predecessor status and its packed value | yes |
| `gbdt/gpu_util/kernel/reorder_single_pass.mojo:352` | `Atomic.load[ACQUIRE]` | the leaf total, status and value in one word | yes |
| `gbdt/gpu_util/kernel/reorder_single_pass.mojo:370` | `Atomic.load[ACQUIRE]` | the predecessor tile's inclusive count | yes |

The three in `reorder_single_pass.mojo` are release-acquire pairs with the
payload packed into the same word the acquire load reads, so there is no
separate payload that could be reordered around the edge. `leaf_total_zeros`
is written plainly next to the release store but is never read inside that
kernel; its reader is `split_points.mojo:1216`, in a different kernel, across a
launch boundary.

### The eleven release stores

`core/device_mutex_check.mojo:65`, `ensemble/.../split.mojo:699`,
`extratrees/.../split.mojo:555`, `reorder_single_pass.mojo:309,331,342`,
`fused_l2_knn.mojo:655,765`, `mutex_probe_main.mojo:160,174,196`. Every one of
them emits, and emits **differently from the plain store** it would otherwise
be, on all four columns. The release half of every handback in the repository
is real.

### The `_ =` prefix is not load bearing

`Atomic.min(p, v)` written as a bare statement and `_ = Atomic.min(p, v)`
produce byte-identical assembly on all three GPU columns. The repository uses
both spellings; the inconsistency is cosmetic. `extratrees/.../kernels/builder_kernels_impl.mojo:398-399`
are the two bare ones.

## The defect that does exist, and is already repaired

The ordering hole the RF lanes found is **not** an emission defect. The spin's
acquire load emits. The hole is that the acquire was on the **failure** path
only: a thread whose first compare-exchange succeeds never executed an acquire
that synchronized with the release it claimed against. `core/device_mutex.mojo`
now closes it with `fence[ordering = Ordering.ACQUIRE]()` after a successful
claim, and its own comment already says not to spell it as a discarded load.

That repair was confirmed here, independently, at two levels:

* **Instruction level.** `fence[ACQUIRE]()` emits `fence.acq_rel.sys` on
  sm_90a, `s_waitcnt vmcnt(0)` plus `buffer_inv sc0 sc1` on gfx942, and
  `fence acquire` in Apple AIR.
* **Artifact level, Apple.** Replacing `fence[ordering = Ordering.ACQUIRE]()`
  with `_ = Atomic.load[ordering = Ordering.ACQUIRE](mtx)` in one probe,
  rebuilt from the same absolute path, moves `__TEXT,__const` (where the
  embedded metallib lives) and leaves `__TEXT,__text` alone. Two builds of the
  unmodified source are identical in both sections. The fence and the discarded
  load are not the same program.

  ```
  stock                    __TEXT,__const 7204a55e61d68dab…
  stock_again              __TEXT,__const 7204a55e61d68dab…
  fence_to_discarded_load  __TEXT,__const 468eabad7059dda4…
  ```

## Three things about the method, learned the hard way

### 1. A section digest is path dependent

Two builds of **byte-identical source** from two different directories differ
in `__TEXT,__text` *and* `__TEXT,__const`. Two builds from the **same**
directory are identical. So an A/B whose arms live in separate directories, or
in separate worktrees, reports DIFFER for a reason that has nothing to do with
the code. Every arm must be built from the same absolute path, editing the file
in place between builds and restoring from a byte copy.

`section_gate/digests.txt` has the failing version; `section_gate2/digests.txt`
has the same comparison done correctly, with a passing null arm.

### 2. `__TEXT,__text` is not a usable section for a source-level A/B

Removing one statement anywhere in the file changed five 16-bit immediates in
host `__TEXT,__text`, each by exactly one. They are the source positions baked
into the `Error.__init__` paths of the `enqueue_function` call sites:
`mov w0, #0xa9` becomes `mov w0, #0xa8`. Replacing the statement with a comment
instead of deleting it does **not** avoid the shift. For a change inside a GPU
kernel, compare `__TEXT,__const`, which carries the metallib; host `__text`
will move for the edit itself.

### 3. On Apple, a sequentially consistent RMW is emitted as a relaxed one

`Atomic.fetch_add[ordering = Ordering.RELAXED](p, v)` and
`Atomic.fetch_add(p, v)` (default, seq_cst) produce **byte-identical AIR**:
`call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %9, i32 1, i32 0, i32 2, i1 true)`.
On NVIDIA the default adds `fence.sc.sys`, on AMD `buffer_wbl2` plus
`buffer_inv`, on arm64 `ldaddal` against `ldadd`. So the ordering an unadorned
`Atomic.fetch_add` carries is stronger on three columns than on Apple.

No site depends on it. All 126 `fetch_add` sites are counters, histogram bumps
or cursor bumps whose value is either discarded or used as an index into memory
the same thread then writes; none of them fences a payload for another thread.
It is recorded so that the next person who reaches for a default-ordering
`fetch_add` **as** a fence knows it is not one on Apple.

## Two notes, neither a defect

* **`extratrees/.../split.mojo` selects its sabotage at runtime**, from an
  `Int32` argument, not at compile time. Every sabotage arm, including the
  `SPLIT_SAB_NO_LOCK` spin at line 528, is therefore present in the shipped
  kernel as a predicated branch. Nothing is wrong with the shipped answer; it
  is worth knowing the code is there.
* **`Atomic.min` on gfx942 is a compare-exchange loop**, not a native
  `global_atomic_smin`. Fifteen sites take it. That is a speed observation, not
  a correctness one, and it belongs to whoever owns AMD speed.

## Recommendation

One cheap guard would stop this shape coming back, and it is not written here
because this lane is an audit:

```sh
git grep -nE '^\s*_ = Atomic\.(load|compare_exchange)' -- '*.mojo'
```

must stay empty. A discarded `Atomic.load` is always dead code; a discarded
`compare_exchange` is not dead, but a claim whose result nobody tests is
almost certainly a mistake. `Atomic.fetch_add`, `Atomic.min` and `Atomic.max`
must stay out of that pattern, since discarding their result is the normal and
correct thing to do.

## Lane verification

This lane changed no code, so there is nothing to verify and the selector says
so rather than sweeping:

```
$ python3 tools/verify_lanes.py --changed-since origin/main --fixtures base
# 3 changed path(s) against origin/main
# docs/lanes/DISCARDED_ATOMIC_AUDIT_2026-09-16.md: inert (prose or evidence)
# 0 of 212 lanes selected
# REFUSING: the selection is empty. An empty run is not a pass; if the change
# really touches no lane, say so in the lane status file rather than running this.
```

Said here, as it asks. The first attempt DID sweep, because an untracked
`.metadata_never_index` in the worktree came back `NOT ATTRIBUTABLE: no lane's
derived source set names it, so every lane`, and the selector fell back to all
212. A stray untracked file in a worktree turns a narrow run into a full sweep;
the file was removed.

The measurement this lane rests on is in
`/Users/andrewhendel/mojolearn-evidence/discarded-atomic-audit/`, and its own
gate is the WITH/WITHOUT pairs that moved.

## Reproducing

```sh
cd /Users/andrewhendel/mojolearn-evidence/discarded-atomic-audit
export MODULAR_HOME=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/share/max
MOJO=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo
# per-kernel assembly for each column
"$MOJO" build -j 1 --emit asm --target-accelerator sm_90a   -o asm_nvidia2/probe.s emit_probe.mojo
"$MOJO" build -j 1 --emit asm --target-accelerator gfx942   -o asm_amd2/probe.s    emit_probe.mojo
"$MOJO" build -j 1 --emit asm --target-accelerator apple-m4 -o asm_apple2/probe.s  emit_probe_apple.mojo
"$MOJO" build -j 1 --emit asm                               -o asm_host/probe.s    emit_probe_host.mojo
# the Apple metallib section gate, with its null arm
bash section_gate2.sh
bash fence_metallib.sh
```

Every build ran at `nice -n 19`, one job, through
`MAC_SLOTS=4 bash ~/mojolearn-evidence/tools/mac_slot.sh run`.
