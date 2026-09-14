# BRIEF, the two MOVED lanes of the 118-lane record (2026-09-14)

The 118-lane three-column run at 71faae781 (`bench/results/identity_break/2026-09-14_118-lanes/`)
carried two MOVED lanes among the 71 the claim-surface census added. MOVED is one box
disagreeing with itself across two fits in one process, a determinism finding before it is
a cross-vendor one. One was the lane's own hashing, fixed here; the other, on the MI300X, was open
until the 136-lane record (section 2).

## 1. byte-lm-resident, MOVED on all nine fixtures on the H100 and the MI300X: the LANE, fixed

Per-part reading of the columns: only the `grads` part moved; `loss`, `params` and
`logits` were equal between the two fits AND equal, bit for bit, to the stateless `byte-lm`
lane on both GPUs (H100 base: byte-lm loss 08c41aea params fa339c06 logits 310c2e67, the
same three in both resident fits). So the resident step consumed and produced the right
bytes; only what the lane hashed for the exported gradient moved.

Cause: `SmallByteLanguageModelTrainer.export_gradients()` returns
`dict(flat_gradients=Array, gradients={name: Array})`. The lane hashed
`np.asarray(grads[k]) for k in sorted(grads)`, and `np.asarray` of a dict is a
zero-dimensional object array whose `tobytes()` is the dict's ADDRESS: different per fit,
different per box. Reproduced on the Apple M4 before the fix (repeats=2, base: MOVED,
moved parts `['grads']`) and STABLE after it on base, ties and odd with two repeats.

Fix (same day, on main): hash `flat_gradients` and the named tensors as arrays. And since
the lane now has both trainers in hand, it holds the resident export to the stateless
path's returned gradient for the same three steps through `_same_bytes`, so gate G1 of
`docs/lanes/DESIGN_lm_device_owned_step_2026-09-11.md` (RUN OWED on the H100 since
2026-09-11) is a cell on every column from now on: a byte between the two reads REFUSED
with the pair named. On the Mac the two agree.

Lesson for the harness: `_h` accepts any object and hashes whatever `np.asarray` makes of
it; an object array is silent nondeterminism. A guard that refuses `dtype=object` in `_h`
is the right fix for the tool and is not made here (the harness is measuring right now on
three boxes); make it in the next harness change and rerun the three columns.

## 2. mamba2-dtlimit, `step` part MOVED once on the MI300X (base, one of nine fixtures): CLOSED, not reproduced after the 2712/2713 fix

CLOSED (2026-09-14 evening). The 136-lane record at 4048e1b51, the commit carrying the
m2_ydiag_kernel over-read fix of section 3.3, reads `mamba2-dtlimit` STABLE on all nine
fixtures of the Hot Aisle MI300X column (two fits per cell, `step` part equal in both, base
`step` 1a8d61928f80731b) and IDENTICAL x3 against apple-m4 and nvidia-h100-sm_90a on all
nine (`bench/results/identity_break/2026-09-14_136-lanes/diff.apple-h100-mi300x-incomplete.txt`,
base b9e7928a2a4d30cf on every column); the MI325X column reads the same nine IDENTICAL x3
(`diff.three-columns.txt`). DEVIATION 2712 was an Apple over-read, not an AMD race, and is
fixed. The over-read is the one mechanism found that touches this part: the step reads
farthest past X_d's end because its T is 1, and a read of unowned memory is free to change
between two fits. That it caused this one MOVED cell is consistent with the record, not
shown by it; no MOVED `step` has been seen on any column since the fix. The paragraphs below
are the ledger as written while the lane was open.

`mamba2-dtlimit/base` on gfx942: train hashes 165b502a1c95f280 vs 669a2db9147247f7, the
moved part is `step` only (the one-token decode after the prefill into a state); forward,
prefill and backward equal; the other eight fixtures stable; the H100 stable on all nine;
the `mamba2` lane (default dt_limit) stable on every column of every run so far (three
runs, nine fixtures, two repeats).

What `step` runs: `mamba2_step` is `mamba2_block_forward` at l = 1 with the state carried
(`mamba/impl/modules/mamba2.mojo:1258`, DEVIATION 786, "no arithmetic of its own"). The
l = 1 path launches two kernels the prefill does not: `m2_step_upstream_kernel`
(`mamba/impl/modules/mamba2.mojo:684`, launched at ~:1027) and `m2_buffer_update_kernel`
(`:556`, launched at ~:1133), which update the SSM state and the conv window IN PLACE. No barrier, block reduction or warp primitive appears
in the block forward, so the candidate is not a 64-lane fold; it is a read-modify-write
on the carried state where one thread reads an element another thread has already
written in the same launch, or two launches on the state that the profile assumes are
ordered. An active dt clamp does not change the code path (contract S9: the clamp is
present at the default too), so the dt_limit value is not the cause; it changes the dt
values, which changes which state elements are touched first. One cell in eighteen on
AMD and none on NVIDIA is the shape of a launch-order race, not a fold.

Read after writing the paragraph above: both kernels are one thread per element with
disjoint writes (`m2_buffer_update_kernel` copies one window element per thread from the
working buffers, `m2_step_upstream_kernel` owns one (b, h, p) row of the state per thread),
so by inspection neither races with itself. Two things the reading did turn up. The step
does NOT clamp dt (`mamba2.mojo:709`, "NO clamp in their step (:313)", upstream fidelity),
so with an active dt_limit the prefill's state is built from clamped dt and the decode
from unclamped dt; that is a property of the profile, not a race, but it is why this lane
and not `mamba2` sees whatever moves. And the remaining suspects are the l = 1 conv-window
read (`m2_conv_window_kernel`, launched at ~:943) and the synchronization between the last
kernel of the step and the download of `y_step` on HIP, where a copy issued before the
last launch retires would read a partially written output once in a while; one cell in
eighteen on one vendor is that shape too.

Addendum, 2026-09-14 afternoon. The clean 120-lane rerun at 65ae7612f read 1080 of 1080
stable on the Apple M4 and on the H100 (zero moved), so section 1's fix and the `_h`
refusal hold. Its AMD column fell back to a DigitalOcean MI325X (Ubuntu 24.04 ROCm image)
and ABORTED after the mamba1 lane, before mamba2 and mamba2-dtlimit, with "Memory access
fault by GPU node-1 ... Reason: Unknown", exit 134, 102 rows in
(`bench/results/e1g/2026-09-14_130705-amd-mi325x-do-identity-120-lanes-clean`, partial JSON
and log beside the record as `amd-mi325x-gfx942.partial.*`). The Hot Aisle MI300X (22.04
container) ran those lanes three times the same day without a fault. Checked against the
out-of-bounds reading above: `m2_buffer_update_kernel`'s source row is
`src = t_work - r + t` with `r = t_work mod Q` and `t < r` (`mamba2.mojo:1128-1131`), so
`src >= 0` and in range at every l, including l = 1; that kernel is not an out-of-bounds
read. The fault's site is between mamba1's last download and mamba2's first launch and is
not attributed; the 24.04 image's driver is the other candidate (the same image linked the
vendor-neutral CPU binding differently on 2026-09-13). So the probe below runs on BOTH a
Hot Aisle MI300X and a DigitalOcean MI325X, and a fault on the MI325X alone is an image
finding, a MOVED on the MI300X alone is the race.

Probe, for the box that showed it (MI300X): run the `mamba2-dtlimit` and `mamba2` lanes
on `base` with `--repeats 20`; if `step` moves again, dump `state.h` and the conv window
after `m2_buffer_update_kernel` for two runs and diff by element to name the index, then
read the two kernels for the write that a neighboring thread reads. If twenty repeats are
stable, record the count and leave the deviation open with this brief as its ledger row.
Owner: the peer session with the box.

## 3. DEVIATION 2712 (2026-09-14, later): the DEFAULT mamba2 lane diverges on one MI300X, stable per process

The clean 120-lane record at 65ae7612f (`bench/results/identity_break/2026-09-14_120-lanes/`)
reads 1080 of 1080 stable on Apple M4, H100 and a Hot Aisle MI300X 8core VM, no MOVED
anywhere, 1071 + 1476 cells identical on three vendors, EXCEPT the `mamba2` lane (default
dt_limit): on all nine fixtures the MI300X column's `step` and `backward` parts differ from
Apple and H100, which agree; `forward` and `prefill` agree on all three. The same lane read
IDENTICAL x3 on a 13core MI300X in the 47-lane record and in both earlier 120-lane runs the
same day, and the 8core column repeats within its run. So the AMD answer is STABLE PER
PROCESS and DIFFERENT PER BOX (or per run); section 2's one moved cell was the first sight
of it. Recorded in SUPPORT_MATRIX as DEVIATION 2712; the CPU gate keeps the 47-lane columns
until it closes.

What the two parts share. `backward` is `_prefill_backward`: it recomputes its own forward
from a ZERO state and never reads the carried state (`python/mojolearn/_mamba_impl.py:830`),
so the carried state is not the common factor. In the lane (`_block_fit`) the order is
forward, prefill, step, backward on ONE block: `step` runs after two forwards, `backward`
after three calls, on the same working stages. A kernel that reads a working row an earlier
call wrote (or one nobody wrote) would give exactly this: the first forward clean, later
calls contaminated by what the block's buffers hold, the same bytes every time in one
process (the allocator repeats itself), different bytes on a box whose allocator or
image hands over different pages. No runtime device-property geometry exists in `mamba/`
(the only core-count constants in the tree are compile-time column constants in
`core/gram_splitk.mojo`), so a per-box grid shape is not it.

The probe, `tools/mamba2_step_probe.py`, separates the three readings. It runs the lane's
exact inputs through FIVE call orders on fresh blocks (`lane`, `backward-only`,
`step-only`, `backward-first`, `step-twice`), N repeats each in one process, saves every
array of the first repeat with a hash per repeat, and diffs two runs element by element:

- a part that differs between `lane` and `backward-only` (or `step-only`) in ONE process
  is ORDER DEPENDENCE, a kernel reading rows an earlier call left;
- a part that differs between two PROCESSES on one box in the same order is a PER-PROCESS
  source, an unwritten buffer;
- a part that differs between two BOXES only is a device-dependent path.

On the Apple M4 (regenerated on demand, ten seconds, see
`bench/results/mamba2_probe/README.md`; the `.npz` is not stored, it is 6.9 MB): 86 arrays, no
in-process move, no order dependence, and a second process equal on all 86. Owed on the
Hot Aisle MI300X and the DigitalOcean MI325X:

```sh
MOJOLEARN_NUMERIC_MODE=identical python3 tools/mamba2_step_probe.py run amd-<box>-1.npz --repeats 20
MOJOLEARN_NUMERIC_MODE=identical python3 tools/mamba2_step_probe.py run amd-<box>-2.npz --repeats 20
python3 tools/mamba2_step_probe.py diff amd-<box>-1.npz amd-<box>-2.npz
python3 tools/mamba2_step_probe.py diff apple-m4.npz amd-<box>-1.npz     # apple-m4.npz regenerated on the Mac
```

The `run` exit code is 1 on any in-process move or order dependence; the first DIFFER line
of the Apple diff names the part and the first differing element, which is the address to
read the kernel at.

### 3.1 The probe's first AMD results (2026-09-14, peer session; rows in `bench/results/mamba2_probe/README.md`)

Hot Aisle MI300X, the SAME 8core VM type and 22.04 container as the divergent column, at
cdcaf7890: two processes of 20 repeats, in-process moved 0, order-dependent parts 0, the
two processes equal on all 86 arrays, and equal to a fresh Apple reference on all 86. So a
COLD process on that box computes the Apple bits through every call order. What differs
from the 120-lane run is what ran before: the identity run reaches mamba2 after about a
hundred lanes in one process. That reads as a read of device memory the lane never
initializes (a working stage, or a neighbor of the carried state) whose contents depend on
the process's earlier allocations; not a launch-order race between step and backward,
which the five orders would have shown.

DigitalOcean MI325X (24.04 ROCm image): both runs abort at the FIRST launch with "Memory
access fault by GPU node-1 ... Reason: Unknown", exit 134, the same fault that ended the
120-lane run there right after mamba1. Deterministic on that image, so it is not the 2712
race and gets its own deviation number: a kernel of the mamba2 path reading past an
allocation that the MI300X's allocator happens to back.

The probe's `--warm` option (same day) tests the uninitialized-read reading directly:
`--warm "poison:8"` allocates and frees eight 2048 x 2048 NaN GEMMs before the orders, so a
read of unwritten device memory becomes a NaN the diff cannot miss; `--warm "lanes:<names>"`
runs other identity_break lanes first in the same process, and the exact condition of the
120-lane run is the list of every lane that precedes mamba2 in LANES order. On the Apple M4
a warm run (five lanes plus four poison rounds) equals a cold run on all 86 arrays. Owed on
the MI300X: cold, `poison:8`, and the full preceding-lane list, each diffed against cold;
the first DIFFER line names the part and the element, and a NaN there is the read.

### 3.2 CORRECTION and diagnosis (2026-09-14 afternoon): 2712 is the Apple column, a NaN from unwritten device memory

Sections 3 and 3.1 attributed the divergence to the MI300X. The per-column hashes say otherwise.
In every record on main (46, 47, 118, 120-2711flip) all three vendors carry `mamba2/base` cell
5b05a3ecbd70248e, step 3fa40f29231409be, backward f252b3fc19f5f9f5. In the 65ae7612f record the
H100 and the MI300X carry exactly those; the APPLE M4 column carries b09925d3d8b074a2 /
16508d9095dbaa6a / 7c3fda003a3a0464 (forward 22f79ee009fd9884 everywhere). Every AMD run since
(the cold and warm probes, the 8core and 13core dumps) carried the canonical values and read
DIVERGENT only because the diff held them against that Apple column. No AMD box ever diverged on
this lane. The peer session confirmed and stopped the AMD legs.

Reproduced on the Apple M4 (this Mac, a clean worktree build, the canonical Mamba binding
41962f89, single-threaded): `identity_break.py --lanes <the 42 lanes that precede mamba2>,mamba2
--fixtures base --repeats 2` with `MOJOLEARN_IDENTITY_DUMP_DIR` set gives mamba2/base
b09925d3d8b074a2, both fits, the Apple column's value. The same lane alone in a cold process gives
5b05a3ecbd70248e. Element diff of the warm dump against the cold dump: `x`, `g`, `forward`,
`prefill`, and the carried state after prefill and after step (h, conv_window, buffer_xbc,
buffer_dtraw) are EQUAL; `step` (64 of 64) and EVERY backward gradient (x, block_norm.weight,
in_proj.weight, conv1d.weight, conv1d.bias, dt_bias, A_log, D, norm.weight, out_proj.weight; all
elements) are NaN, and every NaN is the canonical quiet NaN 0x7fc00000. Bisection: the first 21
lanes then mamba2 gives the canonical value; lanes 22 to 42 then mamba2 gives the canonical value;
all 42 give the NaN. So it is not one lane; it is the process's device-allocation history, which is
the signature of a read of device memory nothing wrote (an allocator that hands back a recycled
block still holding a NaN, on Metal; CUDA and HIP handed back zeros in every run so far, and a
zero read is invisible).

Ruled out on the way, each by a check that could have failed: a launch-order race (five call orders
equal, section 3.1); a device-dependent path (the cold probe equal to Apple on both MI300X VM
types); runtime launch geometry (none in `mamba/`); the numpy view of a block output outliving
its buffer (`Array` pins its store through the buffer protocol; hash unchanged after freeing the
output and allocating 2000 NaN blocks); an asynchronous download (`mamba_download` synchronizes
before and after and copies through a host list); the Mamba binding's bytes (the 33b9b74e build
from the 2711 worktree made the bad Apple column AND the good 2711flip column; 41962f89 reproduces
the bad value warm and the good value cold).

Where the read is. Backward: `mamba/impl/ops/mamba2_ssd_backward.mojo` allocates FORTY device
buffers with `enqueue_create_buffer` and no fill (lines 63-78, 379-384 and the discretize and
conv backward states), one of them `d_c_yoff` sized `b * nc * 256 * N` with the comment "the
launcher writes only real T rows", and `mamba2_postconv_merge_kernel` (`:553-562`) sums
`d_c_yoff` over its full `c_cells` extent; `mamba/impl/modules/mamba2_backward.mojo` allocates 20
more the same way. Every forward stage, by contrast, is `mamba_zeros` (`mamba2.mojo:244-320`).
Step: not located yet. Every buffer the l = 1 resumption reads is zero-filled or uploaded, and
the assembly kernels index the M-sized stages correctly (`m2_assemble_xbc_kernel`); the remaining
candidates are the kernels that map chunk rows back to the M output rows (`m2_skip_kernel`,
`m2_gate_kernel`, the out-proj and residual at row q0 + li) reading a row beyond M or beyond
t_work, and DEVIATION 2713 (the MI325X faulting at the FIRST mamba2 launch, a plain forward) says
an out-of-bounds read exists on the forward path too, benign where the page is mapped.

The fix that cannot be argued with: allocate every Mamba-2 device buffer through `mamba_zeros`
(60 call sites), then a POISON build (`-D MOJOLEARN_MAMBA_POISON=1`: fill every fresh device
buffer with 0x7fc00000 instead of 0) under which the identity lane must read the canonical
hashes cold on every vendor; a lane that still reads NaN under poison names the remaining read.
Owner: the peer session (Mojo, three vendors); the harness side is done.

### 3.3 THE CAUSE (2026-09-14 evening, found by the peer session's poison-and-band gate): one over-read in the Mamba-2 forward

`m2_ydiag_kernel` (`mamba/impl/modules/ssd_minimal.mojo`, seams S13 and S14, Y_diag = M . X_d)
loops `jj` over the chunk width Q = 256 and loads the X_d row `c * Q + jj` for EVERY jj, rows at or
past T included (X_d is [B, T, H, P]; T = 16 in the lanes, 1 in the step), so it reads past the end
of the allocation. The mask value there is the structural +0.0, and 0 x finite garbage is 0: that
is why every CUDA and HIP run and every cold Metal run carried the same bits for weeks. 0 x NaN is
NaN: Metal in a warm process (DEVIATION 2712; the step reads farthest past its end since its T is
1, which is why step and backward went first). And the row past the last allocation is an unmapped
page on the DigitalOcean MI325X: DEVIATION 2713, at the first forward launch. The block check under
poison named it: `ydiag.out` MOVED 256 of 256 cells, first cell 0, a 0x7fc00000, with seg.L, cb.G,
xd.out and dacs.out OK before it and everything after it NaN.

Fix: read X_d only when `c * Q + jj < t_work`, +0.0 otherwise, which is the bits every record
carries. With the fix, poison plus band, cold on the M4: 36 of 36 Mamba training rows and 36
inference rows IDENTICAL x4 against the 2711flip record's three columns (mamba1, mamba2,
mamba2-dtlimit, mamba3; nothing else moved under the band). Owed as of this line: the three
sabotage modes, the H100 and MI300X legs of the gate, the poison build on the MI325X (2713 should
now launch), then the three-column rerun that becomes the CPU gate's columns.

What closed it, in order, each a check that could have failed: the per-column hashes against
earlier records (which column moved), the harness's per-fit dump in the warm process (which
arrays, and that they were NaN, not garbage), the bisection (not a lane, the allocation history),
the three-fill split with a guard band under a poison define (an over-read, not an unfilled
buffer), and the block check under poison (the stage). Two sessions, one afternoon, four AMD legs
spent on the wrong vendor before the first of those.
