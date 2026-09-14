# BRIEF, the two MOVED lanes of the 118-lane record (2026-09-14)

The 118-lane three-column run at 71faae781 (`bench/results/identity_break/2026-09-14_118-lanes/`)
carried two MOVED lanes among the 71 the claim-surface census added. MOVED is one box
disagreeing with itself across two fits in one process, a determinism finding before it is
a cross-vendor one. One was the lane's own hashing, fixed here; one is open on the MI300X.

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

## 2. mamba2-dtlimit, `step` part MOVED once on the MI300X (base, one of nine fixtures): OPEN

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
