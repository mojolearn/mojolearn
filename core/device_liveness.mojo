# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The dead-device canary: a fit must RAISE, never return silent garbage.

================= DEVIATION 2002 =================
NOT A PORT. CatBoost and cuML both assume a CUDA runtime whose API calls
return error codes and whose host wrappers throw on them; a dying CUDA
context surfaces as an exception long before a fit returns. On this
toolchain that assumption is FALSE and it was measured false: during the
DEVIATION 134 loaded window (2026-09-01, nine concurrent GPU fit
processes, 1-min load 18, Metal spamming "Context leak detected"), every
process's fits went to silent garbage SIMULTANEOUSLY -- enqueues stopped
executing, host readbacks stopped delivering, and each fit RETURNED a
coherent-shaped empty model (greedy 0 splits, mse -0.0) as if training
had succeeded. No call raised. PORTING.md's DEVIATION 2002 entry is the
ledger; this file is the fix's shared mechanism.

THE MECHANISM, three legs, all riding the number 2002:

1. THE NONCE CANARY (the greedy symmetric driver,
   `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`):
   per tree, a one-thread kernel -- enqueued AFTER every launch of the
   tree, on the same in-order queue -- writes `MAGIC ^ nonce` into a
   spare slot of a buffer the tree's one drain already copies home. The
   host poisons the landing slot first and compares after the drain.
   On a live device the read-back value IS the expectation: the queue
   is in order, the canary launch precedes the copy, the copy precedes
   the drain the host just returned from. Anything else means the
   device did not execute this tree's command stream (poison = the
   copy never ran; a stale or zero word = the kernels never ran), and
   the fit raises instead of returning the empty model. The nonce is
   what makes tree N's leftover magic unable to vouch for tree N+1.

2. THE POISONED LOSS WORD (`gbdt/methods/doc_parallel_boosting.mojo`):
   the boosting loop's final `functionValue` readback lands in a host
   word the host poisons first. Poison surviving the drain = the loss
   was never delivered = raise. This is the whole-fit backstop that
   covers every grow policy the boosting loop drives, not only the
   symmetric searcher.

3. `assert_device_alive` (below): the end-of-fit form for the forest
   families (extratrees, ensemble/randomforest), whose fits are
   host-driven loops of per-cycle readbacks -- on a dead device those
   deliver stale or zero split records and every node quietly becomes
   a leaf, so the forest comes back as well-formed stumps. One call at
   the end of the fit: reset a device word, launch the canary kernel,
   read the word back, raise on mismatch. Costs one tiny kernel, one
   4-byte copy and one drain PER FOREST FIT, which is noise.

WHY THIS CANNOT FALSE-POSITIVE ON DEGENERATE DATA: no leg looks at the
MODEL. Constant labels, min_data refusals, unsplittable folds all leave
every canary green, because the canary witnesses only that the device
executed the command stream and delivered the readbacks -- which a live
device always does, whatever the data. The split-count question ("is an
empty model legitimate?") is deliberately NOT asked; the incident's
empty models were garbage BECAUSE the stream never ran, and that is the
thing witnessed directly.

THE SABOTAGE ARM, REQUIRED-RED. Build with

    -D MOJOLEARN_2002_SABOTAGE=1

and every canary-writing launch (leg 1's per-tree kernel, leg 3's
helper kernel) plus leg 2's final-loss copy compiles out -- the exact
dead-device signature, faked: the device is fine, but the witnesses the
checks demand never arrive. An armed run of ANY gated fit must FAIL
with a "DEVIATION 2002" error for the gate to pass; an armed run that
returns a model means the validation is not wired. By construction the
armed verdict is deterministic (the only writer of the witnessed word
is compiled out; the expectation is fresh per call) and the arm
terminates (it deletes work, adds none). The define must never reach a
shipped build; nothing but the gate invocation passes it. Verdict
inversion follows the DEVIATION 134f house pattern
(`checks/determinism_soak.mojo`).
==================================================
"""

from max.gpu.host import DeviceContext
from std.sys.compile import is_defined

comptime DEAD_DEVICE_CANARY_MAGIC = UInt32(0x2002A11E)
"""The canary's base word ("2002 alive"). Leg 1 XORs a per-tree nonce in;
leg 3 uses it bare, its device word freshly zeroed the same call."""

comptime DEAD_DEVICE_POISON = UInt32(0xDEAD2002)
"""What the host plants in a readback slot before the copy that must
overwrite it. Chosen collision-free by construction against every value a
live device can put there: leg 1's slot holds `MAGIC ^ nonce` (never
0xDEAD2002 -- the nonce would have to be MAGIC ^ 0xDEAD2002 exactly, and
nonces count up from 1 within a fit, orders of magnitude away), leg 2's
holds a Float32 loss accumulator (bit-compared, and `-w * diff^2` sums
reaching these exact bits would need a loss near -1.57e18)."""

comptime SAB_2002_DEAD_DEVICE = is_defined["MOJOLEARN_2002_SABOTAGE"]()
"""`-D MOJOLEARN_2002_SABOTAGE=1`: fake the dead-device signature on a
live device. REQUIRED-RED; full banner in the file docstring."""


def write_canary_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    slot: Int32,
    value: UInt32,
):
    """One thread, one store: `dst[slot] = value`.

    Launched grid (1,1,1) x block (1,1,1) so no guard is needed. Enqueued
    LAST, its store is ordered behind every kernel of the fit on the
    in-order queue, which is the entire point: reading `value` back
    through a later copy witnesses that the queue executed to here.
    """
    dst.unsafe_store(Int(slot), value)


def assert_device_alive(ctx: DeviceContext, where: String) raises:
    """End-of-fit dead-device check for the host-loop forest families.

    Reset a device word to 0, launch `write_canary_kernel` over it, copy
    it back into a poisoned host word, drain, compare. A live device
    always returns the magic (in-order queue: upload, kernel, copy,
    drain). Poison back = the copy never ran; 0 back = the upload ran
    but the kernel did not; anything else = garbage delivery. All three
    are the DEVIATION 2002 signature and raise.

    Costs one drain, so call it ONCE PER FIT, after training -- never
    per tree or per level (the greedy driver's leg 1 exists because a
    per-tree check must ride the tree's own drain instead).
    """
    var canary = ctx.enqueue_create_buffer[DType.uint32](1)
    var h_seed = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h_read = ctx.enqueue_create_host_buffer[DType.uint32](1)
    h_seed.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_read.unsafe_ptr().unsafe_store(0, DEAD_DEVICE_POISON)
    ctx.enqueue_copy(dst_buf=canary, src_ptr=h_seed.unsafe_ptr())
    comptime if not SAB_2002_DEAD_DEVICE:
        ctx.enqueue_function[write_canary_kernel](
            canary.unsafe_ptr(),
            Int32(0),
            DEAD_DEVICE_CANARY_MAGIC,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
    ctx.enqueue_copy(dst_ptr=h_read.unsafe_ptr(), src_buf=canary)
    ctx.synchronize()
    # reads AFTER the drain; the transfers below hold every buffer past
    # it ([[mojo-buffer-freed-at-last-use]], the step-33 race class)
    var got = h_read.unsafe_ptr().unsafe_load(0)
    _ = h_seed^
    _ = h_read^
    _ = canary^
    if got != DEAD_DEVICE_CANARY_MAGIC:
        var observed = String("an unexpected word (")
        if got == DEAD_DEVICE_POISON:
            observed = String(
                "the host-side poison -- the device-to-host copy never"
                " ran ("
            )
        elif got == UInt32(0):
            observed = String(
                "the reset value 0 -- the canary kernel never executed ("
            )
        raise Error(
            "DEVIATION 2002: refusing to return this model -- the device"
            " did not execute the fit's command stream ["
            + where
            + "]. The end-of-fit canary read back " + observed
            + String(got) + " where " + String(DEAD_DEVICE_CANARY_MAGIC)
            + " was required). This is the dead/saturated-device"
            " signature (Metal context death under load): fits in this"
            " state return coherent-shaped but EMPTY models with no"
            " error, so the fit raises instead. The process's GPU"
            " context is not trustworthy; restart the process on a"
            " quieter box."
            + (
                " [SABOTAGE ARM -D MOJOLEARN_2002_SABOTAGE=1 IS ARMED:"
                " this failure is the REQUIRED-RED verdict.]"
                if SAB_2002_DEAD_DEVICE else ""
            )
        )
