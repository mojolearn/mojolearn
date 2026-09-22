# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one cross-block device mutex claim. Every claim site in this repository calls it.

WHY THIS FILE EXISTS AT ALL. Until 2026-09-16 the claim was written out by hand at six
sites (`ensemble/decisiontree/batched_levelalgo/split.mojo` `_publish_to_global`,
`extratrees/impl/decisiontree/batched_levelalgo/split.mojo`,
`neighbors/impl/detail/fused_l2_knn.mojo` twice, and `neighbors/mutex_probe_main.mojo`
twice). A defect in the protocol was therefore a defect in three subsystems at once, which
is exactly what happened. One definition, one check (`core/device_mutex_check.mojo`).

THE PROTOCOL. Spin on an ACQUIRE load until the mutex reads `available`, take it with a
WEAK RELAXED compare-exchange, then execute an ACQUIRE FENCE. The holder hands the mutex
back with a RELEASE store, which is the caller's job and is not in this file.

WHY THE FENCE. The spin load and the claiming compare-exchange are two separate reads of
the mutex word and they need not observe the same release. An acquire synchronizes with
the release whose value it reads; the spin's acquire therefore synchronizes with whichever
release wrote the free value it happened to see, and the relaxed claim synchronizes with
nothing at all. So a thread can leave the spin on a free value written by holder A, lose
that value to holder B, and then win its relaxed claim on the free value written by B's
LATER release, having performed no acquire that reads B's release. B's plain stores to the
payload are then unordered against this thread's plain loads of it: this thread merges
into a STALE payload and writes the whole thing back, and B's contribution is ERASED. The
observed symptom is a LOST CANDIDATE in one direction only, which is what the MI300X
random forest traces showed.

The fence repairs it. A compare-exchange is a read-modify-write, so its write sits inside
the release sequence headed by whichever release it consumed. An acquire fence executed
after an atomic operation that read a value in a release sequence establishes the same
synchronizes-with edge that an acquire on the operation itself would. The previous
holder's payload stores therefore happen-before this holder's payload loads. The window is
not narrowed, it does not exist.

WHY A FENCE AND NOT AN ACQUIRE LOAD, AND WHY NOT AN ACQUIRE CAS. Both alternatives were
tried and both are wrong here, for different reasons, and this paragraph exists so nobody
spends a fourth day rediscovering either.

  * A DISCARDED ACQUIRE LOAD AFTER THE CLAIM EMITS NOTHING. `_ = Atomic.load[ordering =
    Ordering.ACQUIRE](mutex)` produced ZERO instructions on Metal AIR, PTX and GCN alike
    (measured 2026-09-16: two arms of `_mojolearn_rf.so` and two of `_mojolearn_trees.so`
    differing only by that line had bit-identical `__TEXT` and `__DATA`). The compiler
    drops it because nothing consumes its value. Every acquire load in this repository
    that does emit uses its value. A discarded atomic is not a fence. Do not write one
    here and do not believe a comment that says the compiler will keep it.
  * AN ACQUIRE COMPARE-EXCHANGE is formally equivalent and is what the reference's
    `atomicCAS` plus `__threadfence` translates to, but the Apple backend rejects
    `acquire` on EVERY read-modify-write by name, so it needs a per-vendor spelling and
    leaves Apple on a second protocol. One spelling for every column is the whole point.

`std.atomic.fence` IS NOT `std.gpu.intrinsics.threadfence`. They are different symbols.
DEVIATION 106 reasoned from `threadfence` being comptime-asserted NVIDIA-only and
concluded that no fence was available on this path. That conclusion was wrong and it is
why three separate agents went looking for an ordering they could hang off an existing
atomic operation instead of simply fencing. `std.atomic.fence` compiles and EMITS on all
three columns: `fence acquire` in the disassembled metallib, and `buffer_inv sc0 sc1` on
gfx942 where the stock build has nothing.

REQUIREMENTS ON THE CALLER. Unlock with `Atomic.store[ordering = Ordering.RELEASE]`. Only
a successful holder may unlock. An unsuccessful claimant must not modify the mutex. The
held state must differ from every currently claimable state. Initialization must complete
before the users launch. There is no fairness or residency guarantee, so a kernel must
separately ensure that spinning blocks can make progress.

WHAT THIS FILE DOES NOT CLAIM. This is the C++ and LLVM release-sequence argument plus a
measurement that the fence reaches the ISA. It is not a claim that any backend's lowering
has been proved correct, and it is not a substitute for the device checks.
"""

from std.atomic import Atomic, Ordering, fence
from std.sys.compile import is_defined


#: A/B CONTROL, NEVER A DEFAULT. `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` compiles the
#: PRE-REPAIR claim, with no fence, for every caller. It exists so a leg has an arm that
#: has to be SEEN to move. It is not a matrix row.
comptime CLAIM_STOCK = is_defined["MOJOLEARN_RF_MUTEX_CLAIM_STOCK"]()

#: BUILD-ONLY POSITIVE CONTROL FOR THE SECTION COMPARATOR, NEVER RUN AND NEVER A DEFAULT.
#: `-D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1` writes a different held value, which must move
#: `__TEXT,__text`. Its only job is to prove that a byte comparison of two arms is able to
#: report DIFFER at all, because the previous guard on that comparison could not fail (a
#: whole-file digest over a binary carrying a fresh `mktemp` install name always differs).
#: The protocol still works under it, since only the holder writes the held value.
comptime SECTION_SABOTAGE = is_defined["MOJOLEARN_MUTEX_SECTION_SABOTAGE"]()

#: MEASUREMENT INSTRUMENT, NEVER A DEFAULT. `-D MOJOLEARN_MUTEX_SPIN_RELAXED=1` makes the
#: SPIN TEST a relaxed load instead of an acquire load. It exists to answer, on a column
#: with no AMDGPU disassembler to hand, whether the spin's ACQUIRE ORDERING survives into
#: the instruction stream at all. If this arm is byte-identical to the default, the
#: acquire on the spin emits nothing on that column and the shipped spin-wait has been
#: resting on an ordering it never had. If it differs, the acquire is real. This is a
#: differential build, and it answers the question without disassembling anything.
comptime SPIN_RELAXED = is_defined["MOJOLEARN_MUTEX_SPIN_RELAXED"]()


@always_inline
def claim_device_mutex[origin: MutOrigin, //](
    mutex: MutPointer[Int32, origin], available: Int32, held: Int32
):
    """Take a cross-block device lock. Supports both the `0 -> 1` and `-2 -> -1` claims."""
    var claim_value = held + Int32(1) if SECTION_SABOTAGE else held
    while True:
        comptime if SPIN_RELAXED:
            if Atomic.load[ordering = Ordering.RELAXED](mutex) != available:
                continue
        else:
            if Atomic.load[ordering = Ordering.ACQUIRE](mutex) != available:
                continue
        var expected = available
        if Atomic.compare_exchange[
            success_ordering = Ordering.RELAXED,
            failure_ordering = Ordering.RELAXED,
            weak=True,
        ](mutex, expected, claim_value):
            # The edge is made HERE, with the release this thread actually claimed
            # against, not with whichever one its spin load happened to observe. Do not
            # replace this with a discarded acquire load; that spelling emits nothing.
            comptime if not CLAIM_STOCK:
                # MEASURED 2026-09-16: `fence[ordering = Ordering.ACQUIRE]()`
                # generates valid AIR and then FAILS AT PIPELINE CREATION on
                # Apple, "GPU machine code generation ...
                # XPC_ERROR_CONNECTION_INTERRUPTED", 6 of 6 with arm order
                # rotated, against stock 6 of 6 OK. It broke random forests,
                # extratrees and the fused kNN on this vendor. Every
                # cross-compile check passed, because AIR generation is not the
                # stage that fails.
                #
                # This is an acquire LOAD whose value is CONSUMED, which is the
                # distinction that makes it emit: a discarded one compiles to
                # nothing on AIR, PTX and GCN alike. It reaches the SAME edge as
                # the fence. The claim above is a read-modify-write, so its write
                # sits in the release sequence headed by the release it consumed,
                # and an acquire that reads a value in that sequence synchronizes
                # with the release heading it.
                #
                # `claim_value`, NEVER a literal 1: the kNN consumer claims
                # `-2 -> -1`, and a loop waiting on the wrong value does not fail,
                # it HANGS.
                while Atomic.load[ordering = Ordering.ACQUIRE](mutex) != claim_value:
                    pass
            return
