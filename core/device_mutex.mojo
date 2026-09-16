# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device mutex claim using only weak relaxed CAS and acquire loads.

The acquire MUST follow the successful claim. A pre-claim acquire load
can observe an older unlock than the CAS, even with coherent atomics:
another holder can finish an entire critical section between the two.

After the successful CAS, write-read coherence forces this thread's load
to observe its own RMW (or a later modification). Until this holder hands
off, no other participant may modify the lock, so it observes its own RMW.
That RMW extends the immediately preceding unlock's release sequence.
The acquire load therefore synchronizes with that unlock and orders its
payload stores before the new holder's payload accesses.

Requirements: unlock with Atomic.store[Ordering.RELEASE]; only a successful
holder may unlock; unsuccessful claimants must not modify the mutex; the
held state must differ from every currently claimable state. Initialization
must finish before launching users. There is no fairness or residency
guarantee: kernels must separately ensure spinning blocks can make progress.

This is the C++/LLVM release-sequence argument, not a claim that a compiler
or GPU has been validated merely by accepting the source. See
docs/lanes/AMD_MUTEX_ORDERING_REVIEW.md for evidence and limitations.
"""

from std.atomic import Atomic, Ordering


@always_inline
def claim_device_mutex[origin: MutOrigin, //](
    mutex: MutPointer[Int32, origin], available: Int32, held: Int32
):
    """Acquire a device lock; supports both 0 -> 1 and -2 -> -1 claims."""
    while True:
        if Atomic.load[ordering = Ordering.RELAXED](mutex) != available:
            continue
        var expected = available
        if Atomic.compare_exchange[
            success_ordering = Ordering.RELAXED,
            failure_ordering = Ordering.RELAXED,
            weak=True,
        ](mutex, expected, held):
            # Do not move this above the CAS or remove its unused value.
            _ = Atomic.load[ordering = Ordering.ACQUIRE](mutex)
            return
