# KDE registry and workspace ownership

The existing resident KDE path releases the GIL around preparation and scoring.
Its process registry held a mutable Dict of ResidentKdeFit values, and scoring
borrowed one entry across kernel calls while another fit could rehash the Dict
or release that handle. The existing workspace's atomic lease protected its
scratch use, but could not keep the surrounding Dict value alive.

The registry now protects publication, lookup and removal with a short native
metadata lock. Its SharedSlot layout/name is version 2. Preparation still builds
and uploads the fit outside the lock; release pops the entry and closes/drains
it outside the lock. A query copies only reference-counted native context and
DeviceBuffer handles under the lock. It never holds a Dict entry borrow during
numerical execution. These copies share the existing device allocations and
copy no training or weight bytes.

The same short lookup moves the optional retained workspace into the query's
native lease. An overlapping eligible query creates independent workspace;
ordinary sequential calls reuse the original workspace and its counters. A
completed lease returns one idle workspace to a still-live handle. A second
idle workspace or a workspace whose handle was released is closed outside the
lock. Failed execution drains and discards its workspace, preserving the
original execution error if cleanup also fails. Validation-only failures return
an untouched lease. No lock spans kernels, DMA, host data copying or device
synchronization, and no whole-fit/query lock is introduced.

Unchanged: direct-preparation and immutable-fit switches/defaults, fit and query
uploads, weights staging, fit validation order, kernel and metric parsing,
resident parameters, pooled numerical calls, score sum, status copy, output
copy and ordinary synchronization sequence. No numerical kernel was edited.
This addresses Mojolearn's registry lifetime race, separately from the reported
Modular shared-context concurrent-submission problem.

The existing `kde/checks/reused_workspace_check.mojo` sequential allocation check
should still observe four retained allocations and one reuse after its two
queries. Root owns compilation and execution; this lane has run neither. New
ownership coverage should exercise publication/removal while another lease is
held, exclusive workspace ownership, return after handle release, and reuse
without additional fit uploads. No performance or identity claim is made.

Original source and pre-edit invariants are saved under
`~/mojolearn-evidence/python-mojo-migration-20261110/kde-native-owner/` in
`before-resident-fit.mojo` and `SCOPE.md`.
