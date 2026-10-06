# I22 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual packed TSQR device/host R factors and retained Householder Q application across panel tails, plus Cholesky residual/pivot/signed-zero/subnormal gates.

Remaining original-card scope: Guarded explicit bounded16 MiB factor-state reuse now permits repeated RHS applications using unchanged canonical reflectors; fixture compares one factor+3RHS calls against fresh host replay per RHS at panel tails. A default-off two-chunk strip now reuses each canonical T tile across independent trailing columns, with unchanged per-column folds and barriers; the fixture adds d33 strip/panel tails. Changed TSQR combine arity remains a separate numerical-profile prerequisite. New QR trees and skipped structural-zero arithmetic need separate complete numerical contracts.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Independent executable arms: `campaign.py` renders/builds/validates incumbent
(grid/norm on), reuse only, strip only, combined, and legacy grid/norm-off.
The manifest control is the incumbent and its candidate is combined. Each arm
has its own binary/log/receipt. The campaign validates R/Q replay and lifetime
reach; it deliberately does not time its correctness fixture. Full-operation
timing must compare reuse and strip individually before attributing a combined
gain. Apple remains an identity witness only.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
