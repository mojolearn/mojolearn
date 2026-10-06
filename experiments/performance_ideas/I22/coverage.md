# I22 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual packed TSQR device/host R factors and retained Householder Q application across panel tails, plus Cholesky residual/pivot/signed-zero/subnormal gates.

Remaining original-card scope: Guarded explicit bounded16 MiB factor-state reuse now permits repeated RHS applications using unchanged canonical reflectors; fixture compares one factor+3RHS calls against fresh host replay per RHS at panel tails. A default-off two-chunk strip now reuses each canonical T tile across independent trailing columns, with unchanged per-column folds and barriers; the fixture adds d33 strip/panel tails. Changed TSQR combine arity remains a separate numerical-profile prerequisite. New QR trees and skipped structural-zero arithmetic need separate complete numerical contracts.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
