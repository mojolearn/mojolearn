# I22 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual packed TSQR device/host R factors and retained Householder Q application across panel tails, plus Cholesky residual/pivot/signed-zero/subnormal gates.

Remaining original-card scope: New strip trailing updates, factorization reuse and changed TSQR combine arity remain unimplemented. New QR trees and skipped structural-zero arithmetic need separate complete numerical contracts.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
