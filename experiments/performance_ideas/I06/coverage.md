# I06 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual fused attention/GQA row, backward and KV-grid schedules across tails, masks and repeated cases, with canonical stage bits and required launch tokens.

Remaining original-card scope: A new cross-head shared K/V-load kernel is not implemented; this is the focused executable comparison of already-present tile schedules. Online-softmax arithmetic remains a separate numerical profile.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
