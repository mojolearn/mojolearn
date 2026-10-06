# I20 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual six-kernel KDE public scoring paths, all three supported metrics, bandwidth extremes, weights and staged/fused/transposed schedules with canonical exact score bits plus existing chunk-fold replay. Opt-in `MOJOLEARN_IDN_KDE_PARTIAL_POOL=1` adds a fit-owned 64 MiB scratch budget for the existing chunked fold. Actual resident `score_samples` reuses partial/log-weight/likelihood storage under its immutable fit handle; every numerical value is recomputed on each call. Oversized calls retain the smaller pool and use the original temporary-buffer path; explicit source rebinding drains and invalidates storage. An atomic lease sends concurrent public scores to the original temporary path. Direct in-flight/closed/foreign-source uses refuse, and release drains before destroying the owning context.

Remaining original-card scope: A new log-sum-exp pair-combine profile is not introduced; the card explicitly treats it as a separate numerical-profile proposal, and this experiment preserves the current logical fold. Full public scoring allocation/preparation/readback timing is owed. Pool baseline and candidate use the same current chunked numerical profile; the old chunk-fold rollback is not mixed into the pooling A/B.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
