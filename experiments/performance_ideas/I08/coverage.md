# I08 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual SSD tile/lower-triangle controls with complete host-stage replay, changed lengths and state cases.

Remaining original-card scope: New default-off retained G⊙L arm materializes canonical product words in a16 MiB bounded buffer and feeds the existing tiled arithmetic, with explicit lifetime drain and exact fallback above the bound. Decay reuse remains provided by the existing retained decay stage; no new shared-state extension is implemented. All original recorded/backward inputs remain intact; additional allocation/traffic/wait must be counted in whole-op timing.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
