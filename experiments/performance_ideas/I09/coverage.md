# I09 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual token-parallel convolution/prefill versus public decode, using shared planted weights at multiple state widths and neighboring prefix tails.

Remaining original-card scope: Fixed absolute-position chunking and shared backward suffix reductions remain blocked on a complete versioned host/decode/checkpoint/backward contract. This commit does not establish that new numerical profile.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
