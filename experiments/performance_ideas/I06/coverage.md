# I06 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual fused attention/GQA row, backward and KV-grid schedules across tails, masks and repeated cases, with canonical stage bits and required launch tokens.

Remaining original-card scope: New default-off two-head GQA forward sharing now interleaves64logical rows while retaining32queries per head, staging each compatible KV tile once. Admission requires even GQA groups and the declared27,904byte shared page fitting the column; other regimes retain the existing kernel. Online-softmax arithmetic remains a separate numerical profile.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
