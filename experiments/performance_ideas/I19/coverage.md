# I19 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual stable UInt32 radix pair sort over bounded logical key widths, repeated scratch reuse, duplicate keys and tails, checking secondary original positions and untouched storage.

Remaining original-card scope: A new explicit ragged float-word adapter now carries original positions through stable radix grouping under existing x_prep public NaN/raw-payload/signed-zero/FTZ and categorical canonicalization policies, with caller-owned scratch reuse and empty/tail fixtures. The new resident ragged quantile preparation consumer runs the existing pinned quantile_unit directly after radix grouping and compares full interpolation outputs against canonical host sorting/replay, including empty segments and fractions0/.25/.5/.75/1. Selection-only/bootstrap/encoder integrations remain separate unimplemented arms; rank control refuses segments above4096.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
