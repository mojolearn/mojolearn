# I19 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual stable UInt32 radix pair sort over bounded logical key widths, repeated scratch reuse, duplicate keys and tails, checking secondary original positions and untouched storage.

Remaining original-card scope: A new explicit ragged float-word adapter now carries original positions through stable radix grouping under existing x_prep public NaN/raw-payload/signed-zero/FTZ and categorical canonicalization policies, with caller-owned scratch reuse and empty/tail fixtures. Selection-only callers and every quantile/bootstrap/encoder integration remain separate unimplemented arms; rank control refuses segments above4096.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
