# I19 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual stable UInt32 radix pair sort over bounded logical key widths, repeated scratch reuse, duplicate keys and tails, checking secondary original positions and untouched storage.

Remaining original-card scope: This narrower unsigned key contract does not establish floating NaN/signed-zero public policies, segmented/ragged layouts, selection-only callers or every quantile/bootstrap/encoder integration.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
