# AMD neural full-part columns, 2026-09-20

The MI325X gfx942 column was collected from freshly built IDENTICAL GPU
bindings at `a97676ba18a09fb577ef9faae45ab19a01eec848`, using the release
build set on the same rental. No artifacts or source commits were relabeled.
The release build independently passed its physical/source preflight and
build-proof check before this run.

All 19 requested lanes completed across all nine fixtures, one fit per
cell, using the full default part set: 171 STABLE training cells. Comparing
the recorded numeric parts against the current CPU reference columns found
1,044 exact matches and no discrepancies. The record retains the harness's
reported, unclaimed serial-fold backward-weight differences; they were not
turned into an identity claim or filtered from the evidence.

The lanes cover byte LM (ordinary and resident), Mamba 1/2/3 and their
BF16/int8 variants, Mamba 2 dt-limit, MLP BF16/int8, Samba BF16/int8,
Transformer BF16/int8, and Mamba 1/Transformer decode sessions. The JSON
contains the full fixture/lane list, protocol, parts and package witnesses.

Run command and raw log are retained under
`~/mojolearn-evidence/amd-neural-gap-columns-a976/`; the native build proof
and all 61 extensions (32 CPU bindings) are retained under
`~/mojolearn-evidence/releases/a97676ba18a09fb577ef9faae45ab19a01eec848/hip-gfx942/release-build/`.
The bounded AMD guard exited zero after 114.5 seconds, with peak RSS 4.24 GiB
and peak device memory 10.01 GiB.
