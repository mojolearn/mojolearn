# Additional Apple vendor references

Full-part collection of 38 lanes across all nine fixtures, one repeat each. All 342 cells completed. The 1647 numeric parts having CPU references match those references exactly. This does not claim repeated-run stability from a single repeat.

`apple.json` retains source witness c738f526163cc02ec6b7e1255a2dfb70923941b1, native artifact hashes, and resume signature. The native binaries were freshly compiled at a97676ba18a09fb577ef9faae45ab19a01eec848; arithmetic for these 38 lanes is unchanged in the released 819a47ae48166e91951f54f54e64ee173658e32a source. `cpu-comparison.json` identifies and hashes the CPU reference table used.

This supplemental record was paused twice at saved cell boundaries to prioritize exact-wheel release qualification, then resumed using the harness provenance checks. It does not modify the published wheel. The collection used `MOJOLEARN_APPLE_FULL_DIAGNOSTIC=1`, `--require-backend metal`, `--repeats 1`, and all default parts.
