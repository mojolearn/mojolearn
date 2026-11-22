# Final Linux HDBSCAN host-artifact validation

The final `819a47ae48166e91951f54f54e64ee173658e32a` Linux build's actual
`_mojolearn_hdbscan_host.so` was exercised on x86-64 after the owner-lifetime
fix. Both `hdbscan` and `hdbscan-leaf` completed all nine fixtures, one fit
per cell, with all default applicable parts: 18 STABLE cells and 72 exact
numeric matches to the CPU reference. The CPU backend was required, even
though the rental also has an AMD GPU. Guard exit zero; runtime 8.04 seconds.

The full native build passed its source/physical-device and build-proof
checks. All 61 fetched extension hashes matched the manifest, including all
32 host bindings. The only host binding whose bytes changed from the
preceding `a97676ba1` build was `_mojolearn_hdbscan_host.so`.

Raw execution log and JSON are retained at
`~/mojolearn-evidence/final819-hdbscan-cpu/`; the final native build is at
`~/mojolearn-evidence/releases/819a47ae48166e91951f54f54e64ee173658e32a/hip-gfx942/release-build/`.
