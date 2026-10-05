# Resident GEMM screen — 2026-10-04

Source `fa39073607e3b19c4fbfe063a5545a63318f13c2`, M3 tag
`resident-catalog-t-v1`. All 66 records completed, output words exactly match
the incumbent. One fresh process and one scored call per shape/variant; no
warmups, repetitions or opponent measurements. Context, allocations and input
uploads completed before timing; measured call includes output download,
completion and first full host read. Pipeline is not warmed. These are not
pure kernel or estimator timings.

| Matrix operation | M,N,K | Incumbent ms | Lowest observed variant | Variant ms | Ratio |
|---|---|---:|---|---:|---:|
| dense-nn | 2048,512,512 | 2.274125 | G1 | 1.951042 | 0.858 |
| square-nn | 1024,1024,1024 | 2.049292 | incumbent | 2.049292 | 1.000 |
| tall-projection-nn | 32768,64,220 | 6.826875 | G2 | 3.518250 | 0.515 |
| lowwidth-kmeans-nt | 32768,8,220 | 1.203750 | G2 | 0.865625 | 0.719 |
| odd-nt | 4097,71,221 | 1.701791 | G7 | 1.085459 | 0.638 |
| gram-nt | 1024,1024,220 | 3.563291 | G1 | 2.027875 | 0.569 |

Individual leads for follow-up, not universal or broad-regime guarantees:
G1 dense rectangular, tall projection and Gram; G2 narrow NT. Tall G1 and G2
are essentially tied in this single sample (3.524 versus3.518ms); do not claim
that difference is meaningful. G1 is within about3% of G7 on oddNT, which also
cannot establish a reliable ordering. All squareNN variants were slower here.
G5 tall3.602ms and G9 Gram2.332ms still improve the incumbent at these shapes,
but the earlier cold-screen rankings do not establish resident rankings.

Next: actual caller quality/timing and neighboring shapes. Decomp/PCA adapters
must preserve strides, caller-specific split boundaries and atomic policies.
These results do not validate replacing AFN implementations with an identically
named tile shape: staging and accumulation implementations differ. No GEMM
production default is promoted from this matrix-only screen.

Full evidence: `~/mojolearn-evidence/apple-fast/sync/resident-catalog-t-v1-report.json`.
Quality receipt hash: `45118d4df721f5ae5664f36232e7226cef67b397ed3f97849345d518b27da3e1`.
