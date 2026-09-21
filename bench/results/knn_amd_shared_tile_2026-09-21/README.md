# AMD kNN shared-tile conditional trial receipts

These are compact, tracked copies of the two conditional (`n_features >= 32`)
provider receipts described in
`bench/evidence/2026-09-21_knn_amd_shared_tile.md`. The source artifacts remain
under `/Users/andrewhendel/mojolearn-evidence/2026-09-21_knn_amd_smem/`.

For each provider, `verdict.json` is a byte-for-byte copy. `summary.tsv` is
a whitespace-normalized compact copy: trailing padding is removed and an
empty baseline `paired_ratio_over_first` field is written as `NA`.
`cells_compact.json` is a deterministic projection of the two aggregate cell
JSON files. It retains every outer-pass median, every timed-call digest, all
six paired ratios, aggregate timing/spread/status fields, and the two-dataset
geomean. Its IQR is Q3-Q1 over the six outer-pass medians using Python
`statistics.quantiles(..., method="inclusive")`.

`leg.txt`, `source_sha256.txt`, and `teardown.txt` attribute the source and
provider lifecycle. `gpu.txt`, `box.txt`, `rocm_provenance.txt`, `uname.txt`,
`mojo_version.txt`, `pixi_env.log`, and `versions.txt` record the available
hardware, driver/runtime, compiler, environment, and dependency provenance.
The manifest hashes every receipt except itself.

This directory supports the fixed-workload performance and digest result plus
the final broad identity and sabotage gate. The two timing-trial sources left
the experimental exact-chain admission active below 32 features; the evidence
note records that historical gap and how the final gate closed it.

`hotaisle-mi300x-identity/` contains compact receipts for the corrected final
candidate. Its 22-lane broad matrix was bitwise equal to the CPU oracle, its
narrow outputs were unaffected by the shared-tile sabotage, and all 14 wide
brute-force cases required to traverse the promoted tile moved under sabotage.
The original judge incorrectly required the separate ball-cover radius path
to move; both its failed output and the corrected verdict are retained.
