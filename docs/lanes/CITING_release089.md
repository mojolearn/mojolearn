# Citing the 0.8.9 release and CPU/GPU evidence

For results based on this fixture campaign, cite the fixed evidence archive: **https://doi.org/10.5281/zenodo.22864165** (version `evidence-0.8.9-cpu-gpu-20260920-r2`, Git commit `1c68d3dbc9587c4c29f7a3cbe9d7889a1fbfbb01`). The archive includes the raw columns, canonical reference table, all-vendor audit, and public-wheel receipts.

The report establishes 7,452 numeric cell parts matching exactly across CPU, Apple Metal, AMD HIP and NVIDIA CUDA for 214 nonparallel verification lanes and all nine fixtures. Eighteen explicit CPU model-writer role exceptions are separate. The 59 parallel lanes have separately documented coverage limits. These are fixture results, not a claim about all possible inputs, every GPU model, or every parallel configuration. See `LANE_STATUS_release089-all-vendor-coverage.md` and `release089-all-vendor-fixture-audit.json`.

The published 0.8.9 wheels use frozen source `819a47ae48166e91951f54f54e64ee173658e32a`. Their source archives are https://doi.org/10.5281/zenodo.22864054 (Linux release) and https://doi.org/10.5281/zenodo.22863999 (macOS release). Both archive the same source revision; citing both is usually unnecessary. The evidence archive is later and preserves each record's actual source/build provenance; it does not claim its repository HEAD built the wheels. The wheel files themselves remain on PyPI and GitHub Releases; their hashes and installed checks are in `bench/results/release_verification/2026-09-20_pypi_089/`.

For the paper, cite the specific archive used for the reported results. Future releases do not change that citation. Use the concept DOI, https://doi.org/10.5281/zenodo.22068632, for the evolving project as a whole. Prefer one combined software/evidence archive per meaningful future release rather than separate DOI versions for every platform or test run. The paper can have its own publication DOI and cite these software/evidence records.

Zenodo guidance: https://zenodo.org/help/versioning . A DOI identifies an archived object; it does not certify the scientific claims.
