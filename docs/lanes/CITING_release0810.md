# Citing the 0.8.10 patch release

Release archive: **https://doi.org/10.5281/zenodo.22864816**. Both macOS arm64 and Linux x86-64 wheels are published on [PyPI](https://pypi.org/project/mojolearn/0.8.10/). Both exact wheels passed all 13 expanded installed checks (Metal and CUDA respectively), and both files downloaded from PyPI match the admitted SHA256 hashes. Receipts are in `bench/results/release_verification/2026-09-20_pypi_0810/`.

Package Python, harness and reference source is `fe27b1d053e2c44818dd08736721087880f03f2a`; native libraries are preserved byte for byte from public 0.8.9, source `819a47ae48166e91951f54f54e64ee173658e32a`. The archive tag is `fc0bf2a04e003f341e3a8127a841d2692520cdcc`, which adds the archive export correction without changing the wheel bytes. All 1,675 native build inputs match the native parent. The Linux wheel retains CUDA sm_89/sm_90a and HIP gfx942 libraries.

The downloaded Zenodo ZIP contains all **331 actively referenced raw records**, verified against their SHA256 hashes and the packaged reference-table hash. Its description is corrected and it has no `isDerivedFrom` relationship. The manifest and actual-download verification receipt are retained with the release receipts.

This release was published while additional one-device parallel reference collection continued. At its frozen table, Apple has zero vendor gaps, AMD has 378 missing vendor values, and NVIDIA has 126. The Apple trial's 837 matches comprise 423 CPU comparisons and 414 comparisons against existing GPU references; those GPU-only comparisons do not establish CPU support. This archive does not claim complete cross-vendor parallel coverage or physical two-GPU qualification. Later evidence on main does not retroactively change the wheels or this archive.
