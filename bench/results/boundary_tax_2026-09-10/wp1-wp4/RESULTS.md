# WP1 / WP4 native checks

PASS on local Apple Metal in FAST, IDENTICAL and DETERMINISTIC: `ensemble/checks/oob_check.mojo` and `extratrees/checks/borrowed_upload_check.mojo`.

OOB arm F compares device identity rows against their integer indices for 1/257/1031 rows over three trees. Complete classifier forest/OOB fingerprints agree between borrowed-host and device-download sources in both row-major and column-major layouts, including signed zero and subnormal inputs. Existing OOB mask, analytic classifier/regressor scores, negative control and refusal checks also pass.

ET checks preserve raw uploaded Float32 bits and compare complete classifier/regressor model fingerprints between List and borrowed-X entry paths, with bootstrap enabled and disabled. Labels retain their existing validation and quantization.

Initial fixture compile failures and a WP8 SIMD API compile failure are retained separately. Corrected successful build/run logs are named by mode and check. Binary/source hashes and commands are in `provenance.json`; binaries are not tracked.

No binding build, cross-vendor qualification or speed claim is established by these native checks.
