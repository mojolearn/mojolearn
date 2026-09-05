# macOS 0.6.0 candidate installed UMAP qualification

Built from b85aa1c91687862999dee17675e5bcda690514c6 by release workflow
33972158535 with publish=none. The wheel contains all 45 extensions and
passed the standard Python 3.10–3.14 × three-mode installed smoke matrix.
The additional fresh Python 3.12 environment passed UMAP fit, transform
and both held-out quality cases in FAST, DETERMINISTIC and IDENTICAL.
Every job checked installed site-packages location, wrapper hash, compiled
mode and binary hash. See installed/results.json and the per-job records.

SHA256SUMS identifies the exact locally retained wheel (binary excluded
from git). This is candidate qualification, not PyPI publication or Linux
qualification. The experimental k-NN selector is disabled in this wheel.
