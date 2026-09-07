This candidate was rejected before PyPI upload because its Summary metadata was email-folded. Retained for provenance; use the repaired candidate/release instead. Native and Python payload qualification is unchanged.

MojoLearn 0.6.0 alpha API release.

Exposes public `mojolearn.linalg`, `mojolearn.umap`, and
`mojolearn.training`, including existing optimizers and loss functions,
alongside Mamba/Transformer blocks and the fixed small-trainer Python APIs.

Packaged binaries target Apple silicon macOS and AMD gfx942 Linux. NVIDIA
Linux uses a source build for this release. This API overlay preserves the
identified base wheels' native/runtime bytes; availability still depends on
the compiled symbols in each vendor/mode set. In particular, the newly
authored byte-LM extension is not bundled by the overlay.

The distribution remains alpha software. API availability does not establish
complete CatBoost feature parity, arbitrary-shape correctness, cross-vendor
training/resume identity, or numerical qualification of the updated wrappers.
The installed alpha guide and per-wheel provenance record those boundaries.

These draft assets are the exact publication candidates. PyPI publication is
confirmed separately by matching the uploaded wheel hashes with the retained
manifest; a draft GitHub release is not itself a PyPI release.
