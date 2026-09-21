# mojolearn 0.8.12

Source commit 304df4fcadfb086fe2b177e10d8df0723d472f99 (tag v0.8.12).

## Release verification (CPU and Apple GPU, on the release Mac)

`pixi run -e test release-check` at the source commit, every cell fitted once,
fixtures base, denormal and odd. Each column judged part by part against
python/mojolearn/verify_reference/table.json:

| column | lanes | cells | IDENTICAL | N/A | DIVERGENT | skipped probes |
|---|---|---|---|---|---|---|
| CPU | 214 | 642 | 1878 | 2508 | 0 | 642 batch, 60 rlpair |
| Apple M4 Metal | 208 | 624 | 1839 | 2421 | 0 | 624 batch, 54 rlpair |

Columns kept outside the repository (about 800 KB each):
- cpu/column.json sha256 37eded1760b91a70516547161c22f1975bf6d7619ab143cf2bb21797211bf4fe
- metal/column.json sha256 52319e380b284876c034092141ba061153777671feb18e361bf04599c8e7b3c7

## Linux wheel

mojolearn-0.8.12-py3-none-manylinux_2_35_x86_64.whl, sha256
f56194f44bb8f077bb801a9bef3822bf0aa3ed03a228eaee00949b69f967b109, packed from the
sm_89, sm_90a and gfx942 release builds of the source commit. The 32 host bindings
are byte-identical across the three builds. auditwheel repair passed. The exact
wheel passed tools/qualify_verifier_wheel.py --scope expanded on an NVIDIA H100
(light-smoke-linux.json) and was published on the light route with tag
alpha-api-0.8.12-linux-light-20260921.

## macOS wheel

mojolearn-0.8.12-py3-none-macosx_11_0_arm64.whl, sha256
bd23113c55bc81254a301afb2503b697de799caca2d611e86995243227d98200, built on the release
Mac from the source commit (packaging/macos/build_release_wheel.sh with
MOJOLEARN_PACKAGE_BYTE_LM=1; its installed-wheel verification passed on 5
interpreters). The exact wheel passed tools/qualify_verifier_wheel.py --scope expanded
on the M4 GPU (light-smoke-macos.json) and was published on the light route with tag
alpha-api-0.8.12-macos-light-20260921, the first fresh native macOS build admitted on
that route (packaging/verify_alpha_artifacts.py, 93fd389a7; the tag's commit is the
source commit plus that admission change, branch release/0.8.12).

## Installed from PyPI

`pip install mojolearn==0.8.12` on the M4 (Python 3.12): version 0.8.12, vendor metal,
mojolearn.cross_vendor present, ParallelByteLanguageModelTrainer.shard_gradient and
apply_gradient present, the preprint ships, and the wheel requires no other package.
