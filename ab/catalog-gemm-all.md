# Catalog G1–G10 standalone quality probe

Source-only extension of compiled G1/G5 probe 6abb76673; catalog9ab2d3d3f.
No production dispatch or numerical kernel changes. Binding gemm_probe,
define MOJOLEARN_APPLE_GEMM_PROBE, ABI2, explicit completed-call counters0–10.
Direct tiles: G1 64x64x16, G2 32x32x16, G3 64x128x16, G4 128x64x16.
Shared: G5 64x64x16, G6 64x64x32, G7 64x64x16 pad4,
G8 64x128x32, G9 128x64x32, G10 64x64x32 pad4.

M2 A/B compilation required, then serial M3 tools/catalog_gemm_quality.py
FULLSHA w2-catalog-all-q-20261004. Twelve unscored oracle fixtures per arm;
unchanged strict no-worse and absolute gates. Each variant gets its own
hash/source-bound PASS_Gn.json only if all its cases pass. Failure in one
variant does not validate or reject another. G1/G5 retain exact-pair gate.
No timing harness, default change, caller certification or promotion here.
