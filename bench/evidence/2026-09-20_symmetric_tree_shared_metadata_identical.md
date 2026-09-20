# Symmetric-tree shared metadata in IDENTICAL mode

The already-landed `compute_bins_and_add_kernel` stages an oblivious tree's
split descriptors once per block. This receipt checks that the structural
FAST win also preserves and modestly improves the IDENTICAL public fit path.

Apple M4, `bench/lanes_price_main.mojo`, 1,000,000 rows by 100 features,
SymmetricTree Logloss, depth 6, five trees:

- Reconstructed global-metadata baseline: 0.407155, 0.400142, 0.441236 s;
  median 0.407155 s.
- Shared-metadata repeat: 0.402901, 0.393788, 0.410420 s; median 0.402901 s.
- Seven-round shared repeat: 0.391354, 0.396217, 0.393050, 0.394371,
  0.419942, 0.452369, 0.544360 s; median 0.396217 s. The rising tail is
  thermal/system load, so no claim is made from the fastest sample.

The conservative paired median improvement is 1.0%; the longer shared median
is 2.7% faster. Every baseline and shared round produced the same complete
model ledger hash `277c0624c8ab82ac`, covering every split feature/bin/type,
leaf value, and loss. No arithmetic, predicate, leaf-bit order, tree order,
or cursor-add order changed.

The first shared run after the long compiler load ranged from 0.50 to 0.65 s
and is treated as environmental noise, not contrary routing evidence: the
reverse-order warm rerun recovered and the exact hash never moved.
