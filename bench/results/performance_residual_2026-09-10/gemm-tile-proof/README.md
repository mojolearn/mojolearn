# Rejected whole-tile exponent proof

This second range-proof experiment moves checking out of the FMA fragment
loop: each128x128 block scans its complete A/B operands, uses an integer
block minimum and broadcasts the result. Min biased exponent>=76 implies
products on the2^-148 lattice and finite flushed accumulators on2^-149,
excluding the problematic minnormal rounding interval. Exceptional
operands force the original RN-plus-flush path. Split plans bypass it.
An independent read-only review found no coverage/geometry/proof error.
NVIDIA seven-gate device suite passed, including launch invariance.

The additional scan/code-generation cost lost again: QKV4.589ms,
MLPup18.103ms and MLPdown15.943ms, approximately twice the unchanged
baseline. These are seven-call means from the existing probe. Its two
printed arms both use the same binary's dispatcher; compare the separate
baseline-price/candidate-price logs for timing. The device gates supply
the independent numerical check. No cross-binary SHA claim or opponent
measurement is made. This candidate never entered production source.

Base5011239a with rejected-tile-proof.patch; the build script supplies
MOJOLEARN_GEMM_TILE_PROOF and imports the isolated GEMM package before the
baseline repository. H100580.126.09. No clocks were changed.
