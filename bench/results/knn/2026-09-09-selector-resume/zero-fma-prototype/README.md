# Exact zero-FMA repair

Initial production source: be363158; strict actual-arm gate fa3a1cf7. Apple IDENTICAL only, kNN register-distance
`_rt_step` only. NVIDIA remains on its correct software round-then-FTZ seam.
The matrix disable flag is MOJOLEARN_KNN_IDENTICAL_NO_ZERO_FMA_REPAIR.

The eight hardcoded boundary triples exposed NVIDIA hardware FTZ and Apple's
preexisting Metal FMA pre-round underflow behavior. The standalone exact
integer oracle passes 396,584 triples on Apple and NVIDIA, checking corrected
FMA and a simulated zero only for mathematically subnormal exact results.
NVIDIA also passes after production integration; Apple production fixture
and performance checks are run and archived separately by root.

For already-flushed finite inputs, decode each normal significand as a
24-bit integer. The exact product has at most48 bits. In units2**-150 its
magnitude is P * 2**(exponent_field_a + exponent_field_b -150). A result
rounds to the smallest normal exactly when its magnitude lies in the closed
interval [2**24 -1, 2**24 +1], with both endpoints included by ties-to-even.
Compare the exact scaled product to this interval shifted around the signed
accumulator. Dynamic integer shifts are overflow-checked; comparisons stay
exact even when the product exponent is too small or large for UInt64.

The slow path is called only after the hardware result flushed to zero.
Same-sign normal accumulator/product cannot underflow. Accumulator exponent
fields40 or larger need no repair: for cancellation the product's normal
exponent is at least ce-128, so the48-bit product and accumulator difference
is a multiple of at least2**(ce-175), which is2**-135 at ce40. A strictly
subnormal difference on that lattice cannot lie within2**-150 of minnormal.
For smaller exponents, the scaled24-bit accumulator fits UInt64 (at most63
bits) and both interval endpoints remain representable. Returning the
original zero preserves its sign whenever no minnormal repair is required.

The generator uses Python arbitrary-precision signed integer alignment and
independent RN-even rounding, not host Float64 FMA. It is deterministic:
`python3 tools/knn_zero_fma_oracle.py`. The generated text is not stored
in Git; its hash below permits checking regenerated input. This is bounded
adversarial coverage and an integer derivation, not a claim that every other
Apple arithmetic seam in the repository has been corrected.

Generated fixture SHA256: c7b29cb94ee6c8a222258ce99e472d45fa7375c94f84df15176e24f49d5fee10


Admission history: the strict production actual arm was subsequently run on
Apple and H100 by root and passed396584 triples on each. Root also passed
all Apple ordinary production gates. However, the initially inlined Apple
repair was rejected for cost: at400k/1000/k15, three-round request median
183.970 ms without repair versus2448.000 ms with repair; device168.016 versus
2417.113 ms. It must not be adopted as the default in that form.

Candidate28273628 isolates the UInt64 helper with @no_inline and adds a fast
proof guard: ae+be>=151 means the exact product is a multiple of2**-149, as
is every flushed Float32 accumulator. Their exact subnormal difference then
cannot round up to minnormal, so zero can be returned without the helper.
This is an arithmetic proof guard, not an input-distribution assumption.
The final cold guard passed strict396584 and hardcoded8 Apple cases and
reference fingerprints. Root accepted the correct default at request median
254.752 ms versus183.256 ms without repair (+39.0%), and device238.294 versus
168.122 (+41.7%),400k/1000/k15, three rounds. No further tuning this turn.
Final cold-helper ordinary fixtures also passed:4x143628 raw cells,
24 distance fixtures, identity4/main26/card6, UMAP186/690+20k fingerprint.
Root archives final Apple evidence at
bench/results/knn/2026-09-09-selector-final/apple-cold-final and
apple-cold-price. Other Apple FMA consumers remain outside this scoped repair.
