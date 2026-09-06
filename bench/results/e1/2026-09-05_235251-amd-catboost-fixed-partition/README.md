# Weighted CTR and occupied zero-mass leaf: AMD PASS

AMD MI325X, clean tracked source `d0c9fe949b51eb8bfe1dfe977c7b02bae9e9b8df`.
The complete native tree CTR slice passes in IDENTICAL. The production final
leaf estimator was extracted without changing its arithmetic, so a fixed
occupied zero-weight partition can be tested independently of whether the
split scorer chooses one. Two nonzero-target rows have zero total weight;
the companion positive-mass partition checks an independently calculated
weighted mean. Both run at L2=0 and L2=3. Learned-tree predictions still pass
the separate weighted leaf-membership oracle; unequal weights and unequal
targets expose ignored weights.

Earlier failures required the optimizer to produce a zero-mass leaf even
when a shallower tree was valid. The replacement checks the intended numeric
corner directly; it does not drop the zero-mass requirement or loosen the
prediction threshold. Full logs, exit status, actual device and clean tracked
status are retained here.

This was one serial correctness job, with two compiler workers and affinity
to four CPU cores, on the already guarded AMD machine after its Mamba tests
finished. It is not an end-to-end ordered-boosting implementation, an external
CatBoost parity result, a speed measurement, or a fresh NVIDIA qualification.
