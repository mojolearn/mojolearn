# Permutation SHAP delta default preparation

Measured source 20bbdc37256bd00ef09fc97970f79f6967fa4f2a.
M3 w2-pdelta-pshap-istella: A28222.8 -> B14788.6 ms (-47.6%).
Quality w2-pdelta-quality PASS: three models' attribution arrays byte-identical;
linear relative error1.031583e-8 and additivity6.196e-7 identical; tanh
additivity1.162e-7, two-output5.865e-8 identical. Model rows2713500 ->2215900.
One scored run per arm; no replay or new opponent measurement.

Source review: base629940fe9's four product files match candidate baseline
4b85be62c exactly. This branch imports candidate code without arithmetic
changes, activates only FAST Apple, and keeps MOJOLEARN_PSHAP_DELTA_OFF.
IDENTICAL and other vendors retain old paths. New device kernels distribute
counts, index construction, synthetic cells, mapping and means over grid-stride
threads/blocks. Prefix sum uses parallel scan launches; no global single-thread
or single-block reduction. Per-group background loops preserve arithmetic.

Host changes are metadata and existing model callback glue: compact row count,
shape/allocation, zero-copy prefix view and callback invocation. All synthetic
construction, row equality tests, compaction, reconstruction and attribution
math stay on GPU. Callback must remain deterministic per row, as already
assumed by existing chunked model evaluation. The benchmark's NumPy ridge
callback is the existing boundary; this change adds no NumPy product math.

Manager validation: compile x_trees A=-D MOJOLEARN_PSHAP_DELTA_OFF and B=empty,
then run `python3 tools/pshap_delta_pair.py quality SOURCE_SHA
pshap-delta-default-quality-20261004` on M3. Pair helper now accepts only quality
and checks rollback/default manifest, switches and hashes. Fixed original
quality thresholds preserved. No default merge until compile/quality approval.
