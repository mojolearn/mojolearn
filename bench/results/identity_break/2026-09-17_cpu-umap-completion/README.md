# CPU UMAP row-separable reference and native controls

Both arms run all nine fixtures twice at source `4bb4a377a`, using the
fresh production/sabotage Mac metrics bindings built at `b9902bb46`.
UMAP native sources did not change between these commits. The fixture remains
1024 rows, eight columns and eight epochs. All training, held-out inference,
model and batch columns are stable. Native RNG-epoch sabotage changes all nine
training results; the clean batch invariance checks pass on all nine fixtures.
Binding digests and native sabotage readback are in the records. These are
single-thread CPU verifier results, not qualification of every GPU backend.
