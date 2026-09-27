# gbdt-stochastic-arms: CPU and NVIDIA columns (lane/gbdt-cpu-default-parity, 2026-09-20)

Two columns at commit 0db56f5a3, each cell fitted once (`--repeats 1`):

- `cpu-x86.json`: RunPod CPU pod, x86-64, 8 vCPU, host bindings built from source.
- `nvidia-nvidia-geforce-rtx-4090-sm_89.gbdt-cpu-default-parity-2026-09-20.json`: one RTX 4090.

All nine fixtures agree on all 29 train parts and on the infer, model and batch
cells: 36 of 36 cells IDENTICAL.

An earlier pair at 38c709386 disagreed on the seven RMSE Depthwise and
Lossguide parts and on nothing else. The CPU binding's dispatch sent those fits
to the symmetric RMSE oracle. That pair is not committed.

## The Plain held-out curve is a device defect, not a host one

`heldout_curve_values.py` prints values instead of hashes; the two
`heldout-curve-values.*.txt` files are its part A on each device class.
Measured on the RTX 4090 alone:

- the learn curve is the same bits in three fits of one process, and the
  held-out curve of the third fit, taken after an unrelated fit, is not
  (0.738080 then 0.738699 at tree 1);
- with the eval set equal to the learn set, the held-out curve is not the learn
  curve (0.7366 against 0.6196 at tree 1). On the CPU the two are the same bits;
- the first held-out value is above ln 2, which one tree at rate 0.03 from a
  zero cursor cannot produce.

`gbdt/methods/doc_parallel_boosting.mojo::fit_with_test` fills the test cursor
only under `boost_from_average`; `make_test_arm` creates it with
`enqueue_create_buffer` and nothing else writes it before the first tree is
added. `gbdt/methods/ordered_boosting.mojo` fills its test cursor
unconditionally, which is why `gbdt-ordered`'s held-out curve agrees across
device classes and `gbdt-symmetric-eval`'s does not. The CPU oracle seeds
zero. The device file is outside this lane and was not edited here.
