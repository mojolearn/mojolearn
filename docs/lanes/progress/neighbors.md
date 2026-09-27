# neighbors: progress

Lane table (main), then Additions, then Graph and GP additions. One commit per
algorithm. Pass-1 gate: builds on the pod, `python/mojolearn/tests/test_x_neighbors_sanity.py`
passes against scikit-learn, `tools/algos_lane_check.sh <lane>` reads AGREE.

HOW THE LANE IS BUILT: every primitive is an item function in
`x_neighbors/items.mojo`; `x_neighbors/gen.py` generates the GPU driver
(`device_ops.mojo`), the host driver (`host_ops.mojo`) and both bindings from
its OPS table (run it after editing the table). Python (`_expansion_neighbors.py`)
only moves buffers and does exact integer bookkeeping.

Pod setup that is not in the repo: the default pixi env needs
`python -m pip install scikit-learn pytest networkx` for the sanity tests.

| algorithm | commit | lane | pod verdict |
|---|---|---|---|
| LocalOutlierFactor | (this commit) | x-neighbors-lof | AGREE: compared batch 9, infer 9, train 9 (cuda H100 vs CPU Xeon 8470); sanity vs sklearn 1.9.1 PASS |
