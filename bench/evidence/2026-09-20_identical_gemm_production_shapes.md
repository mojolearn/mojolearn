# IDENTICAL GEMM: production transformer shapes (2026-09-20)

Host: Apple M3 Max, local Metal device, bounded to two compile jobs. Base:
`36b77f669`. Inputs are generated deterministically by
`bench/gemm_production_shapes_main.mojo`; the card times five synchronized
launches and compares every output bit with explicit plan 10 (128x128).

| Consumer shape (M x N x K) | 128x128 raw ms | 64x64 raw ms | Median change | Full-cell comparison |
| --- | --- | --- | --- | --- |
| QKV 32768x2304x768 | 846.274, 1033.320, 1046.437, 940.697, 938.038 | 624.706, 738.127, 732.450, 692.877, 671.341 | 940.697 -> 692.877 (-26.34%) | 0 / 75,497,472 mismatches |
| MLP 32768x3072x768 | 1213.748, 1223.781, 1241.061, 1230.247, 1227.806 | 872.011, 873.126, 938.905, 972.294, 886.034 | 1227.806 -> 886.034 (-27.84%) | 0 / 100,663,296 mismatches |
| LM-head chunk 32768x1024x768 | 409.510, 417.687, 420.403, 411.884, 411.670 | 284.374, 287.074, 289.753, 300.051, 291.683 | 411.884 -> 289.753 (-29.65%) | 0 / 33,554,432 mismatches |

The dispatcher is Apple + IDENTICAL only and only selects existing plan 9 for
`M >= 32768`, `N >= 1024`, `K == 768`. FAST, DETERMINISTIC, other device
columns, and neighboring shapes retain the old plan. Workspace remains one
float. A candidate-dispatch rerun also had zero full-cell mismatches; its raw
times were 811.300/827.747/702.039/737.854/708.497,
962.551/920.143/942.313/983.144/932.793, and
278.790/289.450/299.059/311.469/333.202 ms respectively.

Reproduce:

```sh
MOJOLEARN_PROD_GEMM_PLAN=10 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_production_shapes_main.mojo
MOJOLEARN_PROD_GEMM_PLAN=9 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_production_shapes_main.mojo
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_production_shapes_main.mojo
```

The full chunked-LM-head training path was screened but is not claimed as an
end-to-end win: at this row count its separate exact dWeight kernel dominates
and makes a local run impractically long. NVIDIA qualification remains open;
the two live RunPod instances were externally owned and were not touched.
