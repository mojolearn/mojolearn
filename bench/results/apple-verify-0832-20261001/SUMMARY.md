# Apple (Metal) check of the published mojolearn 0.8.32 on the M3 Ultra, 2026-10-01

`tools/apple_verify_job.sh` (run without `timeout`, which macOS lacks), its own venv, MOJOLEARN_NUMERIC_MODE=identical.

| check | Apple M3 Ultra | equals NVIDIA and AMD? |
|---|---|---|
| matmul, 21 shapes and transposes | all digests | yes (= L4 and MI325X) |
| classical (fixed harness): lu 1024, sgd-reg 20000, sgd-clf 20000, lars 200000, ivf 40000 | b53aeffc5bf56d98, 89cfa8f1827cd043, a6cb75bf0e2f93da, 640cb53e8400f17d, 5bf7822421de6cd5 | yes, all five |
| Kernel PCA 256 / 1000 / 10000 rows | b92023de7a6e41d5 / 45d3b6071431c7c3 / 1b442b0e601b9bfb | yes |
| mlp_step_check 256 / 32 rows | PASS, every byte equal; 5.90 -> 3.26 ms (1.81x) / 5.26 -> 2.65 ms (1.98x) | — |
| optimizer_resident_check adamw / adam / sgd | PASS, every byte equal; ~37 -> ~15 ms (2.38-2.52x) | — |
| **Mamba-3 forward+backward (strides_digest 2 512 384, 8 512 768)** | **FAILS: "Threadgroup memory size (35328) exceeds the maximum threadgroup memory allowed (32768)" at pipeline creation** | — |

The Mamba-3 failure is a bug in 0.8.32 on Metal (a backward kernel over Apple's 32 KB threadgroup limit); the fix
is in progress as its own PR and goes into 0.8.33.
