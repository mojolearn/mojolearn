# Launch probe, MI325X (DO), 2026-10-01 (tree ab-p596061-amd: pre-#65, pre-#67)

byte_lm_launch_probe: 2.38 us per launch on a fresh context, 2.34 us with the B1 L2048 8-layer session open (4-layer: 2.34 / 2.34). No arena on AMD.
Stage tool, MOJOLEARN_..._LAYER_SYNC 0 vs 1: 8 layers lm-forward 39.8 / 39.7 ms, train 74.9 / 75.1; 4 layers ("l4" scratch edit) 30.2 / 29.6, train 40.2 / 40.2.
Per layer: forward ~2.4 ms, train step ~8.7 ms; ~20 ms of the forward is layer-independent (the fresh 64 MiB logits output, fixed by #67).
R2: measurements/2026-10-01/launch-probe-amd.tar.gz.
