# 0.8.8 verifier/reference patch

Status: preparing; 0.8.7 remains the published release until the exact new wheels
pass light smoke and publication completes.

The patch separates CPU replay of GPU-written CTR models from GPU model-byte
references. A recorded CPU N/A cannot erase a GPU numerical reference; different
GPU hashes remain a conflict, and an actual result mismatch remains DIVERGENT.

The targeted reference update is limited to 12 GPU parallel lanes with 148 missing numerical reference entries, nine existing
fixtures and two repeats on Apple, NVIDIA and AMD. Raw records and the strict
three-column comparison accompany admission. Existing references outside those
lanes are preserved. These results do not add CPU routes, prove physical multi-GPU
execution, or establish the paper's broader every-variant four-device claim.

Native libraries are inherited unchanged from published 0.8.7. The new overlay
path checks unchanged native compile inputs and records package/native sources
separately. Each final platform wheel requires one bounded installed smoke;
unrelated algorithms and unchanged native builds are not repeated.
