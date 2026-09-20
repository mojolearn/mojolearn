# Direct public preprocessing output qualification

IDENTICAL mode, Apple Metal, public native ABI. Shape: 1,000,000 rows by 8
Float32 columns. The candidate leaves transform arithmetic and launch geometry
unchanged, scans the device result for the same nonfinite-output refusal, then
copies directly into the Python-owned output pointer. It removes the previous
device-to-host-buffer-to-List-to-Python materialization.

StandardScaler fit plus transform, fresh-process baseline from merged
`5a4f6fa3a` (exact-capacity device downloads), warmed milliseconds: 95.743,
99.383, 107.900, 103.278, 274.551, 137.159; median 105.6 ms. Candidate after
the nonfinite scan, warmed milliseconds: 61.229, 73.726, 67.856, 62.725,
106.669, 98.929; median 70.8 ms, 33.0% faster. Every baseline and candidate
round hashed `423db12a1e1923a0` over all 8,000,000 output cells.

Baseline process maximum RSS rose from 248,315,904 bytes after warmup to
440,434,688 bytes after seven calls. Candidate rose from 247,857,152 to
248,692,736 bytes. Thus the eliminated host materializations also remove
roughly 32 MB of allocator high-water growth per repeated call.

MinMaxScaler candidate fit plus transform warmed milliseconds: 62.733,
67.950, 66.017, 70.807, 79.176, 77.557. Every output hash was
`bcf689e2272ff89e`, matching the merged implementation's qualified hash.

An inverse StandardScaler witness using Float32 maxima was refused after the
device nonfinite scan, preserving the previous arithmetic-overflow contract.
`bindings/build_preprocessing.sh` passed the StandardScaler and MinMaxScaler
fit/transform/inverse native ABI gates.
