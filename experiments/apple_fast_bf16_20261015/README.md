# Apple FAST BF16 public-operation screening

The existing explicit BF16 operations now default on in Apple FAST source,
following Andrew's 2026-10-15 instruction. `current_decision.json` records the
scope, caller audit and completed Apple build qualification. Explicit
`MOJOLEARN_BUILD_DEFINES=MOJOLEARN_AFN_GEMM_BF16_MMA=0` disables the route,
including when aggregate experimental switches are on. Historical measurement
summaries retain their original default-off decision unchanged.

Its public operation uses the existing Apple simdgroup kernel with BF16 input
bits widened exactly and FP32 accumulation/output. It does not silently
quantize FP32 callers or change IDENTICAL dispatch.

Root compiled candidate source `6cec6320d` on Apple M4 and obtained 22 passing
correctness tests with no skips, plus 35 passing cold one-hot probes. The local
comparison used the shipped 0.8.45 IDENTICAL API and PyTorch 2.13.0 MPS as separate
arms. Five scored samples per shape followed a separately recorded first call
and one excluded warmup. Every arm's outputs passed the independent FP64 oracle
bound; quality and failures remain in the retained records.

| M, N, K | Shipped IDENTICAL ms | FAST candidate ms | PyTorch MPS ms | Candidate / IDENTICAL |
| --- | ---: | ---: | ---: | ---: |
| 256, 384, 320 | 0.968500 | 1.037833 | 1.922959 | 1.072 |
| 768, 1024, 640 | 6.625750 | 5.703375 | 14.119250 | 0.861 |
| 769, 1025, 641 | 7.287042 | 6.659166 | 6.010375 | 0.914 |

These are local operation screens, not full-dataset estimator measurements or
a same-source default-promotion A/B. The small case lost 7.2%; the medium cases
took 8.6–13.9% less time. No statistical significance or empty system caches are
claimed. Generic `mojolearn.linalg` routing and an installed-capability guard
were subsequently compiled and qualified on Apple at frozen `30b9c0f3d` with
23 passing GPU correctness tests and no skips. Other builds, installed-release
qualification and full-estimator evidence remain pending.
No existing estimator has been wired to use this candidate implicitly.

The same frozen generic boundary then ran the full 4096 × 4096 × 4096 GEMM
primitive with seed-7 standard-normal inputs. Each arm recorded one first call,
one excluded warmup and five scored samples. Its retained controller is the
portable runner at that freeze; the earlier three cases above remain unchanged.

| Full primitive arm | First call ms | Scored median ms | Scored min–max ms | Maximum absolute error | RMS error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipped 0.8.45 IDENTICAL | 257.059500 | 158.584500 | 154.399500–159.014417 | 0.000041820575 | 0.000006082485 |
| FAST candidate `30b9c0f3d` | 170.914792 | 133.247791 | 128.562917–152.106584 | 0.000786586199 | 0.000039119954 |
| PyTorch 2.13.0 MPS | 831.108000 | 356.609042 | 352.400042–361.400333 | 0.000786586199 | 0.000039119954 |

All 21 outputs were finite and passed the recorded FP64 error bound. The
candidate used 16.0% less time than shipped IDENTICAL and 62.6% less than this
matched Torch operation. Its numerical error was larger than IDENTICAL's;
every sample's quality is retained. Torch receives the same BF16 host bits,
widens them exactly on the GPU, and uses real FP32 `torch.mm` accumulation and
output, with the transfers and widening inside its clock. This does not compare
against a BF16-output Torch multiply. CPU fallback is disabled.

This completes a full-size primitive screen, not an estimator measurement or
full-board run. It compares a shipped IDENTICAL arm with candidate FAST, not
the same-source baseline/candidate configuration of a general promotion
campaign. The historical decision kept the switch off; the subsequent owner
instruction enables only existing explicit BF16 operations. See
`results/20261015-full-4096/comparison.json` for all samples, source/binary/input
hashes, process exit status, qualification and durable artifact references.

Root compared the actual first scored 4096 output arrays: FAST differs bitwise
from shipped IDENTICAL in 15,768,413 of 16,777,216 FP32 elements, with maximum
absolute difference 0.0007781982421875. FAST equals the matched Torch output
bitwise in this one saved local case. This does not establish or promise
cross-vendor FAST identity. The original comparison receipt is retained under
`decisions/20261015-enable-existing-bf16/saved-output-comparison.json`.

## Select an arm

`local_benchmark.py` is a portable adaptation of the original controller. It is
testing code, never part of product execution, and never builds or installs.
Use already prepared, separately selected environments/bindings for each arm.
Run arms serially on an authorized idle Apple GPU; keep full stdout/stderr and
the shell exit status. Each arm refuses to overwrite its existing directory.

```sh
python experiments/apple_fast_bf16_20261015/local_benchmark.py prepare --output /path/to/new-attempt
MOJOLEARN_NUMERIC_MODE=identical python experiments/apple_fast_bf16_20261015/local_benchmark.py released-identical --output /path/to/new-attempt --wheel-manifest /path/to/wheel-manifest.json
MOJOLEARN_NUMERIC_MODE=fast python experiments/apple_fast_bf16_20261015/local_benchmark.py candidate-fast --output /path/to/new-attempt --source-commit EXACT_COMPILED_COMMIT
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 python experiments/apple_fast_bf16_20261015/local_benchmark.py torch-mps --output /path/to/new-attempt
```

Preparation accepts repeated `--shape M,N,K` arguments; all arms then consume
that same hashed input manifest. Candidate execution requires an enabled native
binding and the generic native aliases, and refuses a substitute. For the full
primitive, select `prepare --preset full --output /path/to/new-full-attempt`,
then run the same three arm commands against that new output directory. It
selects the board's 4096-cubed GEMM shape and seed-7 standard-normal inputs; it
does not run a full estimator or full board. The original screening inputs and
results remain untouched. Build owners use
`MOJOLEARN_BUILD_DEFINES=MOJOLEARN_AFN_GEMM_BF16_MMA=1` to explicitly enable
the FAST linalg route; omission now uses that same default. Explicit `=0`
disables the BF16 capability. Setting `MOJOLEARN_BUILD_DEFINE_FLAGS` directly
is insufficient.
Builds remain the orchestrator's responsibility.

The clock starts with the same prepacked BF16 host operands and ends with a
consumed FP32 host result. Packing, allocations, transfers, required widening,
matmul, synchronization and output materialization are included. BF16 test-data
construction, imports, the independent FP64 oracle and evidence writing are
outside the clock. PyTorch uses real `torch.mm` after exact GPU widening inside
its clock; CPU fallback is disabled. The first recorded call is per shape; only
the first shape's first call is also the first operation in that process.

## Evidence and remaining work

`result_summary.json` gives the decision, coverage, exact source/binary hashes,
clock, sample counts, raw evidence paths and pending gates. `results/20261015`
retains the original controller and summaries byte-for-byte, including original
absolute evidence paths. The historical controller SHA256 is
`c861085ff1790ca5b6d26ce59a54320101c54fa34573cf9435b5aa3a78c334d2`;
the enabled native binding SHA256 is
`b2c32bea3eea8321f9909d10e8f497b4f15529dd8d51cb1340751e836cf41565`.
Historical timings were produced by those retained controller bytes, not the
portable adaptation. Binary, input/output array bytes and full logs remain in
the recorded evidence directory; hashes alone are not replacement retention.

The portable runner has CPU preparation/argument checks and the completed
full-primitive GPU comparison at frozen `30b9c0f3d`. The earlier three-case
compile/correctness and timing evidence still refers to frozen `6cec6320d`;
the later compile06 and 23-test qualification refer to `30b9c0f3d` separately.
The full primitive recipe is `tools/bench_board_neural.py`'s `GEMM_SHAPES.full`,
4096 by 4096 by 4096 from seed 7, with input hashes retained when materialized.
It is not an estimator dataset. Existing transformer/Mamba
packed BF16 weights are unpacked to FP32, and training is FP32-only: broader
wiring needs native packed-weight retention or an explicit precision opt-in,
actual caller mapping, and full-workload recipes.

Root archived the original experiment, including all source, binary, input,
output and failed-attempt logs, to bucket `mojolearn-data`, key
`measurements/records/v1/a2397742b2a1cba2865940141d63cc8f2b0483b096cfb72d9b7ad4c16a8e4558.tar.gz`.
The 181,871,269 bytes were independently GET/SHA256 verified; see the durable
reference in `result_summary.json`. This archive predates generic wiring.

The full-primitive archive is separately GET/SHA256 verified in the same bucket:
`measurements/records/v1/7f44cb98392244faad9795b85783ce60ed421df94da819a27f1db5082a8664b2.tar.gz`
(1,595,961,388 bytes). It retains the new frozen source and candidate binary,
inputs, all 21 output arrays, executed controllers and full logs. The reused
shipped IDENTICAL binary and wheel manifest are retained in a supplementary
GET/SHA256-verified archive referenced by the full-primitive comparison record.
Both summaries and archive inventories are portable repository artifacts;
original absolute paths remain in the byte-for-byte historical receipts.

The enabled source is scoped to explicit BF16 public operations. Float32-A /
BF16-B and transpose variants have correctness evidence but no timing claim.
No estimator has been wired to BF16, and any broader model change still needs
its full-workload evidence. Root compiled frozen `235774987` in default-on, explicit-off and IDENTICAL
configurations. All three builds and capability checks passed, with 23 GPU
correctness tests and 43 CPU policy tests passing without skips. The full4096
output replay reproduced the earlier FAST and IDENTICAL outputs respectively;
no new timing measurement was performed. Receipts are under
`decisions/20261015-enable-existing-bf16/validation/`. Release qualification
and publication remain pending.
