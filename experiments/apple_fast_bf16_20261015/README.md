# Apple FAST BF16 public-operation screening

The native `mojolearn.linalg_fast` BF16 candidate remains disabled by default.
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
claimed. Generic `mojolearn.linalg` routing, board capability integration, other
builds, installed-release qualification and full-workload evidence remain pending.
No existing estimator has been wired to use this candidate implicitly.

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
binding and refuses a substitute. Build owners use
`MOJOLEARN_BUILD_DEFINES=MOJOLEARN_AFN_GEMM_BF16_MMA=1` when building the FAST
linalg binding; setting `MOJOLEARN_BUILD_DEFINE_FLAGS` directly is insufficient.
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

The portable runner has CPU-only preparation/argument checks; its GPU path has
not been rerun. Reuse the accepted native compile/correctness evidence: this
delivery changes only experiment tooling, exclusions and comments after the
measured native commit. Keep the candidate off until the applicable whole-
workload evidence supports promotion. A Mac-only new-version release is
technically possible, but no release or public documentation is authorized here.
