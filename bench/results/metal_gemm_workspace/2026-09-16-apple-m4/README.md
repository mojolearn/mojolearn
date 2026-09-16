# Apple M4 retained GEMM workspace evidence

See `docs/METAL_GEMM_WORKSPACE.md` for the change, protocol and limitations.

- `quiet-timings.json`: final six-run comparison; no task-owned build running.
- `timings.json`: earlier six-run comparison with concurrent CPU compilation.
- `quiet-*.log`: each final oracle gate; all passed.
- `workspace-run.log`: growth, queued reuse, bitwise oracle and no-wait budget.
- `sync-control.log`: expected failure when reuse is replaced by the old synchronous call.
- `forward-bcd.log`, `backward-expanded.log`: successful optional checks.
- `baseline-forward-e.log`, `forward-expanded.log`: same pre-existing refusal-harness failure on both arms; not counted as passing.

No CUDA/HIP validation was run for this change. Binaries are outside Git in
`~/mojolearn-evidence/metal-gemm-workspace/`; their hashes are in the timing JSON.
