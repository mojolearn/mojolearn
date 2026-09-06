# Guarded remote resume campaign

Authored and source-reviewed only; this script has not been executed or qualified. Root alone runs these commands on already leased Linux NVIDIA/AMD hosts, sequentially after other jobs finish. It neither provisions hosts nor transfers checkpoints. No Apple execution is supported.

From the same frozen repository checkout on each host, choose the actual GPU architecture and fresh absolute output directories:

```sh
bash tools/training_resume_serial.sh --vendor cuda --gpu-arch sm_89 --out /artifacts/resume-cuda-new
bash tools/training_resume_serial.sh --vendor hip --gpu-arch gfx942 --out /artifacts/resume-hip-new
```

Each invocation builds the existing B2/L8/DM32/V64, 13,376-parameter driver in IDENTICAL mode with `-j 2`, then runs continuous16 followed by head8. The script already invokes the appropriate vendor guard for every build and model job; wrapping the entire script in the same guard would conflict with its exclusive job lock. A two-core affinity and 1,785-second outer timeout plus at most 15 seconds of termination cleanup bound the complete campaign to 1,800 seconds. Each child guard also enforces its own deadline, 12 GiB process-group RSS limit and GPU admission checks. Failed or timed-out output directories retain partial evidence and must not be reused.

Root must transfer each actual `head8.ckptbin` to the other host. Then run these separately on their respective destination hosts, reusing that destination's exact retained driver:

```sh
bash tools/training_resume_serial.sh --vendor cuda --gpu-arch sm_89 \
  --out /artifacts/resume-cuda-from-hip-new \
  --incoming /transferred/hip-head8.ckptbin --build-from /artifacts/resume-cuda-new
bash tools/training_resume_serial.sh --vendor hip --gpu-arch gfx942 \
  --out /artifacts/resume-hip-from-cuda-new \
  --incoming /transferred/cuda-head8.ckptbin --build-from /artifacts/resume-hip-new
```

Incoming mode requires both options, checks the retained binary/build/source hashes and vendor/architecture, captures exactly 161,008 incoming bytes, and runs only resume16. The native helper seals those captured bytes before loading and hashing them. Sources are checked before each device leg and again during evidence collection. The source snapshot records file hashes, not a copy of the repository; preserve the corresponding checkout separately.

Each output retains `orchestrator.sh`, `vendor-guard.py` and hashes, `runtime.txt`, `source.json`, build command/log/exit status, `build.json`, and `resume-driver`. Each action retains its exact command, guard log, shell exit status, hash-bound zero-exit receipt, checkpoint/native receipt, and an `<action>-evidence` directory produced by the existing file-only evidence collector. `COMPLETE.txt` means those local jobs completed; it does not admit cross-vendor equality.

Collect all evidence directories on one host and use the comparator in [PUBLIC_TRAINING_RESUME_COMMANDS.md](PUBLIC_TRAINING_RESUME_COMMANDS.md). Map `continuous16-evidence`, `head8-evidence`, and `resume16-evidence` to the six appropriate vendor/direction arguments. Full admission additionally requires the effective retained missing-moments control described there. Without that control, otherwise matching six-leg evidence exits 3 with `full_resume_claim_admitted=false`. This orchestrator does not run the control, external gradient oracle, or any benchmark. A successful comparator covers only this fixed profile and retained source inventory.
