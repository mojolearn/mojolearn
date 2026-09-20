# GPT-3-small AMD IDENTICAL GEMM dispatch qualification

## Contract and hardware

- Mode: explicit `MOJOLEARN_NUMERIC_IDENTICAL`; FP32 only.
- Device: DigitalOcean AMD Instinct MI325X VF (`gfx942`), 255.6 GiB VRAM.
- Baseline commit: `83d30bb05`; candidate commit: `aed49a659`.
- Shapes: QKV rows 1,024 through 16,384; MLP up/down at widths 2,048 and
  3,072; rows=32,768 combined QKV, MLP-up, and LM-head chunk.
- Five alternating baseline/candidate samples per shape after warm-up. The
  candidate qualification was repeated twice in separate processes.
- Every tested arm and shape reported `mismatches_vs_plan10 0` over the full
  output, with stable hashes matching NVIDIA and Apple receipts. Explicit
  workspace remained one float; the change does not add retained storage.

RunPod MI300X secure capacity was unavailable (`There are no instances
currently available`), and no RunPod resource was created. Both MI325X runs
used the repository's guarded DigitalOcean workflow with independent local
and on-droplet 60-minute dead-men.

## Baseline profile

The shipped packed body must be retained for narrow and down-projection
shapes: versus exact plan 10 it measured 0.404x at `(1024,768,768)`, 0.553x
at `(2048,768,768)`, 0.432x at `(2048,768,2048)`, and 0.370x at
`(2048,768,3072)`.

For `k=768` with either more rows or a wider output, however, the same packed
body was consistently slower than plan 10:

| shape `(m,n,k)` | old shipped / plan 10 | candidate / plan 10, run 1 | run 2 |
|---|---:|---:|---:|
| `(4096,768,768)` | 1.286 | 1.006 | 1.013 |
| `(8192,768,768)` | 1.258 | 1.001 | 0.994 |
| `(16384,768,768)` | 1.221 | 0.999 | 0.995 |
| `(2048,2048,768)` | 1.202 | 0.995 | 0.996 |
| `(2048,3072,768)` | 1.258 | 1.005 | 0.998 |
| `(32768,2304,768)` | 1.225 | 1.003 | 1.000 |
| `(32768,3072,768)` | 1.193 | 1.000 | 1.001 |
| `(32768,1024,768)` | 1.243 | 0.999 | 0.998 |

Thus the AMD-only guard makes the affected production projections roughly
16--22% faster (`new / old` about 0.78--0.84). Two direct alternating
candidate-versus-plan-10 runs remained 0.990--1.009 on all guarded shapes.
The four deliberately unguarded shapes retained their packed-body advantage:
candidate/plan-10 was 0.402--0.555 for narrow QKV and 0.373--0.434 for MLP
down. This is both the performance boundary and a routing-isolation check.

## Dispatch

Only AMD IDENTICAL builds are changed. In the shipped packed-body dispatcher,
`k == 768 && (m >= 4096 || (m >= 2048 && n >= 1024))` routes to exact plan
10. NVIDIA and Apple routing is compile-time unchanged; FAST and DETERMINISTIC
are untouched. The plan changes scheduling only: the contract partition,
per-cell product sequence, and fold tree remain fixed.

## Resource receipts

- Baseline droplet `602222795`: DELETE HTTP 204; follow-up GET reached HTTP
  404 at 2026-09-20 14:54:18 EDT; local dead-man cancelled afterward.
- Candidate droplet `602223807`: DELETE HTTP 204; follow-up GET reached HTTP
  404 at 2026-09-20 15:01:02 EDT; local dead-man cancelled afterward.
- Both source archives matched their remote SHA-256 receipts; both payloads
  exited zero. Raw logs are retained outside the repository under
  `mojolearn-evidence/gpt3-vendor-gemm-amd/`.
