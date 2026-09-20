# Chunked LM-head v2 NVIDIA qualification

- Date: 2026-09-20
- Exact source: `92aee317bd77ed3ca5f2565b58a6db4b918535d5`
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, driver 580.159.04
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Numeric mode: IDENTICAL; architecture/cache partition: sm_89
- Transport: exact git archive; R2 staged two artifacts; bincache staged 149
  entries. Compiler concurrency was capped at two.

The direct device gate printed `CHUNKED_LM_HEAD_V2_DEVICE_OK`. Loss, row
maxima, denominators, dHidden, and dWeight matched the normative CPU oracle
bit-for-bit across the 513-token boundary fixture. The surrounding GEMM card
also matched Apple at all 60 hashed stages.

End-to-end ByteTrainer timings in nanoseconds:

| run | small V1 | small V2 | large V1 | large V2 |
|---:|---:|---:|---:|---:|
| 1 | 1,478,437 | 4,314,508 | 2,070,616 | 84,122,502 |
| 2 | 1,470,217 | 4,325,539 | 2,050,246 | 84,180,194 |
| 3 | 1,484,137 | 4,291,808 | 2,068,306 | 84,146,203 |

Small is B=1, L=8, DM=16, V=513. Large is B=1, L=64, DM=64,
V=8,192. Large retained head-path storage is 1,323,009 float cells for V1
and 16,389 for V2. `nvidia-smi` reported 1 MiB used before and after the
process; the persistent allocation accounting is therefore the useful
per-trainer memory witness at this scale.

At V=8,192 the four exact recomputation passes execute 32 chunks each: 128
chunk GEMMs plus their serial fold/update launches. The stable 84.1 ms V2
versus 2.06 ms V1 gap makes launch/recomputation the next bottleneck. No new
kernel arm was landed: removing passes or combining independently reduced
chunk gradients would violate the fixed V2 operation order, and no exact
guarded fusion was qualified during this rental.

The first pod (`4pvttd16wqr1wn`) proved the oracle gate but its optional timing
loop used an unavailable `/usr/bin/time`; it was deleted with HTTP 204 and
verified absent with HTTP 404. The retained second pod (`skbiaitzkbygbq`)
completed every gate and timing with `extra_exit=0`, then was likewise deleted
with HTTP 204 and verified absent with HTTP 404. Persisted R2/cache data was
not deleted.
