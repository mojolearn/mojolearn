# Mamba3 performance continuation, September 9, 2026

All production changes in this lane are IDENTICAL-only. Numerical fold order,
state/report boundaries, buffer refusal order, and trace-enabled records remain
unchanged. FAST/DETERMINISTIC were neither built nor measured here.

## Evidence organization

- [Apple final gates and commands](apple/README.md) are archived by the root
  lane, including original/final traces and rebuilt combined binding tests.

- `l40s/m3-public`: initial public host-copy A/B. Its `surface-baseline.log`
  ran zero unittest cases and is **not** a surface validation; later explicit
  `test_mamba_surface.py` runs provide the real gates.
- `l40s/m3-finish`: trace-disabled download removal, full-output SHA comparison,
  combined Mamba1/Mamba3 Python surface, and Mamba1 device-decode gate.
- `l40s/m3-refusal`: integer first-bad-index GPU refusal scan and adversarial gate.
- `h100/m3-h100`: same-pod old-path baseline and optimized path through refusal
  scans. The old host-heavy path has substantial CPU/NUMA timing jitter.
- `h100/m3-next`: additional direct input/state/output transfers; no hardware
  fold. Five-round medians: smoke 1.600310 ms, narrow 104.846373 ms,
  wide 219.728276 ms. Full output SHA matches same-pod baseline for all rows.
- `opponent-admission`: original September 7 ours/torch logs and a read-only
  replay of their missing input-witness admission step. All 30 emitted input
  witnesses agree. The original numerical gates passed all three shapes.

The exact public grid is smoke B2/L4/D32, narrow B8/L4096/D512,
wide B8/L1024/D2048, seed 7. H100 driver 580.126.09 matches the historical
opponent hardware/driver; L40S driver 580.159.03 is a separate exploratory box.
The historical torch reference medians are 1.477005, 16.396241, 19.383267 ms.
They are a plain FP32 reference scan, not the upstream BF16 fused kernel.
Input admission was reconstructed from original logs; the original process
itself exited2 solely because `--mojo-log` was omitted.

## Rejected hardware arithmetic experiment

`h100/m3-hardware-rejection.log` records a genuine boundary mismatch on H100:
a=0x3f7fffff,b=0x00800000,c=+0. NVIDIA `fma.rn.ftz` returns +0, while the
existing software seam `ftz(fma(ftz(a),ftz(b),c))` returns 0x00800000.
The experimental runtime fold calls were withdrawn; no performance result
here enables that intrinsic. Parent also observed a pre-existing Metal
software-FMA boundary discrepancy. Same-baseline identity gates here do not
claim to repair or universally certify that separate cross-column issue.

## Final joint result

The default build at `8cda8cee`, with root's corrected GEMM/matrix and Mamba1
binding integration, passed the complete final H100 job at `h100/m3-joint-final`.
The manifest records exact hashes of those joint sources. Five-round medians:

| Shape | September 7 ours | Final ours | Historical torch | Final/reference |
|---|---:|---:|---:|---:|
| B2/L4/D32 | — | 1.631379 ms | 1.477005 ms | 1.10× |
| B8/L4096/D512 | 1275.875799 ms | 78.300729 ms | 16.396241 ms | 4.78× |
| B8/L1024/D2048 | 1650.892882 ms | 149.868213 ms | 19.383267 ms | 7.73× |

This is 16.29×/11.02× faster than the September 7 ours narrow/wide timings.
Reference comparisons use matching H100 model/driver and the exact original
fixture; they do not claim that both arms were retimed on one physical card
today. The original reference numerical gates passed, and its omitted input
admission was closed by read-only replay of all 30 original input witnesses.
The requested 1–1.5× reference target remains unmet for the large shapes.

Final NVIDIA validation: native 28-tag cards and oracle comparisons, eight
repeats and batch companions, B2/L65/D64, decode-cross, continuation, refusal,
153 adversarial refusal-helper cases, Mamba1 device decode, and Python surface
102 checks/0 failures. Full output SHA256 matches the old same-pod baseline
for every byte at all three grid shapes (two 64 MiB outputs plus 1 KiB smoke).
YSTATE scale-first sabotage still fails at its own first stage, with 10,380
moved cells. Parent final Apple default native gates passed, including original 28-tag
trace byte-equality at default and B2/L65/D64, decode-cross, continuation,
36 refusal plants, and rebuilt combined Python surface 102/0. An independent
agent reviewed both tiles and the direct constructor with no findings.

Final defaults: direct pinned transfers/fresh device weights, tiled ystate,
tiled QKS, and shared decayed V for IDENTICAL, each with a LEGACY A/B switch.
No rejected hardware-FMA primitive is called by Mamba3 runtime kernels.
The jointly corrected GEMM cost is included in final timings; intermediate
best prices are not substituted for it.

Remaining costs are documented in the final phase log: host input/weight/state
uploads and output/state downloads, projections, then the smaller serial
state/intra-chunk folds. No pageable-pointer experiment was staged or enabled.
