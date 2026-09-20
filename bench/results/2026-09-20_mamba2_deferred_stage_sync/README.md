# Mamba2 deferred stage synchronization

Date: 2026-09-20
Base: `658661b599a49280536bccc7e45370876c1ac363`
Hardware: Apple M4 Metal, local shared machine

## Finding

The production Mamba2 forward issued thirteen host drains per block between
kernels and copies already ordered on one `DeviceContext` queue. Disabled
identity tracing is the shipping state, and the public binding downloads the
output and carried state before returning. The candidate therefore retains
the old drains for enabled trace/card runs but defers them for production.
The SSD primitive keeps `drain=True` by default for standalone callers; the
block explicitly passes the trace state.

The price card runs the public block composition at B=2, L=770, d_model=64,
with carried state, twelve weighted layer calls, and a final synchronization
inside every timed interval. Thus the numbers include completion, rather than
measuring enqueue latency. Each binary warms once and then records seven
repetitions. Rotated baseline/candidate process order was B/A/A/B.

| rotation | baseline median | candidate median | change |
| --- | ---: | ---: | ---: |
| 1 | 321.250 ms | 160.148 ms | -50.2% |
| 2 | 244.641 ms | 140.480 ms | -42.6% |
| production-form repeat 1 | 284.899 ms | 226.417 ms | -20.5% |
| production-form repeat 2 | 319.956 ms | 222.247 ms | -30.5% |

The machine showed substantial thermal/background drift, but every rotation
favored deferred drains and the smallest observed median gain was 20.5%.
Every run produced exactly the same final output hash
`2462879301494985825` and carried-state hash `8082684280126638406`.

## Gates

- `pixi run check-mamba2-block`: PASS, all 26 traced stages bitwise equal to
  the oracle; repeated launches and batch-companion independence pass.
- `python3 tools/mamba_host_gen.py --check`: PASS after regenerating the two
  mechanically derived host files.
- `pixi run build-mamba2-ssd-backward-probe`: PASS.
- `pixi run mamba-grad-m2-public`: PASS, all 55 tensors in the public-prefill
  comparison.
- Long-sequence untraced output/state hashes: identical across baseline and
  candidate in all four rotated pairs.

No cloud resource was provisioned. Raw timing output is in `apple/`.
