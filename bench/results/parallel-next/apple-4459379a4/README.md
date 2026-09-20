# Apple parallel next-release trial

Python and harness source: 4459379a4b21bf8045ac53e01c3c448575afba4c. All 66 native libraries were extracted unchanged from the published 0.8.9 Mac wheel, native source 819a47ae48166e91951f54f54e64ee173658e32a; wheel and library hashes are in provenance.json. This run uses one physical Metal device, devices=(0,), and makes no multi-GPU placement claim.

The initial 25-lane base preflight completed with one refusal: par-causal-lm rejected Metal. Its complete raw record and full error sidecar are retained. The other 24 lanes then ran all nine fixtures with default full parts, one repeat, and --fail-on-refused. All 216 cells completed; exit 0; part verdict counts: {'STABLE': 819, 'N/A': 909}. A single repeat does not establish repeated-run stability.

par-causal-lm is explicitly pending its separate single-owner Metal code fix and collection; it is not included in the successful 24-lane scope. The revised par-arima model metadata needs the matching revised CPU reference; the old model hash is not silently accepted. These are next-release source tests and do not modify the published wheel.
