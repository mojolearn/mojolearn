# Ordinary hold qualification continuation

Branch `lane/qualify-ordinary-23`, based on main `10d3ad583`.

The six new kernel lanes were not selectable by `classical_host_gate.py`.
This branch adds saved-model record/replay probes using the harness's exact
0.125 float32 held-out scaling. Old RBF probes remain unchanged. This repairs
an actual recording blocker; it does not remove the six recording debts.

`tools/capture_ordinary_holds.py` captures the exact 23 ordinary holds with two
repeats, all nine fixtures, and all retained optional properties. Each lane has
its own strict harness checkpoint. Receipts/logs are retained per attempt;
wall-clock and per-stage limits terminate the process group. A GPU run also
records and CPU-replays the six kernel saved models against its GPU column.
The script supports source captures and exact installed-wheel captures: the
latter verify every wheel package file and require the installed harness to
match the saved-model tool's harness. It never admits references or promotes
holds. A zero exit is explicitly CAPTURED_UNQUALIFIED.

Validation: 48 lightweight tests passed (capture command/property selection,
resume selection, wrong/missing installed bytes, process timeout, kernel input
contract, and existing CPU gate/fault reporting).

Still owed: actual independent current NVIDIA/AMD captures, strict scoped
admission preserving all existing properties, observed installed CPU verifier
replay, and explicit promotion after those gates. Prior CPU/Apple kernel and
native fault records remain retained and are not reinterpreted as new hardware
or wheel qualification. Root coordinates rentals; this lane provisions none.

## Native checkpoint

The fixed six-kernel recording gate now ran successfully on Apple M4 and the
independent CPU saved-model path: 54 models (six lanes, nine fixtures), with all
54 replay outputs matching both their GPU recording and the retained Apple
identity inference column. Slot execution 5.55 seconds, one numerical worker.
Committed receipts, models, logs and recipe are in
`bench/results/classical_host/2026-09-18-apple-kernel-variants`.
Retained kernel-identity native binaries were reused with explicit hashes;
this is not a freshly rebuilt current-main native/wheel qualification.

## Remote execution

`tools/ordinary_holds_gpu_leg.sh` is an on-box source body; it never rents a VM.
It bounds builds and captures together, compiles eleven required binding
families serially, records each build log/status, derives exact archive commit
provenance, and runs the 23-route capture tool. Host compilation explicitly
clears GPU architecture and switches the column to CPU. The outer rental
controller must still enforce teardown and collect the output after failure.

Example after the guarded controller has staged this exact source:

```sh
MOJOLEARN_GPU_ARCHS=gfx942 bash tools/ordinary_holds_gpu_leg.sh \
  hip amd-mi300x-gfx942 /root/gemm_leg_out/ordinary-holds 3600
```

Build time has not been measured for this complete set on the new source. The
3600-second example is a hard total bound, not a claim the full set finishes
within it. Captures checkpoint per lane/cell; an incomplete stage stays pending.
For installed evidence, first install the new candidate and run
`capture_ordinary_holds.py --wheel <wheel> --python <venv-python>` with the same
backend/vendor/output/budget options; it verifies the installed package bytes.
All 23 holds remain until independent property admission and installed verifier
replay pass. Apple records now exist for the six saved-model debts; remaining
NVIDIA/AMD/installed qualification is still explicit.

Remote ordering refined: kernel build/capture/replay first, then each classical
family, then neural bindings/captures. No longer waits for eleven builds before
retaining its first evidence. Default total body budget is 2400 seconds (pass
2400 explicitly in the example above), leaving controller setup/fetch time in
a typical 60-minute rental guard. `MOJOLEARN_ORDINARY_LANES=kernel-ridge-poly,...`
selects any validated subset; unused families are not built. Missing/failed
families leave exit_code=1 while subsequent independent groups can still run.

The mocked remote-body regression confirms kernel capture occurs before GP
compilation, unused neural builds are skipped for a subset, GP build failure
still allows independent SVC capture, and the final receipt remains failed.
Combined capture/gate suite: 49 tests passed; these mocks are orchestration tests,
not additional GPU evidence.

Inventory follow-up: all six new selectable saved-model gates now appear in
`host_surface.inference_lanes()`. The inference registry test evaluates the
binding-free gate registry; its former literal-key regex missed generated lane
families. The separate debt registry continues to name outstanding NVIDIA/AMD
and installed qualification even though Apple models are recorded. Tests require
all nine real saved-model records for declared lanes retaining that qualification
debt. All 157 host-surface source tests passed.

## Fresh AMD references admitted, default promotion still pending

Single guarded MI300X 8-core run at `f771338c7`: all six kernel columns, all nine
fixtures twice, passed. All 216 numerical parts agree with retained CPU and
Apple columns. All 54 new AMD saved-model records replay exactly on the freshly
built independent CPU estimators binding. Evidence was fetched before the later
families finished; the active rental's final deletion receipt is still owed.

Strict scoped table admission changed only 54 selected lane/fixture cells; all
2,070 other cells remained identical. Each admitted cell has eight retained
parts with three agreeing classes: four numerical and four explicit N/A. These
classical lanes do not declare rlpair. Detailed source hashes and admission
checks are in `2026-09-18-ordinary-holds-amd-kernels/admission.json`.

The six lanes moved from `no reference` to qualification-pending candidates.
Default availability is unchanged; the holds are not removed. NVIDIA evidence
and installed CPU verifier replay remain owed. Source records are now usable
by diagnostic `--include-pending` verification rather than returning OWED for
all their numerical parts.

### Promotion policy versus release certification

- `_verify_reference.build_table` admits repeated current input/protocol/revision
  witnesses and preserves per-class disagreement. It does not, by itself,
  impose a hardware quorum or certify a wheel. This admission additionally
  required all CPU/Apple/AMD values to agree for every selected part.
- The five neural `one column` holds describe the absence of any second device
  class. Adding one agreeing GPU class clears that particular static reason,
  but still requires watched installed CPU replay before default promotion.
  The completion plan additionally requires current NVIDIA/AMD/full-property
  evidence; clearing the reason alone is not satisfying that broader plan.
- The twelve classical candidates explicitly owe NVIDIA/AMD alongside their
  existing CPU/Apple evidence and installed replay. The six new kernel
  candidates now owe NVIDIA and installed replay. They follow that same
  four-column completion policy, not a newly weaker rule.
- Exact release-wheel certification additionally pins native bytes, the wheel,
  target architectures and property/fault gates. A source reference admission
  or development installed replay does not replace this release certificate.

Validation after scoped admission: 217 host-surface/reference/coverage tests
passed. The separate 49 capture/gate orchestration tests also passed earlier.
