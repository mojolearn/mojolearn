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
