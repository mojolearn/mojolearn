# Sub-byte layout wiring validation

Both commands exited zero on the local Apple column under the shared build lock:

- `pixi run check-sub-byte-layout`
- `pixi run check-hardware-matrix`

Each log contains exactly one final gate marker and all three expected
negative-control observations: fold coverage16/16 versus8/16, replica
collisions0 versus128, and shared allocation8192 versus4096 for8192 needed.
The umbrella also retains its existing `check_hardware_matrix OK` marker.

These are internal modeled negative controls, not expected process exits or
failed production compilation. No GPU kernel executes. No GPU qualification
is claimed. Production behavior is unchanged.
