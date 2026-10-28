# Python runtime probe

`tools/python_runtime_probe.py` is the empirical companion of the static scan in
`tools/public_runtime_inventory.py`: for every public name the scan enumerates, it builds the
object with `numeric_mode='identical'` on the CPU column (`MOJOLEARN_VENDOR=cpu`), runs each
public endpoint (constructor, `fit`, `predict`, `transform`, `score`, `get_params`,
`set_params`, properties and the rest) on the ordinary verifier's tiny fixtures, and records with
`sys.setprofile` every Python frame that executes strictly inside the public call's own entry
frame, plus the C functions those frames call and any Python frame entered from native code
(a callback). A public operation whose only Python frame is its entry, handing its arguments to one
native call, is clean; any other recorded frame is a finding, listed by file, function and count.
What it proves: on these inputs, on this build, these frames ran. What it does not prove: absence
of Python on other inputs, branches, devices or numeric modes; anything about bit identity, GPUs
or speed. It is CPU column only, uses small inputs, refusals are recorded as refused rows, and a
failed binding build shows up as refused or errored rows for the families it serves. Reports live
under `experiments/python_runtime_probe_<date>/`.
