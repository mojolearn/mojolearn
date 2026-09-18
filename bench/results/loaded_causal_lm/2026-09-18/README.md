# Loaded-CausalLM local captures

Capture source commit `3accbf594`. `cpu.json`, `cpu-replay.json`, and `metal.json`
use the same profile-source digest, nine cases and six parts per case. All 54
CPU replay comparisons and all 54 CPU/Metal comparisons match bitwise. Each case
also independently reloads its checkpoint and checks logits against the first
load. Native input fixtures have two layers, nonzero weights, two rows, five
tokens, prefill length three and two carried-state decode steps. Tested formats
are FP32, BF16 and int8; architectures are Llama tied/untied and Mamba1 tied.

Retained native host and identical Metal bindings came from
`/Users/andrewhendel/mojolearn-wt/release-087-final/python/mojolearn/`;
actual loaded binding hashes are embedded. No new native compiler run. A
transient untracked `identical` directory symlink selected Metal binaries and
was removed afterwards. These are source-tree captures with retained binaries,
not clean rebuilt installed-wheel certification.

Three current captures ran serially inside `mac_slot.py --timeout 90 metal`,
`nice -n19`, OMP/OpenBLAS/VECLIB thread limit 1, in 5.4715 seconds total on the
local Apple machine. Records include platform, backend, requested execution
route, capture source commit, source digest and native binding hashes.

`exploratory-*` records are earlier captures, retained only as history; their
profile provenance differs. In particular `exploratory-cpu-native-fault.json`
loaded a directory called training-sabotage but reads native sabotage=False.
It matches clean bytes and is NOT a qualifying fault experiment. The observed
composition control removes one full layer and changes logits in every case.
Native arithmetic fault evidence remains owed.

No NVIDIA/AMD or physical multi-GPU claim, no installed-wheel admission, no real
external model smoke and no Mamba2/ragged fixture coverage are implied here.
