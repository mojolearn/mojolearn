# Local installed qualification with retained binaries

`qualification_wheels.py` uses the ordinary split-wheel writer to assemble
`mojolearn-core` and the installed vendor packages from an immutable source
checkout, an existing installation, and explicit new native artifacts. It never
imports or modifies the existing installation, compiles a binding, changes a
binary, patches the backend, or admits a release reuse proof.

The output uses `linux_x86_64`, which PyPI does not admit. Every retained payload
file is verified against its old installed RECORD. Package source, native source,
binary, build-evidence, metadata, and final wheel SHA256 inventories are retained
in `qualification-inputs.json`. Native source hashes describe the checkout;
individual binary records state their actual retained/replacement origin. They
do not imply that old binaries were built from current source. `split-audit.json`
checks installed file ownership, version pins and package markers.

Example on the AMD qualification box (Python 3.11 or newer; the script itself may
be copied from a later tools commit while `--source-root` remains frozen):

```sh
freeze=/root/results-core-completion-20261020/freeze-c767f8fb68fe6d34d0b35a76515364a064f8001d
oldenv=/root/results-core-completion-20261020/venv
out=/root/results-core-completion-20261020/installed-c767-generation
mkdir -p "$out"
```

Write `"$out/native-overlay.json"`, substituting the actual retained binary and
build-log paths (the assembler requires both to exist):

```json
[
  {
    "member": "mojolearn/host/_mojolearn_tokenizer_host.so",
    "path": "/root/results-core-completion-20261020/freeze-c767f8fb68fe6d34d0b35a76515364a064f8001d/tokenizer-host/_mojolearn_tokenizer_host.so",
    "source_commit": "c767f8fb68fe6d34d0b35a76515364a064f8001d",
    "build_evidence": ["/absolute/path/to/retained/tokenizer-host-build.log"]
  }
]
```

Resolve the base interpreter's site path without importing mojolearn, then pack:

```sh
base_site=$("$oldenv/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
python3 /path/to/qualification_wheels.py \
  --source-root "$freeze/source" \
  --source-commit c767f8fb68fe6d34d0b35a76515364a064f8001d \
  --installed-site "$base_site" \
  --native-overlay "$out/native-overlay.json" \
  --out "$out/wheels" > "$out/pack.log" 2>&1
```

Retain the command exit status; any nonzero exit is a packaging failure. The
wheel directory must not already exist. Existing environments remain untouched.
Create a separate environment and install these exact local wheels:

```sh
python3 -m venv "$out/venv"
"$out/venv/bin/python" -m pip install "$out"/wheels/*.whl \
  numpy pytest > "$out/install.log" 2>&1
"$out/venv/bin/python" -m pip check > "$out/pip-check.log" 2>&1
```

For reproducible dependency comparisons, use the retained base environment's
NumPy/pytest versions instead of unpinned names. Run the actual installed verifier
outside both checkouts with `PYTHONPATH` and `PYTHONHOME` unset and the package's
ordinary GPU selection. This preliminary guard rejects silent CPU fallback:

```sh
cd "$out"
env -u PYTHONPATH -u PYTHONHOME MOJOLEARN_NUMERIC_MODE=identical \
  "$out/venv/bin/python" -I -c '
import importlib.metadata as md
import json
from pathlib import Path
import sys
import mojolearn
from mojolearn import _backend
package = Path(mojolearn.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert md.distribution("mojolearn-core").version == mojolearn.__version__
assert _backend.vendor() == "hip", _backend.vendor()
assert _backend.gpu_plugin()["distribution"] == "mojolearn-amd"
print(json.dumps(dict(package=str(package), vendor=_backend.vendor(),
    arch=_backend.gpu_arch(), plugin=_backend.gpu_plugin())))
' > "$out/installed-guard.log" 2>&1
```

This guard establishes location and selection only. The installed verifier must
still consume numerical outputs, audit the product call, retain its exit status,
and verify GPU launches. A package import or marker alone is not execution proof.

For NVIDIA, use its old installation site, change the guard to `cuda` and
`mojolearn-nvidia`, and supply every new native artifact explicitly. The original
PTX manifest is preserved and the package source witness names the current
checkout. If they differ, `ptx.current_source_admissible` is false and normal PTX
loading refuses. Do not change either witness to evade that check. A partial new
native architecture directory does not supply missing model block bindings.
Neither this controller nor its file-only tests claims complete target coverage.
