# MAX session initialization repair (2026-10-25)

The Apple MAX initialization failure was caused by the slot launcher's forced
`MODULAR_THREAD_BUSY_WAIT_US=0` environment override. The same MAX 26.5.0 public
session constructor succeeds when that override is absent. Remove the override
from `tools/mac_slot.py`; preserve its admission, lock, deadline, thread-count
and build-job policies. Caller-supplied runtime settings are still inherited.
No adapter, kernel, compiler, or runtime binary changes are needed for this
repair. Integrated installed-wheel MAX training qualification remains owed;
Torch-only success and session initialization do not qualify that adapter.

## Reproduction and evidence

The orchestrator ran the following public MAX API operation on the Mac, in the
retained installed-wheel gate environment, using the Metal execution slot:

```python
from max.driver import Accelerator
from max.engine import InferenceSession

device = Accelerator(0)
session = InferenceSession(devices=[device])
```

With the launcher's override present, imports and accelerator construction completed. Session construction aborted
at `max/engine/api.py:598`, in `_InferenceSession(...)`, with exit status 134:

```text
LLVM ERROR: AsyncRT::getOrCreateCPUDevice called requesting different options to those used to create the existing CPUDevice.
```

No graph was constructed, no kernel ran, and no mojolearn code was imported.
The previous explanation that mojolearn's nested training library caused this
specific failure was not established by the old one-line crash log. Process
isolation of the MAX and mojolearn test oracles did not repair it.

The orchestrator's bounded controls used Python 3.12.11 on macOS 26.5.2 arm64:

| Control | Runtime override | Exit | Result |
| --- | --- | --- | --- |
| `InferenceSession(devices=[Accelerator(0)])` | `0` | 134 | CPU-device options abort |
| `InferenceSession()` (CPU only) | `0` | 134 | Same abort |
| GPU session with `num_threads=1` | `0` | 134 | Same abort |
| Default GPU session | absent | 0 | Initialization completed |

The successful control used `env -u MODULAR_THREAD_BUSY_WAIT_US` **inside** the
slot command, so the old launcher could not reintroduce it. Existing controllers
started under the old launcher can still carry its environment: remove the
inherited override from those MAX launch commands, or start a fresh controller
with the repaired launcher.

Evidence directory: `~/mojolearn-evidence/max-runtime-final-20261025/`.

- `probe_init.py`, `probe-init.log`, and `probe-init.exit`: flushed stage markers,
  faulthandler stack, execution-slot result, and exit status.
- `probe_session.py`, `probe-cpu`, `probe-threads1`, and `probe-clean-env`
  `.log`/`.exit` files: the fresh-process controls above.
- `installed-file-integrity.json`: the installed MAX extension, engine Python
  module, `libmax`, and two shared runtime libraries match their distribution
  `RECORD` SHA-256 digests. No inspected installed source or runtime was patched.
- `source-tests.log`, `pytest.log`, and `scheduler-tests.log`: CPU-only
  source/diagnostic and scheduler checks; the combined pytest run passed 44 tests
  in 4.02 seconds. This is not MAX runtime qualification.

The retained interpreter is
`~/mojolearn-evidence/release/0.8.56/af15b57a96ad/gate-apple/adapters.failed-20261024-194734/venv/bin/python`.
Package metadata declares MAX, max-core and max-mojo-libs 26.5.0 and
mojo-compiler 1.0.0, matching the installed dependencies' requirements.
The PyPI release metadata checked on 2026-10-25 lists no 26.5 patch beyond 26.5.0.
MAX-core 26.6.0 requires mojo-compiler 1.1.0 and 26.7.0 requires 1.2.0; upgrading
MAX therefore requires an explicit toolchain qualification, not merely changing
the adapter pin. Selected release metadata is retained alongside the logs.

## Runtime configuration ownership

The installed `InferenceSession` implementation creates the host CPU device
automatically, then passes the supplied devices and `num_threads or 0` to the
native session. Its public thread setting defaults to automatic selection. The
documented device and thread parameters are supported configuration; replacing
CPU-device internals, manipulating private runtime state, or modifying compiler
output is not an acceptable mojolearn fix.

Primary references:

- [MAX InferenceSession API](https://max.modular.com/api/python/generated/max.engine.InferenceSession/)
- [Modular's AsyncRT options check](https://github.com/modular/modular/blob/b2d6b83ba7e17ddaf1c31acdf926c25194e44b2a/AsyncRT/lib/Runtime/HostSystem.cpp#L26)

The public source reference explains the fatal comparison in current upstream
source, whose options include `threadBusyWaitTime`; it is not a claim that the
installed 26.5.0 binary was inspected at that field. The failed/passing public API
controls establish the environment trigger without modifying those internals.

No upstream support blocker was established for the default session setup.
An optional upstream report could describe the nondefault environment interaction;
no report was sent. A dependency upgrade is unnecessary for this repair.

## Qualification owed after the prerequisite passes

Run `tools/check_max_training_adapters.py` against the integrated installed wheel
with its shipped `.mojoc` and native library, the retained Torch training outputs,
and `--smoke`; the normal release smoke supplies these paths. The checker now
persists the last attempted stage before MAX calls and records native-child exit
status so a subsequent abort is attributable to initialization, graph loading,
input transfer, execution, or output handling. Follow with the required complete
training/resume checks for the integrated source, and installed-wheel smoke on
each vendor. Reuse valid kernel identity evidence; adapters add no identity-table
columns or arithmetic sabotage arms.
