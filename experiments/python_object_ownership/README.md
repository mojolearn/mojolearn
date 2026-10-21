# PythonObject ownership investigation

This is an internal upstream-report draft and a standalone testing fixture.
No issue has been posted externally. The ordinary pinned Mojo 1.0 build
confirms reference retention in the supported interop API. No compiler or
standard-library files are patched, and no compensating reference operations
have been added to product code.

The extension imports only `std.python`, `std.python.bindings` and the standard
`abort` entry used for module initialization errors. Its six
operations distinguish generated binding overhead from direct attribute
assignment, attribute retrieval, builtin calls and a builtin list method.
`probe.py` compares CPython reference counts with an independent ownership
oracle, drops the expected owning references, collects garbage and checks weak
references. Neither source imports mojolearn.

Build once with the pinned supported compiler, retaining stdout/stderr and the
exit status. Put the resulting extension beside a copied `probe.py`, then run
the probe in an ordinary CPython process and retain its complete JSON report.

```sh
pixi run mojo build --emit shared-lib _mojo_python_ownership.mojo -o _mojo_python_ownership.so
python probe.py
```

The expected outcome is zero net retained references for every operation after
its intentional Python references have been removed. `no_op` and
`read_attribute` are controls. A mismatch is reported as a failed case and a
nonzero process exit; no result is converted into an expected pass.

The motivating project-level probe uses only
`CausalLMState(1, 4, layers, owner=owner)`. On the parent's compiled tokenizer
stage 19, the state itself is collected, but the caller's owner and original
list each retain one additional reference after deleting the state and
collecting garbage. Source and output are retained locally at:

- `~/mojolearn-evidence/finish-generation-lifetime-direct-assignment.py`
- `~/mojolearn-evidence/finish-generation-lifetime-direct-assignment.log`
- `~/mojolearn-evidence/finish-generation-lifetime-minimal.log`
- `~/mojolearn-evidence/finish-generation-lifetime-phases.log`

The broader lifetime regression in
`python/mojolearn/tests/test_model_native_lifetime.py` currently fails for all
three model families on that frozen binary. Numerical tests, native-execution
audits and the separately repaired exception-class ownership tests passed;
those passes do not establish correct object lifetime.

Reference API source, retained without changes:

- `~/mojolearn-evidence/mojo-1.0-python-object.mojo`, official
  `modular/modular` tag `mojo/v1.0.0`, `mojo/stdlib/std/python/object.mojo`:
  `__setattr__` lines 455–467 and `__call__` lines 1270–1297.
- `~/mojolearn-evidence/mojo-1.0-cpython.mojo`: borrowed-reference semantics of
  the ordinary CPython calls.

The standalone run on 2026-10-20 used Mojo 1.0.0 (`ed45d567`), CPython 3.14.7
and macOS 26.5.2 arm64. The build inside the supported pixi environment exited
0; the ownership probe exited 1. The initial compiler invocation outside that
environment failed to locate `std`; its log and exit code remain separate.
The same extension binary was independently loaded by the retained release's
CPython 3.12 interpreter and reproduced the same four failures. Exact command
paths, both interpreter versions and artifact hashes are in the result JSON.

| Operation | Extra value references after release | Weak-reference result | Verdict |
|---|---:|---|---|
| `no_op` | 0 | Collected | Pass |
| Direct `owner.value = value` | 1 | Retained | Fail |
| `return owner.value` | 0 | Collected | Pass |
| Builtin `type(value)` | 1 | Retained | Fail |
| Builtin `list(value)` | 1 | List cannot be weak-referenced | Fail |
| Builtin `owner.append(value)` | 1 | Retained | Fail |

The probe deletes the intentional attribute/list/result references before
measuring, and checks receiver references separately. Each receiver count is
unchanged. A testing-only control using ordinary Python operations passes all
six cases, exit 0; its retained report is
`~/mojolearn-evidence/finish-generation-ownership-probe-oracle-control.json`.

Complete source, binary, logs, probe JSON and exit statuses are retained at
`~/mojolearn-evidence/core-completion-20261020/stdlib-ownership/`.
[result-20261020.json](result-20261020.json) records the exact result and
SHA-256/size of each retained artifact. Numerical and native-call audit passes
remain valid for their measured boundaries; they do not qualify object
lifetime or complete native ownership.

The retained standard-library source suggests two ownership transfer sites
for upstream investigation: the setter passes an owned value through
`steal_data()` to `PyObject_SetAttrString`, and the call path adds `Py_NewRef`
around a transferred argument before inserting it into the call tuple.
This is a source-level diagnosis, distinct from the independently reproduced
retention. CPython documents the ordinary
[attribute-setting API](https://docs.python.org/3.14/c-api/object.html#c.PyObject_SetAttrString)
and the [reference stolen by tuple insertion](https://docs.python.org/3.14/c-api/tuple.html#c.PyTuple_SetItem).

The request for Modular is a supported release in which direct
`PythonObject` attribute assignment and ordinary calls balance their owned
references, including calls to builtin types and methods. Please retain the
six operation controls above and add corresponding error-path lifetime
coverage upstream. The same ordinary build and unchanged probe must return
exit 0 before this blocker is cleared; project model/state/buffer lifetime
tests must then pass as well. We will wait for that supported functionality
under the repository's explicit rule instead of patching the toolchain,
compensating references or substituting a Python runtime.
