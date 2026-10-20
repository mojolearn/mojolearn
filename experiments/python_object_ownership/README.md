# PythonObject ownership investigation

This is an internal upstream-report draft and a standalone testing fixture.
No issue has been posted externally. The standalone result is pending the
orchestrator's ordinary Mojo 1.0 build; no compiler or standard-library files
are patched.

The extension imports only `std.python` and `std.python.bindings`. Its six
operations distinguish generated binding overhead from direct attribute
assignment, attribute retrieval, builtin calls and a builtin list method.
`probe.py` compares CPython reference counts with an independent ownership
oracle, drops the expected owning references, collects garbage and checks weak
references. Neither source imports mojolearn.

Build once with the pinned supported compiler, retaining stdout/stderr and the
exit status. Put the resulting extension beside a copied `probe.py`, then run
the probe in an ordinary CPython process and retain its complete JSON report.

```sh
mojo build --emit shared-lib _mojo_python_ownership.mojo -o _mojo_python_ownership.so
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

The current evidence suggests an ownership problem in standard interop, but
the standalone build must confirm or reject that diagnosis before it is
classified as an upstream support blocker. If confirmed, the requested
upstream behavior is balanced reference ownership for the supported setters
and calls, with these positive and negative lifetime controls retained.
We will wait for supported Modular functionality rather than modifying the
toolchain or compensating reference counts in product code.
