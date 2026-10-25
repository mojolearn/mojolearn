# Native dictionary lookup crash investigation

Testing-only reduction seed and documented CPython control; no product repair,
compiler change, compiler flag workaround, or reference compensation is present.
The four-case reduction has not yet been compiled or executed. Root owns that
single frozen build; lanes must not compile it.

The original complete tokenizer binary at source
`cc6ee39cc12f27e8daf44f83870b9321021e70d4` segfaults in public
`KMeans(...).get_params(False)` under CPython 3.14.6 on Apple arm64. It also
segfaults without the runtime auditor or Torch. Direct calls to the original
owner with test-only schema copies isolate the ordinary instance-field read:
empty names and the complete fields-only plan pass; class-field return,
property refusal, and missing-field refusal pass. The instance-field branch
segfaults. These diagnostic contexts are never installed in the product.

The raw crash reports `Py_NewRef` receiving `0x736e69746c697562` (ASCII
`builtins`). In the retained binary, `_value`'s present-field branch reaches
`PythonObject.__getitem__` at offset `0x6ce314`, using a stack argument area
which still contains an earlier String. This is evidence of bad argument
materialization, not yet a confirmed minimal compiler bug or an accepted repair.

`_mojo_item_lookup.mojo` preserves the exact original `_value` body. It imports
only its unchanged helper dependencies and exports two test controls: a bare
`return fields[name]`, and the documented CPython dictionary read with its own
owned reference. Python's [dictionary API](https://docs.python.org/3.14/c-api/dict.html#c.PyDict_GetItemWithError)
returns a borrowed reference; [Py_NewRef](https://docs.python.org/3.14/c-api/refcounting.html#c.Py_NewRef)
establishes the returned object's owned reference. A successful control is not
permission to change the product or a proof that every native callsite is safe.

Build exactly once from the frozen source checkout, using the existing supported
pixi compiler environment and its normal optimization settings. The root runs
this command, retaining the output and exit status outside the checkout:

```sh
pixi run mojo build --emit shared-lib -I . experiments/python_object_item_lookup/_mojo_item_lookup.mojo -o /absolute/evidence/directory/_mojo_item_lookup.so
```

Copy `probe.py` beside the extension. From that directory, use the clean pinned
CPython test interpreter, with no product PYTHONPATH. Run each case in a separate
subprocess so a signal is retained as a failed case and does not hide controls:

```sh
python probe.py --root /absolute/frozen/checkout --path original --case instance
python probe.py --root /absolute/frozen/checkout --path original --case class
python probe.py --root /absolute/frozen/checkout --path original --case descriptor
python probe.py --root /absolute/frozen/checkout --path original --case missing
python probe.py --root /absolute/frozen/checkout --path direct --case instance
python probe.py --root /absolute/frozen/checkout --path direct --case missing
python probe.py --root /absolute/frozen/checkout --path cpython --case instance
python probe.py --root /absolute/frozen/checkout --path cpython --case missing
```

A probe passes only if it preserves returned identity or raises the required
refusal without invoking the supplied Python property. No crash is an expected
pass. Preserve source bytes, compiler/interpreter versions, exact command,
compiled SHA-256, full per-case logs, and actual exit/signal results.

Original complete-binary evidence is retained at
`~/mojolearn-evidence/native-params-crash-20261025/`: `plain.log`, `pytest.log`,
`raw-crash.ips`, `value-disassemble.log`, `fields.log`, `class.log`,
`descriptor.log`, `missing.log`, and per-process `.rc` files. Failed debugger
launches are retained separately and were terminated; no service was installed.
