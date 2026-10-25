# PythonObject instance-field lookup terminates the process

Prepared locally for Modular; this report has not been submitted.

## Reproduced behavior

Mojo 1.0.0 (`ed45d567`), macOS 26.5.2 build 25F84, arm64.
The same shared library fails under CPython 3.14.6 and 3.13.15.

The two-file reproduction is `_mojo_item_lookup_std.mojo` and `probe_std.py`.
It uses only `std.python`, `std.python.bindings` and `std.os`. It imports no
mojolearn package, project helper, NumPy, Torch or scikit-learn. It has no custom
C bindings, raw-pointer operations or explicit reference-count operations.

The operation first checks class descriptors, then returns an existing value
from a plain instance dictionary. The expected result is the same object that
the caller stored. The `instance` case instead terminates with SIGSEGV
(subprocess return code -11; shell status 139).

## Run

Copy the two files into an empty directory and use the pinned supported Mojo
environment:

```sh
mojo build --emit shared-lib _mojo_item_lookup_std.mojo -o _mojo_item_lookup_std.so
python probe_std.py --path original_shaped --case instance
```

Run each control in a separate process:

```sh
python probe_std.py --path original_shaped --case class
python probe_std.py --path original_shaped --case descriptor
python probe_std.py --path original_shaped --case missing
```

| Case | Python 3.14.6 | Python 3.13.15 |
|---|---:|---:|
| Existing instance value | -11 | -11 |
| Existing ordinary class value | 0 | 0 |
| Property refused without executing it | 0 | 0 |
| Missing value produces the expected error | 0 | 0 |

The library compiled successfully. Compilation does not qualify its runtime.
This fixture's source revision is `45e9ad477998769d1d7c7b8c2fb40802cd9c60bc`.
The report may be updated without rebuilding those unchanged source bytes.

## Scope and requested support

Please identify and fix the compiler or standard-library behavior responsible
for this supported PythonObject lookup. The reproduction establishes a
Modular-side interoperability failure; it does not establish which internal
compiler pass or standard-library implementation is responsible.

We need an existing-instance-value lookup to return the same live object, with
documented ownership, while preserving ordinary class fallback and missing-value
errors. Please provide a supported fix/release and confirm the regression under
both Python versions above. A fix must also preserve descriptor precedence and
must not require executing arbitrary Python property callbacks.

No compiler flags, standard library, binary output or reference counts have
been patched to compensate. No product fallback has been introduced. A direct
documented native dictionary API passed isolated diagnostic controls, but is
not an admitted product repair and does not replace public runtime validation.

## Retained evidence

Local evidence is under
`~/mojolearn-evidence/native-params-crash-20261025/standalone-stdlib/`:

- `build.log`: supported build output; compiler exit 0.
- `toolchain-and-files.json`: compiler, source, probe and binary hashes.
- `run-v1/result.json`: all eight process outcomes.
- `run-v1/*.log`: complete individual output, alongside the exact source,
  probe and binary bytes that were executed.

Earlier project-helper and self-contained C-API reductions, original public
runtime failures and corrected test-oracle evidence remain in sibling evidence
directories. They have not been overwritten by this narrower reproduction.
