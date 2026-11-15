# Verify decoupling, 2026-11-15 (lane verify-runtime-report)

Source: `docs/plans/VERIFY_COVERAGE_AUDIT_2026-11-15.md` section 4 and section 5 lanes 1 and 8. The shipped verify
modules (`python/mojolearn/_verif*.py`) read product internals (`_backend`, `_identity`, `_families`, `._b`, `._id`,
`._extension`, ...). They now read one public report instead, so a product change that moves an internal does not
break verify.

## API

`mojolearn.runtime_report()` returns a new plain `dict` on every call. It is built by ONE native call
(`backend_public_native_call("runtime_report", (), context)` in `bindings/backend_public_native_door.mojo`, the
compiled entry `_native_backend_public.py` already binds from the core host unit `_mojolearn_tokenizer_host`). No
new binding and no new loader. It never loads a binding, never calls a Python loader or hook, and is never on a
product call path: it is its own call, for verify and diagnostics. It reads the selection the package made at
import, and the loaded modules as they are at the moment of the call.

| field | type | meaning |
|---|---|---|
| `version` | `str` | `mojolearn.__version__`. |
| `vendor` | `str` | `'cuda'`, `'hip'`, `'metal'` or `'cpu'`; the same value and the same refusals as `mojolearn.vendor()`. |
| `numeric_mode` | `str` | The loaded tier, `'identical'` or `'fast'`; the same value and checks as `mojolearn.numeric_mode()` (raises when the loaded gbdt binary was compiled for the other tier). |
| `default_mode` | `str` | The process default a call without `numeric_mode=` uses (`mojolearn.set_numeric_mode`). |
| `code_mode` | `str` or `None` | What the loaded gbdt binary of the selected tier says it was compiled as (`'identical'`, `'fast'`, `'unknown'` for an unreadable code), or `None` when no gbdt binary is loaded or it has no read-back. Verify compares it with `numeric_mode`. |
| `cpu_only` | `bool` | True on a CPU-only install (no GPU binary set loaded). |
| `gpu_arch` | `str` or `None` | `mojolearn.gpu_arch()`. |
| `gpu_arch_how` | `str` or `None` | `mojolearn.gpu_arch_how()`. |
| `host_dir` | `str` | The directory the CPU (host) bindings load from (`mojolearn/host/`, or `MOJOLEARN_HOST_DIR`). |
| `host_families_built` | `tuple[str, ...]` | Sorted basenames (`_mojolearn_<family>_host`) of every host binding present in `host_dir`: its own `.so` files plus every member its merged-file index names whose file exists. |
| `bindings` | `dict[str, dict]` | Every native binding module of this package that is loaded, plus every member of a loaded tier that is not loaded yet. Key: the module name (`mojolearn._mojolearn_gbdt` for the selected binding, `mojolearn._sets.<mode>._mojolearn_<x>` for a tier member, `mojolearn._host._mojolearn_<x>_host` for a host binding). Value: `{'path': str, 'loaded': bool, 'identity': str (sha256 hex of the file's bytes, computed natively; one hash per distinct file), 'vendor': str or None (the binding's own compiled read-back `<suffix>_vendor()`; None when not loaded or it has none), 'mode': 'identical', 'fast' or None (the tier, for tier members)}`. |
| `fallbacks` | `tuple[str, ...]` | The names in the frozen authorized Python fallback list (`_runtime_fallbacks.FALLBACKS`); empty today. |
| `harness_source` | `dict` | `{'path': str or None, 'how': str}`: the identity harness this install runs, from `MOJOLEARN_IDENTITY_BREAK` (how `'MOJOLEARN_IDENTITY_BREAK'`), the wheel copy `mojolearn/_identity_break.py` (`'wheel copy'`), or the checkout's `tools/identity_break.py` (`'checkout'`); `path` None and how `'missing'` when none exists. |
| `source_commit` | `dict` | `{'commit': str or None, 'how': str}`: `MOJOLEARN_COMMIT` (how `'MOJOLEARN_COMMIT'`), else the wheel's `identity_columns/COMMIT` witness (`'wheel COMMIT witness'`, first token, 7 to 40 lowercase hex), else commit None with how `'git rev-parse HEAD of the checkout (the harness reads it)'`. |
| `checkout_root` | `str` or `None` | The repository root when the package sits in a checkout (the nearest of six parents holding `tools/identity_break.py`), else None. |
| `package_dir` | `str` | The directory of the imported `mojolearn` package. |

Rules for readers:
- A binding verify qualifies (path, sha256, vendor) is read from `bindings` AFTER the operation that loads it has run.
  `runtime_report()` never loads one; a binding that is neither loaded nor a tier member is absent.
- The report is read-only. Changing a value in the returned dict changes nothing in the product.
- A field is added here first, then in the native door, then used.
</content>
</invoke>
<invoke name="Bash">
<parameter name="command">bash -c 'cd ~/mojolearn-wt/verify-runtime-report && git add docs/plans/VERIFY_DECOUPLE_2026-11-15.md && git commit -q -m "VERIFY_DECOUPLE_2026-11-15: the runtime_report() API (fields, types, reader rules) the verify lanes code against

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && MOJOLEARN_WIDE_REBUILD_OK=1 git push -q origin HEAD:lane/verify-runtime-report > ~/mojolearn-evidence/briefs-20261115/vrr-push1.log 2>&1; echo rc=$?; tail -3 ~/mojolearn-evidence/briefs-20261115/vrr-push1.log; git log -1 --format=%h'