# BRIEF: expose the GPT-2 tokenizer (workstream D, first item), 2026-09-14

Branch `lane/expose-tokenizer` from `71faae781`. The claim-surface census
(`docs/lanes/BRIEF_claim_surface_census_2026-09-14.md:217`) listed 1,861
lines of tokenizer code that are built and gated with no public door, and
the plan (`docs/lanes/TEMP_claim_surface_plan_2026-09-14.md:40,99`) made it
the first lane of workstream D because it has no float arithmetic. This
brief is the census, the door, the identity lane body for the harness's
owner, the host answer and the doc corrections. Everything measured here
was measured on ONE Apple M4 under `nice -n 19` with two threads; NVIDIA
and AMD runs are owed and section 5 gives the commands.

## 1. Census (before this lane)

Where it lives. `tokenizer/`, 1,955 lines by `wc -l` (the census counted
1,861 without `tools/gen_unicode_categories.py`):

| file | lines | what |
|---|---|---|
| `tokenizer/encoding.mojo` | 204 | the public Mojo surface: `Gpt2Tokenizer` with `encode_bytes` (`:105`), `decode_bytes` (`:150`), `encode`/`decode` string wrappers, `token_spelling`, `load_gpt2_tokenizer` (`:191`) and `load_gpt2_tokenizer_from(ranks_path, unicode_path)` (`:199`); `GPT2_N_VOCAB = 50257` (`:46`), `GPT2_ENDOFTEXT_ID = 50256` (`:49`) |
| `tokenizer/impl/pretokenize.mojo` | 190 | the seven-alternative pattern, hand-rolled |
| `tokenizer/impl/ranks.mojo` | 210 | `RankTable`, `load_rank_table(path)` (`:143`), `token_bytes`, `append_token_bytes` |
| `tokenizer/impl/unicode_class.mojo` | 232 | `\p{L}`, `\p{N}`, `\s` range tables, `load_unicode_classes(path)` (`:100`), the UTF-8 decoder |
| `tokenizer/impl/bpe.mojo` | 86 | the merge loop |
| `tokenizer/impl/byte_unicode.mojo` | 103 | GPT-2's byte-to-unicode bijection |
| `tokenizer/checks/tokenizer_check.mojo` | 445 | THE GATE, `main()` at `:325` |
| `tokenizer/checks/json_lite.mojo` | 385 | the fixture reader |
| `tokenizer/data/gpt2_ranks.tsv`, `unicode_categories.tsv` | 982,310 and 10,988 bytes | the two tables, read at run time from repository-relative paths (`encoding.mojo:43-44`) |

Entry points and what the existing check proves. `pixi run check-tokenizer`
(`pixi.toml:1109`) runs `tokenizer/checks/tokenizer_check.mojo`: 43 cases
recorded from tiktoken 0.14.0 (`tokenizer/checks/fixtures/gpt2_reference.json`),
exact id-sequence equality and byte-exact decode round trips, plus the
pattern/table/bijection preconditions and the pre-tokenizer reach arms
(`tokenizer_check.mojo:3-40`). `endoftext_as_special` is the one case run with
the special token allowed (`:341-345`). The README records three sabotages, two
of which only the reach arm catches (`tokenizer/README.md:90-106`). No float
arithmetic anywhere, so there is no IDENTICAL arm by construction
(`encoding.mojo:9-17`).

What the shipped bindings exported for it: nothing. `git grep -n -i tokeniz
-- bindings/` returned no line before this lane; `def_function` registrations
mentioning it: none. `python/mojolearn/` wrapped nothing either: the only two
hits for "tokenizer" under `python/mojolearn/*.py` were prose in
`_byte_lm_impl.py:396` and `tools/lm_step_memory_probe.py:238` saying the byte
LM has no tokenizer. `tokenizer/README.md:108-111` and
`tokenizer/NOT_IMPLEMENTED.tsv:13` said so in words, and main's
`python/mojolearn/__init__.py` registered it in `_NOT_YET` as `"Tokenizer"`.

What was missing for `import mojolearn` to reach it: a binding that moves a
byte string in and int32 ids out (and back), a Python class in `__all__`, a
way to find the two tables from Python, tests, and a manifest entry so the
CPU gate and the docs know the binding exists.

## 2. The door

Design. The tokenizer has no GPU binding to be a twin of, so the door is a
HOST binding that is the family's only one, loaded by path like the byte
LM's and the forest's, through the loader every host family already uses
(`_backend.load_host_module`, which reads back vendor `cpu`, IDENTICAL and
the CPU column and refuses a sabotage build outside the gate). The same
binary serves a GPU box and a CPU-only install.

| piece | file |
|---|---|
| binding | `bindings/_mojolearn_tokenizer_host.mojo`, module `_mojolearn_tokenizer_host`, exports `tokenizer_host_numeric_mode`, `tokenizer_host_vendor`, `tokenizer_host_column`, `tokenizer_host_sabotage`, `gpt2_load(ranks_path, unicode_path) -> _Gpt2Handle`, `gpt2_n_vocab(h)`, `gpt2_max_token_bytes(h)`, `gpt2_encode(h, text_addr, n_bytes, out_addr, out_cap, allow_endoftext) -> count`, `gpt2_decode(h, ids_addr, n_ids, out_addr, out_cap) -> count`. The handle owns the parsed tables (`add_type`, `PythonObject(alloc=...)`, the byte LM session's pattern), so the 982 KB rank table is parsed once per `GPT2Tokenizer`, not per call. Ids cross as int32 at a caller-sized address; an encoding never has more ids than bytes, so `out_cap = n_bytes` suffices; decode's output is bounded by `n_ids * gpt2_max_token_bytes`. The address contract is in the file's docstring and mirrored in the Python module |
| build | `bindings/build_tokenizer_host.sh`, the two-line shim over `bindings/build_host_family.sh tokenizer` (usage string extended) |
| Python | `python/mojolearn/tokenizer.py::GPT2Tokenizer`: `encode(text, allow_endoftext=False)` (str or bytes-like), `encode_bytes`, `decode(ids, errors="replace")` (tiktoken's own default), `decode_bytes`, `n_vocab` (50257), `eot_token` (50256), `data_directory`; `data_dir()` resolves `MOJOLEARN_TOKENIZER_DATA`, then `mojolearn/data/tokenizer/` (absent until a wheel carries it), then the checkout's `tokenizer/data/`, refusing by name. Exported as `mojolearn.GPT2Tokenizer` and `mojolearn.tokenizer`; the `"Tokenizer"` row left `_NOT_YET` in the same commit |
| refusals | text not str or bytes-like (TypeError, names the type); `encode_bytes` given a str; `allow_endoftext` not a bool; ids not a sequence; an id that is a bool or not an int (position named); an id outside [0, 50257) (value and position named, before the binding is called; the binding refuses the same way); a data directory missing a table (path named); an unbuilt binding (ImportError naming the build script) |
| manifest | `python/mojolearn/host_surface.py` family `tokenizer` (`routes=None`, classes `GPT2Tokenizer`, sabotage define `MOJOLEARN_TOKENIZER_HOST_SABOTAGE`, gate `pixi run check-tokenizer` and `test_tokenizer_surface.py`, not in a wheel); `markdown_table` now shows the classes of every path-loaded family, not the byte LM's alone; the CPU identity gate workflow builds the binding so its read-back step, which takes every declared binding, can pass |
| docs | `python/mojolearn/ALPHA_API.md` row; `SUPPORT_MATRIX.md` capability row and the regenerated CPU surface table; `docs/BYTE_LM_CPU_TRAINING.md`'s regenerated table; `tokenizer/README.md` and `NOT_IMPLEMENTED.tsv` corrected |
| sabotage | `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` writes encode's ids in reverse; `tokenizer_host_sabotage()` reads it back |

Tests, as modules (`cd python && python3 -m mojolearn.tests.<name>`) and
under pytest:

| module | checks | needs the binding |
|---|---|---|
| `mojolearn.tests.test_tokenizer_surface` | 13: ten fixed byte strings with their ids spelled out; the 43 recorded cases id for id; the 43 round trips byte for byte; invalid UTF-8 round-tripping; both readings of `<|endoftext|>` and the split at it; `decode` text and `errors="strict"`; the vocabulary constants; the read-back trio; five refusal groups by name; the missing-table refusal | yes (exit 2 or pytest skip, naming the build script, when unbuilt) |
| `mojolearn.tests.test_tokenizer_manifest` | 8 source checks: manifest entry, `__all__`, ALPHA_API, the shim, the gate build step, the sabotage define and the export list against the binding source, table resolution, the by-name refusal before any load | no |

Measured on the Apple M4 (section 6).

## 3. The identity_break lane body (for the harness's owner; NOT added here)

`tools/identity_break.py` is owned by another session this week and was not
edited. The lane below follows the byte-lm lane's derivation
(`identity_break.py:909-924`): the fixture bytes are the fixture's float32
values viewed as bytes, as `_ids` does at `:698-705`, here without the
modulus because the tokenizer takes every byte value. `ties` therefore hands
it a stream heavy in repeated bytes and `hashed` a flat one; `denormal` and
`denormal_ftz` differ in exactly the bytes their subnormals occupy, so the
two cells must differ. The train column hashes the ids of the first 4,096
fixture bytes with the special token allowed, and the decoded bytes; the
infer column encodes the held-out fixture's first 4,096 bytes; the model
column is `n/a:no-save` since the tokenizer saves nothing, which the harness
records on its own when `save`/`save_checkpoint` are absent (`:1065`).

```python
@lane("tokenizer")
def _(ml, X, yc, yr, Xh=None):
    """GPT2Tokenizer (python/mojolearn/tokenizer.py), host integers and
    tables through _mojolearn_tokenizer_host; no float arithmetic, so
    cross-vendor identity is by construction and what this measures is
    that the SAME binary bytes were built on every box. The fixture bytes
    are the first 4,096 bytes of X viewed as bytes (the byte-lm lane's
    derivation without _ids' modulus), encoded with <|endoftext|> allowed,
    then decoded back; the held-out probe encodes Xh's first 4,096 bytes."""
    tok = ml.GPT2Tokenizer()
    raw = np.ascontiguousarray(X).tobytes()[:4096]
    ids = np.asarray(tok.encode_bytes(raw, allow_endoftext=True), dtype=np.int32)
    back = np.frombuffer(tok.decode_bytes(ids.tolist()), dtype=np.uint8)
    assert back.tobytes() == raw, "tokenizer lane: decode(encode(x)) != x"
    return _fit(dict(ids=_h(ids), decoded=_h(back), n_vocab=_h(np.int64(tok.n_vocab))),
                tok, lambda e: (np.asarray(e.encode_bytes(np.ascontiguousarray(Xh).tobytes()[:4096],
                                                          allow_endoftext=True), dtype=np.int32),))
```

The CPU column: the binding is host code, so on a CPU-only runner the lane
reads STABLE, not REFUSED; the manifest lists no `training_lanes` for it
because the lane is not a training lane and the gate's `--require-columns 4`
diff is over training lanes. Once the lane is in, add `"tokenizer"` to the
family's `inference_lanes` only if `tools/classical_host_gate.py` grows a
matching entry (`test_inference_lanes_are_classical_gate_lanes` holds the
two lists equal); otherwise leave both empty, as now.

**Superseded on the `training_lanes` point (11c5f2192, 2026-09-15).** The
family now declares `training_lanes=("tokenizer",)`. What changed is the gate,
not the lane's nature: the CPU identity gate builds every family into its
sabotage host set with that family's own define, so
`MOJOLEARN_TOKENIZER_HOST_SABOTAGE` reverses the encoded ids and the lane's
train, infer and batch parts all move. `inference_lanes` is still empty, and
for the reason this paragraph gives.

## 4. The host twin

None is needed and none was written: the tokenizer never runs on a GPU. The
binding that serves it on a CPU-only install is the same
`_mojolearn_tokenizer_host` every install uses; there is no GPU entry whose
ids a CPU path must reproduce. The manifest entry says so (`routes=None`).
What can differ between boxes is the Unicode table generated by
`tokenizer/tools/gen_unicode_categories.py` (`tokenizer/README.md:81-88`),
which is committed, not generated at build time, so it cannot.

## 5. Owed runs (no box was rented)

The Python door through the built binding on each vendor's box, from a
checkout at this lane's head with the host binding built there:

    # NVIDIA H100 (RunPod) or AMD MI300X (Hot Aisle), inside the leg after the GPU set is built:
    MOJOLEARN_BUILD_JOBS=2 sh bindings/build_tokenizer_host.sh
    pixi run check-tokenizer
    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_tokenizer_surface
    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m pytest -q mojolearn/tests/test_tokenizer_surface.py mojolearn/tests/test_tokenizer_manifest.py

and, once the section 3 lane is in the harness, the three-column rerun that
session already runs picks the `tokenizer` row up with no extra command.
The seven-runner CPU gate run happens on the push to main that merges this
lane (`.github/workflows/cpu-identity-gate.yml` now builds and reads back
the binding).

## 6. What was measured on the Apple M4

One box, an Apple M4 under `nice -n 19`, `OMP_NUM_THREADS=2
MOJOLEARN_CPU_THREADS=2 MOJOLEARN_BUILD_JOBS=1`, one run at a time,
2026-09-14 07:06 to 07:12 local. Nothing here is evidence for any other box.

| what | result |
|---|---|
| `sh bindings/build_tokenizer_host.sh` | first build FAILED: `add_type` derives `write_repr_to` by reflection and `Optional[Gpt2Tokenizer]` is not Writable; spelled out (the byte LM session does the same); second build wrote `python/mojolearn/host/_mojolearn_tokenizer_host.so`, 350,688 bytes |
| `python3 -m mojolearn.tests.test_tokenizer_surface` | first run RED on two of MY expectations, not the binding: the `raw_bytes` fixture is three bytes (`\x00\x01\x7f`), not the two I had copied, and "é" is one GPT-2 token (2634), not two bytes; the 43 recorded cases and the 43 round trips passed on that run. Fixed the expectations; second run GREEN, 13 checks: 43/43 exact id sequences, 43/43 byte-exact round trips, ten fixed byte strings, both readings of `<\|endoftext\|>`, invalid UTF-8 round-tripping, `decode` under `errors="replace"` and `"strict"`, the read-back trio and five refusal groups by name |
| `python3 -m mojolearn.tests.test_tokenizer_manifest` | GREEN, 8 source checks, no binding loaded |
| `pytest -q` on the two tokenizer modules and `test_host_surface.py` | 74 passed |
| `python3 packaging/check_ext_lists.py` | all five lists agree (16 extensions; the host binding is not a `_MODULES` entry, as no host binding is) |
| `python3 tools/docs_facts.py --write` then `pixi run check-docs-facts` | rewrote the CPU surface table in `SUPPORT_MATRIX.md` and `docs/BYTE_LM_CPU_TRAINING.md` (one tokenizer row each); check OK, 13 facts, 12 marked spans |
| sabotage build (`-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` into a scratch directory, `MOJOLEARN_HOST_DIR` pointed at it, `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`) | the surface module went RED on 6 of 13 checks: 37 of 43 id sequences differ (every multi-id case, reversed), 36 of 43 round trips differ, the fixed strings, both `<\|endoftext\|>` readings, the invalid-UTF-8 round trip and the decode text check all fail. The 7 checks that still pass are the ones a reversal cannot move (single-id cases, refusals, constants, read-back). So the test reads this binary |
| the same sabotage build without the allow flag | NOT RUN, exit 2: `_backend.load_host_module` refused it by name as a SABOTAGE build |
| the one full `pytest -q mojolearn/tests` (`pixi run -e test`) | 1059 passed, 82 skipped, 89 subtests passed, 13.26 s; no failure |

Two deprecation warnings the compiler prints for this binding are shared
with its neighbors and left as they are: positional `__getitem__` on a
`MutPointer` (`bindings/_mojolearn_tsa_host.mojo` indexes the same way) and
`memcpy` (`bindings/hostptr.mojo` calls it).

## 7. Doc corrections (statements found FALSE, and what is true)

| where | said | true |
|---|---|---|
| `tokenizer/README.md:108-111` (before) | "No Python surface and no `bindings/` entry point" | `mojolearn.GPT2Tokenizer` over `bindings/_mojolearn_tokenizer_host.mojo`; FIXED in this lane |
| `tokenizer/NOT_IMPLEMENTED.tsv:13` (before) | a Python surface and a bindings entry point NOT IMPLEMENTED, "a binding would need to move `List[Int]` ids across the boundary, which is the converter question `mojolearn-native-convert-lane-sep10` owns" | implemented; the ids cross as int32 at a caller-sized address, no converter needed; FIXED |
| `python/mojolearn/__init__.py` `_NOT_YET["Tokenizer"]` (main at 71faae781) | door-less | the row is deleted in this lane, as the register's own rule says |
| `python/mojolearn/host_surface.py:35,394` (before) | "the two bindings loaded by path" | three; FIXED |
| `python/mojolearn/_backend.py:875-883` (before) | "The two bindings loaded by path" | three; FIXED |
| `.github/workflows/cpu-identity-gate.yml:9` (before) | "nine families as of 2026-09-14" | ten; FIXED |
| `bindings/build_host_family.sh:44` (before) | the usage string listed nine families | ten; FIXED |
| `docs/lanes/BRIEF_claim_surface_census_2026-09-14.md:217,264` and `docs/lanes/TEMP_claim_surface_plan_2026-09-14.md:25,40` | tokenizer has no public door | dated records of the census; true on their date, superseded by this lane; NOT edited (not this lane's files) |
| `docs/lanes/BRIEF_claim_surface_census_2026-09-14.md:217` | "1,861" lines | `wc -l tokenizer/**/*.mojo tokenizer/tools/*.py` is 1,955; the census excluded the 94-line generator. Not a falsehood, a different count; noted, NOT edited |
