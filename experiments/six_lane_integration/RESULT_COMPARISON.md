# Retained result comparison

`tools/six_lane_compare_results.py` consumes selected retained full-workload queue
receipts. It compares scored A across routes and scored B across routes
separately. A may differ from B. It never runs estimators, reads tensor payloads,
rebuilds artifacts, invokes a board writer, admits measurements, or promotes a
default.

IDENTICAL requires `nvidia-native`, `nvidia-ptx`, `amd`, `apple`, and `host` for a
complete comparison. Every available pair has its own result under
`cases[].arms.A.pairs` and `cases[].arms.B.pairs`. NVIDIA native/PTX versus AMD can
report `MATCH` while the five-column result remains `INCOMPLETE`. Missing model
state cannot pass, even when output hashes match. Apple timing does not vote on
IDENTICAL promotion. Apple FAST reports identity `NOT_REQUIRED`; its retained
quality gate remains separate and receipt coverage can still be incomplete.

## Inputs

Supply a manifest with schema `mojolearn.six-lane-comparison-input/1` and a
nonempty `cases` list. Each case has these fields:

| Field | Value |
| --- | --- |
| `id` | Unique report case ID; stable across snapshots |
| `configuration_id` | Frozen `workload.master_selection.id` |
| `implementation_ids` | Exact namespaced IDs attributed by the scored worker |
| `source_sha` | Full frozen worker checkout commit, not the comparator's HEAD |
| `workload_id` | Exact saved full-workload ID |
| `mode` | `identical` or `fast` |
| `expected` | Reviewed scope and logical controls described below |
| `columns` | Map from the route names above to selected receipt paths or objects |

`expected` must pin `dataset_sha256`, `dataset_version`, `dataset_split`, `seed`,
`dimensions`, `estimator_settings`, `harness_sha256`, and `timed_boundary` to the
saved recipe. `dimensions` and `estimator_settings` use the exact worker record
formats. Do not learn expected values from whichever column happens to finish
first. All compared columns must use the pinned source and scope. This tool does
not decide whether differing source commits have equivalent numerical closures.

Also include `expected.configurations.A` and `.B`, each containing exactly
`defines` (list), `environment` (object), and `runtime` (object). B is the saved
incumbent, with its existing defaults. Defines are compared independent of list
order; duplicate macro names are refused. Every numerical switch remains part of
the logical configuration.

`expected.capture_paths.outputs` and, for IDENTICAL,
`expected.capture_paths.model_state` are nonempty lists of exact leaf paths from
the reviewed complete capture contract. Each capture must declare
`complete_declared_scope`, an empty `missing_state` list, a hash, scope, encoding,
and typed manifest with unique paths, dtype and shape. Its path set must equal
the declared set. Do not invent state paths or treat an unavailable public state
export as complete state. Unknown state coverage stays incomplete.

Set `expected.repeated_operations` to the saved recipe's repeated-operation
count when applicable (default zero). All declared repeated output and state
captures are compared separately by index. Extra, missing or undeclared repeated
captures block comparison rather than being silently omitted.

A column can be a path string, or an object:

```json
{
  "receipt": "amd/cell/attempts/attempt-0002/receipt.json",
  "history": ["amd/cell/attempts/attempt-0001/receipt.json"],
  "transport_environment": {
    "A": {"MOJOLEARN_VENDOR": "hip"},
    "B": {"MOJOLEARN_VENDOR": "hip"}
  }
}
```

Paths resolve relative to the comparison manifest. Omit `transport_environment`
unless these keys actually occur in the scored `configuration.environment`.
Only explicitly supplied `MOJOLEARN_VENDOR` and `MOJOLEARN_CUDA_CODE_FORMAT`
values may be excluded from the logical configuration. Other differing controls
block comparison. Compiler, hardware and artifact hashes may differ by column;
their provenance is retained rather than compared for equality. The queue's
declared artifact path/hash map must match the worker's observed loaded map,
with numerical source, compiler, target and defines present. This reads metadata
only; it does not revalidate builds or open libraries.

Route names are explicit collector attributions in the input manifest. Select
the actual native/PTX artifacts and retain their target provenance. The tool
does not infer code format from GPU hardware. Reusing identical receipt bytes
for distinct columns is refused.

The selected file must be a queue `receipt.json` with embedded
`mojolearn.full-ab-result/1` scored results in `runs[].result`. Exactly one
non-excluded, successful scored sample per arm is required. Warmup payloads are
not compared. Failed/unfinished attempts, changed scopes, duplicate scored
samples, or missing typed capture remain incomplete. Complete worker logs and
result-file paths remain referenced. Explicit history entries retain original
failure summaries and hashes; `previous_receipt` links are also preserved but
are not automatically followed or used as replacement evidence.

## Invocation and outputs

After the owner authorizes the retained-result assessment:

```sh
python3 tools/six_lane_compare_results.py \
  --manifest /retained/comparison-input.json \
  --out /retained/comparison-snapshot-001
```

The output directory must be new. The tool writes:

- `report.json`: exact input hashes, selected receipts, scoped per-arm pairs,
  `MATCH`/`MISMATCH`/`INCOMPLETE`, missing coverage, failures and provenance.
- `inventory.json` and `index.json`: inputs for the existing
  `tools/performance_measurement_board.py`.
- `future_command.json`: that board tool's argv, with `execution: NOT RUN`.

Exit 1 reports a mismatch; exit 2 reports incomplete receipt coverage; exit 0
means the requested comparison is complete or identity is not required. None of
these exit codes establishes task quality, campaign-wide coverage or promotion.
Malformed top-level input is an error. `MATCH` trusts the collector's declared
capture contract; it cannot establish coverage of unexported model state.

Board cells retain timings, quality and identity separately. New cells remain
`PENDING_ADMISSION`, `FAILED_OR_INCOMPLETE`, `IDENTITY_MISMATCH`, or
`QUALITY_FAILED`; this tool never synthesizes `MEASURED` rows or opponent ratios.
Use the existing board tool to render these pending inputs. Existing task-quality
assessment and admission remain separate later actions.

To retain an earlier board snapshot, pass both `--previous-inventory` and
`--previous-index`. Original cells, notes and decisions are copied unchanged;
new evidence is appended. Inputs and existing reports are never overwritten.
No incumbent/opponent board or historical comparison is modified directly.

## Validation scope

`tools/test_six_lane_compare_results.py` uses invented metadata only, including
different A/B digests, missing columns/state, scope drift, typed metadata
mismatches, failed attempts, transport controls and history preservation. It does
not import a product binding, execute an estimator, measure time, compare actual
candidate outputs, or qualify a device. Passing these fixtures is not execution
verification of the measurement harness.
