# Full-workload campaign materialization

The matrix is an inventory, not a runnable workload. Retained native H100
artifacts cover `I.X.complete-proposed` with 32 paired bindings; their `sm_90`
images are not the NVIDIA default/PTX track. The retained integration ledger has
no AMD or NVIDIA default-target receipts. Do not rent an empty timing worker or
compile without the owner's current authorization.

1. Freeze committed source and copy its Python package shell, excluding shared
   libraries and bytecode, to isolated A and B package directories. Deploy only
   exact compatible retained libraries. Preserve the original receipt, compiler,
   argv and artifact SHA256. A is candidate; B is incumbent. Keep mode/vendor
   directory layout accepted by `_backend.py`; do not relabel native binaries.
2. Supply saved per-workload facts to `six_lane_materialize.py --workloads` and
   concrete deployed package/receipt manifests to `--deployments`. The module
   docstring defines the inputs. These are independent of cell enumeration, so
   one audited full dataset/settings recipe can be reused across applicable
   configurations. Hash files on the destination worker, never instantiate a
   model or perform a smoke fit to materialize the recipe. Missing facts remain
   BLOCKED in `coverage.json`, with no synthetic or reduced replacement.
3. Run `six_lane_ab.py queue --vendor V --recipes recipes.json --select ID
   --output queue.json`. This writes worker documents but does not authorize or
   execute them. The campaign owner records authorization in the queue and
   admitted worker documents, then starts the full controller under the device's
   shared exclusive lease. Set a meaningful whole-cell timeout for the actual
   full workload. Preserve one excluded warmup and one scored sample per arm.

The full classical PCA/OLS/k-means saved `big` blocks are priority candidates;
existing metadata must show full train/query counts and native lane cap audits.
PCA's covariance fit and held-out transform both belong in the full operation.
GMM is admissible only with the existing explicit full-GMM recipe metadata.
Neural runtime-only public operations and ambiguous tree output consumers remain
pending until matching saved full recipes exist. Runtime controls declared only
for neural callers no longer block the combined configuration's classical PCA
workload, but unresolved controls within an affected family still block.

`task_quality=PENDING` and unavailable complete model state remain explicit in
master results; timing is not acceptance or promotion. The unchanged quality
functions provide metrics, followed by their independent saved gates. IDENTICAL
compares A across vendors and B across vendors separately; A/B equality is not
required. Do not infer bits or quality from compilation or elapsed time.

Future authorized builds support native AMD `--vendor amd --accelerator gfx942`
(or the actual supported native target), and separate NVIDIA
`--vendor nvidia --nvidia-target native|default` output roots. `default` preserves
compiler target selection by omitting the architecture flag; it is not a claim
that a given compiler output is portable PTX. No generic gfx target, IR rewrite,
unsupported compiler mode or toolchain workaround is introduced.
