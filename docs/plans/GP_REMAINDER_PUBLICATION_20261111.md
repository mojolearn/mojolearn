# GP remainder: native publication and output packing

Source `3432926612bb2f5beea7aaa061e4911694add25c` follows the frozen GP score
tranche. It changes only the GP classifier publication owner, its GP worker
return consumer, GP output packing, and a new native scalar summary helper.

`GPCFitPublication` carries each native likelihood through public member
publication. `GPCFitSummary` uses the existing HP_F64/HP_SUM reduction in the
same class order, then divides by the class count. The public `estimators_`,
member arrays/scalars and outer `n_iter_` retain their existing objects and
aliases. Binary fit retains its direct scalar/array aliases and allocates no
aggregate summary buffer. Worker transport still crosses interpreters as its
established public data; the native publication owner never crosses that
boundary. `_gpc_binary` remains live for worker-side query reconstruction.

GP tuple PACK now allocates the final exact tuple and transfers converted
member references into its C slots. The temporary Python list and its final
list-to-tuple conversion are removed. Public list outputs remain lists. Partial
outputs own completed slots on failure; input pins remain alive through the
last conversion. Existing Array view/exporter ownership is unchanged.

The source audit refreshes the original eight mechanism entries:

| Original mechanism | Current status |
| --- | --- |
| Mixed numeric label ordering/index | Pending; preserve exact NumPy/large-integer ordering and first-object rules |
| General label snapshot/flatten container | Pending |
| GPC likelihood publication/readback | Removed from local and worker-return composition |
| GPR score envelope | Removed in the preceding score tranche |
| GPC score envelope | Removed in the preceding score tranche |
| Inherited host argument rebind envelope | Pending; also part of the broader public-call admission capture |
| Tuple PACK staging list | Removed |
| Kernel constructor argument tuple | Pending |

Thus four of those eight are closed and four remain. This is not a complete
remaining-work denominator. The operation trace additionally identifies shared
score truth flatten/admission metadata and GPC archive kernel-unpack lists and
mode-validation dictionary as candidates. Public saved-kernel nodes, persisted
Arrays and reconstructed model/member fields remain required boundary state.
Direct inspection corrected the preliminary archive inference: the `rows`
helper returns an admitted Array, not a staging list. Both the preliminary
record and its correction are preserved. Shared ancestry/class metadata
admission remains explicitly uncertified by this bounded audit.

The removed-container count increases by **one**: the tuple staging list. The
GPC cleanup adds zero such rows: its public scalar fields remain required and
its reduction scratch was already a native List. The three native declarations
are not counted as three removed Python roles.

Evidence is under
`~/mojolearn-evidence/python-mojo-migration-20261110/gp-remainder-20261111/`,
including before snapshots, `before-audit.json`, `after-audit.json`, and
`archive-candidate-correction.json`. The separately counted role is in
`python-object-typed-gp-publication-tail.json` in the parent evidence directory.
The native summary closure has 13 local sources and zero Python imports;
audited numerical sources are unchanged.

Affected binding roots are `_mojolearn_runtime_family_host`,
`_mojolearn_runtime_parallel_host`, and `_mojolearn_runtime_trees_host`.
No provider ABI, public archive format, solver, kernel, switch or transfer route
changes. Existing tests remain; native fixture ownership stays with the test
lane. Compilation, runtime, measurement and identity are unrun for this source.
No A/B switch or box job was added.
