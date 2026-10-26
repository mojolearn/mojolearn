"""Inert fixtures for the 2026-10-24 reach rules of tools/verify_coverage_audit.py:
native-owner dispatch, constant probes, valueless definitions, the Python-shell
rule and the whole-surface sabotage column. Static: nothing is built or run."""
import ast
from collections import defaultdict

import verify_coverage_audit as audit_tool


class _Graph:
    """A definition graph with no edges: a branch names only its own literals."""

    def resolve(self, path, ident):
        return ()

    def text(self, node):
        return ""

    def idents(self, node):
        return frozenset()

    def targets(self, node):
        return ()

    def units(self, path):
        return {}


def _dispatch(exports, records=(), generic=frozenset(), owner_files=None):
    od = object.__new__(audit_tool.OwnerDispatch)
    od.exports = frozenset(exports)
    od.graph = _Graph()
    od._memo, od._prefix_memo, od.op_values = {}, {}, set()
    od.records = list(records)
    od.generic = frozenset(generic)
    od.owner_files = defaultdict(set, owner_files or {})
    od.shared_files = frozenset()
    return od


OWNER = '''def handle(operation: String, family: String) raises -> PythonObject:
    if family == "FastICA":
        if operation == "fit":
            return ica_fit(entries["x_decomp_ica_fit"])
        elif operation == "transform":
            return project(entries["x_decomp_ica_project"])
    if operation == "croston_fit" or operation == "croston_refit":
        _ = py_call_typed(entries["croston"], args)
    else:
        _ = py_call_typed(entries["unrelated"], args)
'''


def test_branches_read_nested_conditions_and_alternatives():
    od = _dispatch({"x_decomp_ica_fit", "x_decomp_ica_project", "croston", "unrelated"})
    got = {conds: ents for conds, ents, _ in od._branches("bindings/x_native.mojo", OWNER)}
    assert got[(frozenset({"FastICA"}), frozenset({"fit"}))] == {"x_decomp_ica_fit"}
    assert got[(frozenset({"FastICA"}), frozenset({"transform"}))] == {"x_decomp_ica_project"}
    assert got[(frozenset({"croston_fit", "croston_refit"}),)] == {"croston"}
    # an else branch holds no positive condition, so it credits no lane
    assert all("unrelated" not in ents for ents in got.values())


def test_composed_names_resolve_against_exports():
    od = _dispatch({"xn_kernel", "xn_colsum", "other"})
    assert od._named({"kernel", "colsum", "rowsum"}, {"xn_"}) == {"xn_kernel", "xn_colsum"}


def test_reached_needs_a_specific_value_or_the_owner_entry():
    records = [((frozenset({"FastICA"}), frozenset({"fit"})), frozenset({"x_decomp_ica_fit"}),
                "bindings/decomp_native.mojo", frozenset()),
               ((frozenset({"fit"}),), frozenset({"x_decomp_ipca_fit"}), "bindings/ipca_native.mojo",
                frozenset())]
    od = _dispatch(set(), records, generic={"fit"},
                   owner_files={"ipca_native_call": {"bindings/ipca_native.mojo"}})
    ents, files = od.reached(frozenset({"FastICA", "fit"}), frozenset())
    assert ents == {"x_decomp_ica_fit"} and files == {"bindings/decomp_native.mojo"}
    # `fit` alone credits the IncrementalPCA branch only through its owner entry
    ents, _ = od.reached(frozenset({"fit"}), frozenset({"ipca_native_call"}))
    assert ents == {"x_decomp_ipca_fit"}
    ents, _ = od.reached(frozenset({"fit"}), frozenset())
    assert ents == set()


def test_reached_relays_an_owner_operation_named_by_a_branch():
    records = [((frozenset({"family_msel_split"}),), frozenset(), "bindings/model_selection_native.mojo",
                frozenset({"family_msel_split_loo_at"})),
               ((frozenset({"family_msel_split_loo_at"}),), frozenset({"leave_range_i64"}),
                "bindings/model_selection_native.mojo", frozenset())]
    od = _dispatch(set(), records)
    ents, _ = od.reached(frozenset({"family_msel_split"}), frozenset())
    assert ents == {"leave_range_i64"}


def test_constant_probe():
    assert audit_tool.constant_probe('def a_binding() -> PythonObject:\n    """Version."""\n'
                                     '    return PythonObject(1)\n')
    assert audit_tool.constant_probe("def v() -> PythonObject:\n    return PythonObject(String(COMPILED_VENDOR))\n")
    assert not audit_tool.constant_probe("def f(x: PythonObject) -> PythonObject:\n"
                                         "    var n = Int(py=x)\n    return PythonObject(n)\n")


def test_valueless_definitions():
    assert audit_tool._valueless('comptime _DEVCTX_SLOT = "MojoCoreContextIdentical"')
    assert audit_tool._valueless("@fieldwise_init\nstruct Handle(Movable):\n    var id: Int")
    assert audit_tool._valueless("def __init__(out self):\n    pass", ("m", "f.mojo", "Store", "__init__"))
    assert not audit_tool._valueless("def gather(x: Int) -> Int:\n    return x + 1", ("t", "f.mojo", "gather"))


def _method(src):
    return ast.parse(src).body[0]


def test_trivial_method():
    assert audit_tool.trivial_method(_method('def fit(self, X, y=None):\n    """Doc."""\n    return self\n'))
    assert audit_tool.trivial_method(_method("def close(self):\n    pass\n"))
    assert not audit_tool.trivial_method(_method("def fit(self, X):\n    return _CALL(self, 'fit', X)\n"))


def test_python_shell_rule():
    native = {"x_cnn_gemm": ("covered", ["x-cnn-conv2d"]), "gap_entry": ("GAP", [])}
    row = dict(public="mojolearn.Conv2d", op="forward", lanes=["x-cnn-conv2d"], entries=["x_cnn_gemm"])
    assert audit_tool.py_triage(row, native)[0] == "covered"
    row = dict(public="mojolearn.Conv2d", op="backward", lanes=[], entries=["x_cnn_gemm"])
    st, lanes, _ = audit_tool.py_triage(row, native)
    assert st == "covered (via native)" and lanes == ["x-cnn-conv2d"]
    row = dict(public="mojolearn.MaxPool2d", op="fit", lanes=[], entries=[], trivial=True)
    assert audit_tool.py_triage(row, native)[0] == "n/a"
    row = dict(public="mojolearn.Thing", op="fit", lanes=[], entries=["gap_entry"])
    assert audit_tool.py_triage(row, native)[0] == "GAP"


def _row(entry, **kw):
    row = dict(entry=entry, sabotage_defines=[], sabotage_patches=[], changed_kernel=[], n_changed=0,
               closure_kernel=[], n_closure_kernel=0)
    row.update(kw)
    return row


def test_whole_surface_sabotage_column():
    assert audit_tool.sabotage_of(_row("kmeans_centers_release"))[0] == "n/a"
    assert audit_tool.sabotage_of(_row("door_only"))[0] == "n/a"
    st, why = audit_tool.sabotage_of(_row("kernel_entry", closure_kernel=["metrics/x.mojo:score"],
                                          n_closure_kernel=3))
    assert st == "owed" and "metrics/x.mojo:score +2" in why
    assert audit_tool.sabotage_of(_row("kernel_entry", sabotage_defines=["MOJOLEARN_HOST_SABOTAGE"]))[0] == "armed"


def _native_row(entry, **kw):
    row = dict(entry=entry, src="bindings/_mojolearn_x_sequence.mojo", lanes_precise=[], py_files=0,
               constant_probe=False)
    row.update(kw)
    return row


def test_adapter_only_entry_is_not_a_gap():
    st, lanes, why = audit_tool.triage(_native_row("adapter_entry", adapter_files=1))
    assert st == "n/a" and lanes == [] and "adapter" in why
    assert audit_tool.triage(_native_row("package_entry", py_files=1, adapter_files=1))[0] == "GAP"
    assert audit_tool.triage(_native_row("lane_entry", py_files=1, lanes_precise=["x"]))[0] == "covered"
    # an audit written before the adapter count existed keeps its old reading
    assert "no Python caller" in audit_tool.triage(_native_row("old_row"))[2]


def test_import_installed_entries_name_their_lanes():
    for entry, src in (("msel_rebase_offsets_i32", "bindings/_mojolearn.mojo"),
                       ("msel_rebase_offsets_i32", "bindings/_mojolearn_core_host.mojo"),
                       ("shard_topk_merge_f32", "bindings/_mojolearn.mojo"),
                       ("shard_topk_merge_f32", "bindings/_mojolearn_core_host.mojo"),
                       ("declared_params_native_call", "bindings/_mojolearn_tokenizer_host.mojo"),
                       ("linear_constructor_native_call", "bindings/_mojolearn_tokenizer_host.mojo"),
                       ("full_reset_constructor_native_call", "bindings/_mojolearn_tokenizer_host.mojo"),
                       ("array_native_call", "bindings/_mojolearn_tokenizer_host.mojo"),
                       ("params_setter_native_call", "bindings/_mojolearn_tokenizer_host.mojo"),
                       ("public_attribute_native_call", "bindings/_mojolearn_tokenizer_host.mojo")):
        st, lanes, why = audit_tool.triage(_native_row(entry, src=src, py_files=1))
        assert st == "covered" and lanes and why, (entry, src)
    assert "public-set-params" in audit_tool.NEW_LANES


def test_dead_and_probe_entries_read_sabotage_na():
    # lane coverage-sabotage-arms-20261025: the 0.8.57 owed sabotage rows; lane
    # dead-native-entries-0859 deleted the dead ones, the two reach probes remain
    for entry in ("km_rbf_staged", "x_decomp_mcd_cov_reach"):
        st, why = audit_tool.sabotage_of(_row(entry, changed_kernel=["core/x.mojo:k"], n_changed=1))
        assert st == "n/a" and why, entry
    # a reaching arm still wins over the table
    st, _ = audit_tool.sabotage_of(_row("km_rbf_staged", changed_kernel=["core/x.mojo:k"], n_changed=1,
                                        sabotage_patches=["kernel_methods/checks/sabotage/coverage_x.patch"]))
    assert st == "armed"


def test_deleted_dead_entries_leave_the_table():
    # lane dead-native-entries-0859: a deleted entry has no table row (docs/TOMBSTONES.md)
    for entry in ("adam_hyper_f64", "lr_schedule_fill", "byte_lm_launch_probe", "forest_host_gbdt_sigmoid",
                  "gbdt_sigmoid", "gp_log64", "x_decomp_knn_select", "x_neighbors_purity_flags",
                  "sched_exp_block"):
        assert entry not in audit_tool.SABOTAGE_NA_ENTRIES, entry
