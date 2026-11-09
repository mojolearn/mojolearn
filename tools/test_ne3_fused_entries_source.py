"""Source contract of lane ne3-g-svc-gp-neighbors's auxiliary x_neighbors
operations and fused doors; source-only, no runtime claim."""
import re
from pathlib import Path
from gen_x_neighbors_provider import EXTRA, generated

ROOT = Path(__file__).resolve().parents[1]
NEW = [('ocsvm_labels', (4, 6, 5), 5, 'x_neighbors_ocsvm_labels', 4),
       ('nc_labels', (5, 4, 0), 6, 'x_neighbors_nc_labels', 5),
       ('nc_proba', (5, 4, 0), 7, 'x_neighbors_nc_proba', 5),
       ('lp_dense_proba', (5, 6, 2), 8, 'x_neighbors_lp_dense_proba', 6)]


def _read(path):
    return (ROOT / path).read_text()


def test_extra_operations_are_appended_and_generated():
    names = [name for name, _ in EXTRA]
    assert names[:5] == ['ocsvm_alpha_init', 'unit_ff', 'ocsvm_resident', 'capabilities', 'pcs_draw']
    for name, counts, aux, _, _ in NEW:
        assert (name, counts) in EXTRA
        assert names.index(name) == aux
    for path, expected in generated(ROOT).items():
        assert _read(path) == expected, path
    schema = _read('x_neighbors/descriptor_schema.mojo')
    assert 'comptime XN_OPERATION_COUNT = 87' in schema
    for name, counts, aux, _, _ in NEW:
        assert f'if name == "{name}":\n        return {78 + aux}\n' in schema


def test_aux_operations_exist_on_both_targets_with_capability_bits():
    for target in ('device', 'host'):
        aux = _read(f'x_neighbors/native_descriptor_aux_{target}.mojo')
        for name, _, op, _, bit in NEW:
            assert re.search(rf'\bop == {op}\b', aux), (target, name)
        assert '<< 4)' in aux and '<< 5)' in aux and '<< 6)' in aux


def test_exports_registered_on_both_bindings_and_listed():
    surface = _read('python/mojolearn/_surface_neighbors.py')
    execute = _read('providers/python/x_neighbors_descriptor_pyapi.mojo')
    for suffix in ('', '_host'):
        binding = _read(f'bindings/_mojolearn_x_neighbors{suffix}.mojo')
        for name, _, _, export, _ in NEW:
            assert f'("{export}")' in binding, (suffix, export)
    for name, _, _, export, _ in NEW:
        assert f'"{export}"' in surface
        assert f'name == "{name}"' in execute


def test_fused_doors_keep_the_op_chain_fallback():
    ocsvm = _read('bindings/svc_ocsvm_native.mojo')
    assert '_ocsvm_labels_available(binding)' in ocsvm and '"p2m_sign_label"' in ocsvm
    local = _read('bindings/neighbors_local_native.mojo')
    assert '_nc_fused_available(binding)' in local and '"row_argmax"' in local and '"knn_select"' in local
    semi = _read('bindings/neighbors_semi_native.mojo')
    assert '_lp_dense_available(binding)' in semi and '"row_normalize"' in semi
    cluster = _read('bindings/mcls_cluster.mojo')
    assert '"homogeneity_completeness_v_measure", False' in cluster and '"v_measure_score"' in cluster
    for path in ('bindings/_mojolearn_metrics.mojo', 'bindings/_mojolearn_metrics_host.mojo'):
        assert '("homogeneity_completeness_v_measure")' in _read(path)


def test_v_measure_has_one_composer():
    for path in ('metrics/impl/stats/detail/v_measure.mojo', 'metrics/host/metrics_oracle.mojo',
                 'metrics/estimator.mojo', 'bindings/_mojolearn_metrics_host.mojo'):
        text = _read(path)
        assert 'v_measure_from' in text, path
        assert '(1.0 + beta) *' not in text, path


def test_svc_score_compares_codes():
    svc = _read('bindings/svc_kernel_native.mojo')
    body = svc[svc.index('if operation == "family_svc_svc_score":'):]
    body = body[:body.index('if operation == "family_svc_svr_fit":')]
    assert '_svc_predict_codes(' in body and 'svc_decode' not in body


def test_host_lp_resident_loop_runs_the_door_ops():
    host = _read('x_neighbors/iter_host.mojo')
    start = host.index('def op_lp_iterate_knn(')
    loop = host[start:host.index('\n\n\n', start)]
    assert '_host_absdiff_sum(' in loop and 'op_lp_knn_product(' in loop
    assert '_host_lp_clamp(' in loop and '_host_ls_clamp(' in loop
    assert 'absdiff_sum_item(' not in loop
    assert 'def lp_fast_resident(' not in host
