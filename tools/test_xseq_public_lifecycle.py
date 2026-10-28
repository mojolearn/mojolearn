"""Source guards for actual public recurrent/MLP lifecycle ownership."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_markers_exist_only_with_a_native_lifecycle_envelope():
    for filename,base in [('_x_sequence_mlp.py','_BaseMLP'),('_x_sequence_rnn.py','_RecurrentBase')]:
        tree=ast.parse((ROOT/'python/mojolearn'/filename).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==base)
        assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_mojolearn_native_public' for t in n.targets) and isinstance(n.value,ast.Constant) and n.value.value is True for n in cls.body)
    source=(ROOT/'bindings/xseq_native.mojo').read_text()
    for required in ('XseqPublicLifecycle(model, operation, args, context)','estimator_fit_begin(py_fields(model))','finally:\n                    estimator_fit_end(model, depth)','lifecycle.finish(model, result, context)','NotFittedError:','active_not_fitted_class'):
        assert required in source
    assert 'xseq_rnn(model, "rnn_predict", predict_args, context)' in source
    assert 'xseq_mlp(model, "mlp_predict", predict_args, context)' in source


def test_validation_and_output_semantics_are_native_not_removed():
    source=(ROOT/'providers/python/xseq_public_lifecycle_pyapi.mojo').read_text()
    for required in ('classical_require_fitted(model','classical_refuse_sparse(self.source','Complex data not supported','Expected 2D array','Found array with 0','_finite(x, "X"','_finite(y, "y"','lda_native_labels(','classical_warn_column(','inconsistent numbers of samples','classical_check_width(','estimator_feature_check(','estimator_fit_complete(','classical_widen_f32_f64(','cannot be fit on n_samples=1','cannot be fit on n_features=1'):
        assert required in source,required
    assert 'py_call(method' not in source and '_sk_validate' not in source.replace('_sk_validate.reraise','former validation error mapping')
    neural=(ROOT/'bindings/xseq_neural_native.mojo').read_text()
    assert 'xseq_take(fields["classes_"],' not in neural
    assert neural.count('xseq_take(xseq_public_labels(fields, context),')==3


def test_common_feature_and_caller_boundary_are_reused():
    forest=(ROOT/'bindings/trees_public_lifecycle_native.mojo').read_text()
    assert 'from providers.python.estimator_lifecycle_pyapi import' in forest
    assert 'def _outside(' not in forest and 'def _feature_check(' not in forest
    bridge=(ROOT/'python/mojolearn/_native_xseq_bridge.py').read_text()
    assert 'xseq_lifecycle_context=lifecycle' in bridge
    assert 'classifier_labels_namespace=dict(vars(ClassLabels))' in bridge
    assert '"all_finite_f64"' in bridge


def test_finite_validation_walks_admitted_bytes_without_packed_dispatch():
    source = (ROOT/'providers/python/xseq_public_lifecycle_pyapi.mojo').read_text()
    finite = source.split('def _finite(', 1)[1].split('\n\n\nstruct ', 1)[0]
    assert 'host_inference_raw_finite(' in finite
    assert 'xseq_base(' not in finite
    assert 'py_call(' not in finite
    assert 'byte_strides = view.strides' in finite
