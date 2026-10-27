#!/usr/bin/env python3
"""Public ONNX behavior probe; root executes once with its new Mac build."""
import argparse
import json
from pathlib import Path
import tempfile
import traceback


def run():
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    import mojolearn
    from mojolearn import ops
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn.importers import _program
    x = np.array([[-2, 3, 4], [5, -6, 7]], dtype='f4')
    source = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    destination = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph([helper.make_node('Relu', ['x'], ['y'])], 'native-import', [source], [destination])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'model.onnx'
        onnx.save(model, path)
        audit = audit_native_call(mojolearn.import_model, str(path), numeric_mode='identical')
        imported = audit.result
        actual = np.asarray(imported.run(x))
        assert actual.tobytes() == np.maximum(x, 0).tobytes()
        # Old internal Python/extension call doors must no longer execute. The
        # public source-kind dispatcher remains recorded debt in audit.report().
        called = []
        def old_door(*args):
            called.append(args)
            raise AssertionError('old inner compiled-call dispatch reached')
        entries = _program._IMPORT_CONTEXT['modes']['identical']
        previous = {name: entries[name] for name in ('onnx_import_file', 'onnx_import', 'onnx_buffers')}
        try:
            entries.update(dict.fromkeys(previous, old_door))
            again = mojolearn.import_model(str(path), numeric_mode='identical')
            assert np.asarray(again.run(x)).tobytes() == actual.tobytes()
            missing = Path(folder) / 'missing.onnx'
            try:
                mojolearn.import_model(str(missing), numeric_mode='identical')
            except FileNotFoundError as error:
                assert error.errno == 2 and str(error.filename) == str(missing)
            else:
                raise AssertionError('missing file accepted')
        finally:
            entries.update(previous)
        assert not called
    return {'status': 'selected_behavior_passed_public_dispatch_pending', 'runtime_audit': audit.report(),
            'pending': ['Public Python source-kind dispatcher and arbitrary Torch/ModelProto export',
                        'Reader-object source route not covered by this probe']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run()
    except BaseException:
        result = {'status': 'failed', 'traceback': traceback.format_exc()}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'evidence': str(args.output)}))
    return result['status'] == 'failed'


if __name__ == '__main__':
    raise SystemExit(main())
