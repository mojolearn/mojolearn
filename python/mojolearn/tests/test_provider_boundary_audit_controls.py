# SPDX-License-Identifier: Apache-2.0
"""Focused public admission controls from the 2026-10-27 source audit.

These are runtime tests, not source qualifications. The audit lane only
syntax-checks this file; an owner with the affected compiled bindings runs it.
The hook counters are independent of profiler visibility.
"""
import pytest


def test_public_import_tuple_never_enters_private_graph_descriptor_dispatch():
    from mojolearn import import_model

    calls = []

    class Operation:
        def __str__(self):
            calls.append('str')
            return 'not-a-private-operation'

    with pytest.raises(TypeError):
        import_model((Operation(), {}))
    assert calls == []


@pytest.mark.parametrize('source', [(), ('not-a-private-operation', {})])
def test_public_import_rejects_tuple_as_artifact(source):
    from mojolearn import import_model

    with pytest.raises(TypeError):
        import_model(source)


@pytest.fixture
def artifact_program(tmp_path):
    from mojolearn import import_model
    from mojolearn.tests import _onnx_fixture as fx

    path = fx.make_model(str(tmp_path / 'relu.onnx'),
                         [fx.node('Relu', ['x'], ['y'])],
                         [('x', 'f32', (2, 3))], [('y', 'f32', (2, 3))])
    return import_model(path, numeric_mode='identical')


@pytest.mark.parametrize('field', [
    'inputs', 'outputs', 'input_shape', 'output_shape', 'output_dtype',
    'program_id', 'session_id', 'program_layout', 'session_layout',
])
def test_public_run_torch_refuses_metadata_hooks_before_consuming_them(artifact_program, field):
    import torch

    program = artifact_program
    tensor = torch.ones((2, 3), dtype=torch.float32)
    calls = []

    class Hook:
        def __len__(self):
            calls.append('len')
            return 1

        def __getitem__(self, index):
            calls.append('getitem')
            return ('x', (2, 3), 'float32')

        def __str__(self):
            calls.append('str')
            return 'float32'

        def __int__(self):
            calls.append('int')
            return 2

        def __index__(self):
            calls.append('index')
            return 2

    fields = vars(program)
    target = program
    original_class = type(program)
    original_value = None
    changed_field = None
    if field in ('inputs', 'outputs'):
        changed_field = field
        original_value = fields[field]
        fields[field] = Hook()
    elif field in ('input_shape', 'output_shape', 'output_dtype'):
        changed_field = 'inputs' if field == 'input_shape' else 'outputs'
        original_value = fields[changed_field]
        name, shape, dtype = original_value[0]
        replacement = (name, shape, Hook()) if field == 'output_dtype' else (name, (Hook(), 3), dtype)
        fields[changed_field] = [replacement, *original_value[1:]]
    elif field in ('program_id', 'session_id'):
        if field == 'session_id':
            target = fields['_s']
            fields = vars(target)
        changed_field = '_id'
        original_value = fields['_id']
        fields['_id'] = Hook()
    else:
        if field == 'session_layout':
            target = fields['_s']
        original_class = type(target)

        class HookLayout(original_class):
            __slots__ = ()

            @property
            def __dict__(self):
                calls.append('dict')
                return {}

        # Test-only mutation: exercise post-construction class/layout admission.
        target.__class__ = HookLayout

    try:
        with pytest.raises(TypeError):
            program.run_torch(tensor)
        assert calls == []
    finally:
        # Restore owner metadata even if an earlier assertion failed, so ordinary
        # session lifetime cleanup does not observe a synthetic invalid handle.
        if changed_field is not None:
            fields[changed_field] = original_value
        else:
            target.__class__ = original_class


def test_public_run_torch_accepts_ordinary_artifact_and_tensor(artifact_program):
    import torch

    value = torch.tensor([[-2., -1., 0.], [1., 2., 3.]], dtype=torch.float32)
    result = artifact_program.run_torch(value)
    # This is a host adapter smoke, not cross-vendor arithmetic evidence.
    assert result.device == value.device
    assert torch.equal(result, torch.tensor([[0., 0., 0.], [1., 2., 3.]]))


@pytest.mark.parametrize('method,dispatch_field', [('run', '_run_native'), ('run_torch', '_run_torch_native')])
def test_public_execution_never_dispatches_through_mutable_instance_callable(artifact_program, method, dispatch_field):
    import numpy as np
    import torch

    program = artifact_program
    operation = getattr(program, method)
    value = torch.ones((2, 3)) if method == 'run_torch' else np.ones((2, 3), dtype=np.float32)
    calls = []

    def callback(*args):
        calls.append('dispatch')

    fields = vars(program)
    original = fields[dispatch_field]
    fields[dispatch_field] = callback
    try:
        try:
            operation(value)
        except TypeError:
            pass  # Refuse the changed entry, or ignore it via fixed native dispatch.
        assert calls == []
    finally:
        fields[dispatch_field] = original


@pytest.mark.parametrize('method', ['run', 'run_torch'])
def test_bound_public_execution_admits_layout_before_instance_lookup(artifact_program, method):
    import numpy as np
    import torch

    program = artifact_program
    # Resolve the method before test-only mutation: only product wrapper work is
    # inside the operation, not the application's initial attribute lookup.
    operation = getattr(program, method)
    value = torch.ones((2, 3)) if method == 'run_torch' else np.ones((2, 3), dtype=np.float32)
    calls = []
    original = type(program)

    class HookLayout(original):
        __slots__ = ()

        def __getattribute__(self, name):
            calls.append(name)
            return object.__getattribute__(self, name)

    program.__class__ = HookLayout
    try:
        with pytest.raises(TypeError):
            operation(value)
        assert calls == []
    finally:
        object.__setattr__(program, '__class__', original)
