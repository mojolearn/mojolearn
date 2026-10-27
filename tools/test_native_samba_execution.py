"""Source contracts for native Samba traversal; runtime is explicitly pending."""
from pathlib import Path
from check_generation_native_boundary import audit
ROOT=Path(__file__).resolve().parents[1]


def test_native_execution_and_extended_primitive_closures_are_python_free():
    _, failures=audit(ROOT,('training.native_samba_execution','training.native_primitive_dispatch'))
    assert not failures, failures


def test_native_layer_orders_error_refresh_compute_and_writeback():
    source=(ROOT/'training/native_samba_execution.mojo').read_text()
    layer=source.split('def forward(mut self, input_address')[1].split('struct NativeSambaModel')[0]
    assert layer.index('self.failure.byte_length()') < layer.index('copy_to(')
    assert layer.index('copy_to(') < layer.index('self.result = self.endpoint.call')
    assert layer.index('self.result = self.endpoint.call') < layer.index('copy_from(')
    assert layer.index('copy_from(') < layer.index('self.completed = True')
    assert 'self.native' not in source  # executor owns plain native fields


def test_public_boundary_uses_native_model_and_preserves_partial_publication():
    source=(ROOT/'providers/python/samba_execution_pyapi.mojo').read_text()
    assert 'AdmittedNativeExecution' in source
    assert 'layer.failure = String(problem)' in source
    assert 'except problem:\n            self.publish()' in source
    assert 'if not layer.completed:' in source
    assert 'if not layer.visited:' in source
    assert 'self.native.forward(' in source
    assert 'layer.refresh_layouts.append(pin.layout.value().copy())' in source
    public=(ROOT/'bindings/samba_inference_native.mojo').read_text()
    assert 'samba_provider_forward(fields, config, entries, ids, state, step' in public


def test_native_backward_walk_preserves_saved_tapes_and_original_loss_routes():
    source=(ROOT/'training/native_samba_execution.mojo').read_text()
    assert 'for reverse in range(len(self.layers))' in source
    assert 'self.close_tapes()' in source
    assert 'self.backward_failure.byte_length()' in source
    assert 'self.validate_gradient and all_finite_f32' in source
    for operation in (6,7,8,9,11):
        assert f'self.primitive.call({operation},' in source
    assert 'requested_items: Int, automatic_items: Bool' in source
    boundary=(ROOT/'providers/python/samba_execution_pyapi.mojo').read_text()
    assert 'layer.backward_failure = String(problem)' in boundary
    assert 'layer.tape = True' in boundary
    assert 'publication.publish_reports = False' in boundary
    assert 'neural_descriptor_construct(publication.provider' in boundary


def test_resident_lifecycle_uses_shared_owner_and_validates_before_reserving():
    native=(ROOT/'training/native_resident_dispatch.mojo').read_text()
    for function in ('samba_resident_forward', 'samba_resident_train_step',
                     'samba_resident_accumulated_step', 'resident_open',
                     'resident_upload', 'resident_download', 'resident_close'):
        assert f'return {function}(' in native
    assert 'owner[].busy = True' in native and 'owner[].busy = False' in native
    boundary=(ROOT/'providers/python/samba_resident_pyapi.mojo').read_text()
    assert 'neural_descriptor_adopt(provider, handle, 13, [12])' in boundary
    step=boundary.split('def samba_resident_native_step(')[1]
    assert step.index('neural_descriptor_execute(provider, 19,') < step.index('neural_descriptor_reserved_execute(')
    assert '[Int(py=opt["_res"]), Int(py=fields["n_total"])], [], False)' in step
    assert 'samba_resident_native_forward(fields, config, entries, ids)' in (ROOT/'bindings/samba_inference_native.mojo').read_text()


def test_native_lock_is_an_exact_samba_admission_owner():
    import ast
    source=(ROOT/'python/mojolearn/_native_samba_bridge.py').read_text()
    ast.parse(source)
    assert '"opaque_types": (_thread.LockType, type(_thread.RLock()), _OperationLock)' in source
    assert 'classes = tuple(dict.fromkeys(classes))' in source
    assert 'classes + (array_cls,)' in source


def test_optimizer_dispatch_preserves_same_native_functions_and_float_conversions():
    source=(ROOT/'training/native_resident_dispatch.mojo').read_text()
    for operation, function in ((25,'per_call_step'),(26,'resident_step'),(27,'resident_step_io')):
        branch=source.split(f'    if op == {operation}:')[1].split('    if op ==')[0]
        assert f'return {function}(' in branch
        for index in range(8):
            assert f'Float32(f[{index}])' in branch
    step=(ROOT/'bindings/samba_step_native.mojo').read_text()
    assert 'microbatch_split_is_identical(b * length, a)' in step
    for operation in (25,26,27):
        assert f'samba_optimizer_execute(provider, {operation},' in step
    assert 'neural_descriptor_execute(provider, 9,' in step


def test_accumulation_reuses_admitted_native_metadata_and_keeps_original_reduction():
    native=(ROOT/'training/native_samba_execution.mojo').read_text()
    loop=native.split('def accumulated_gradients(')[1]
    assert 'for k in range(accumulation)' in loop
    assert 'losses[k] = Float32(self.gradients(' in loop
    assert 'src=i32_ptr(ids) + k * self.rows' in loop
    assert '[n, accumulation, self.rows * accumulation]' in loop
    assert '[1, accumulation, -1]' in loop
    assert 'registry_failure.byte_length()' in loop
    assert 'self.tape and self.tape_started' in native
    assert 'self.close_endpoint.call(self.reset_operation' in native
    for file in ('mamba/native_tape_dispatch.mojo','transformer/native_tape_dispatch.mojo'):
        source=(ROOT/file).read_text()
        assert 'if op == 5:' in source
        assert 'or not tape[].consumed:' in source
    boundary=(ROOT/'providers/python/samba_execution_pyapi.mojo').read_text().split('def samba_provider_accumulated_gradients(')[1]
    assert boundary.count('AdmittedSambaExecution(') == 1
    assert 'admitted.native.accumulated_gradients(' in boundary
    assert 'for k in range(accumulation)' not in boundary
