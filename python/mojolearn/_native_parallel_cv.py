# SPDX-License-Identifier: Apache-2.0
"""Import-only connection of native parallel-CV process and schema metadata."""


def connect(context, package):
    # glue: fixed native worker/bootstrap interfaces and already-connected owners
    import io
    import os
    import sys
    from types import BuiltinFunctionType, MethodDescriptorType, ModuleType
    from . import _backend, _gbdt_adapters, ensemble

    classes = {}
    class_ids = {}
    for cls in context['family_msel_estimators']:  # glue: stable identifiers for exact native constructor schemas
        key = cls.__module__ + ':' + cls.__qualname__
        if key in classes:
            raise TypeError('duplicate native CV estimator schema identifier')
        classes[key] = cls
        class_ids[cls] = key
    metrics = {}
    metric_ids = {}
    for function in context['family_msel_metric_plans']:  # glue: native metric identifiers, not importable callbacks
        key = function.__module__ + ':' + function.__qualname__
        if key in metrics:
            raise TypeError('duplicate native CV metric schema identifier')
        metrics[key] = function
        metric_ids[function] = key
    filename = None
    binding_vendor = binding_mode = None
    try:
        binding = _backend.binding('_mojolearn_gbdt', 'identical')
        if type(binding) is _backend._HostBinding:
            binding = _backend.load_host_module(binding._HostBinding__basename)  # cpu-route: import-only native owner connection
        if type(binding) is ModuleType:
            filename = binding.__file__
            binding_vendor = binding.gbdt_vendor
            binding_mode = binding.gbdt_numeric_mode
    except (ImportError, AttributeError, ValueError, RuntimeError):
        pass
    if type(io.BytesIO.getvalue) is not MethodDescriptorType:
        raise TypeError('native CV captures require the native BytesIO owner')
    artifacts = {}
    for name, entry in (('cv', package.model_selection._NATIVE), ('trees', _gbdt_adapters._TREES),
                        ('archive', ensemble._ARCHIVE)):  # glue: actual native owner files, including separate development DSOs
        if type(entry) is BuiltinFunctionType:
            owner = entry.__self__
            if type(owner) is not ModuleType:
                owner = sys.modules.get(entry.__module__)
            if type(owner) is ModuleType:
                artifacts[name] = owner.__file__
    if filename is not None:
        artifacts['gbdt'] = filename
    context['family_msel_parallel'] = dict(
        executable=sys.executable, environ=os.environ._data, vendor=_backend.vendor(),
        cpu_only=_backend._CPU_ONLY is not None, classes=classes, class_ids=class_ids,
        metrics=metrics, metric_ids=metric_ids, bytesio_type=io.BytesIO,
        bytesio_getvalue=io.BytesIO.getvalue, bytesio_seek=io.BytesIO.seek, bytesio_close=io.BytesIO.close,
        exception_args=BaseException.args.__get__, last_run=None,
        capture_classes=(_gbdt_adapters.GradientBoostingRegressor, _gbdt_adapters.GradientBoostingClassifier),
        gbdt_class=ensemble.GradientBoosting, gbdt_entry=_gbdt_adapters._TREES,
        gbdt_context=_gbdt_adapters._TREES_CONTEXT,
        archive_entry=ensemble._ARCHIVE, archive_context=ensemble._ARCHIVE_CONTEXT,
        binding_path=filename, binding_sha256=None,
        binding_vendor=binding_vendor, binding_mode=binding_mode,
        artifact_paths=artifacts, artifact_hashes=None,
        selected_arch=_backend.gpu_arch(), selected_arch_how=_backend.gpu_arch_how())
    entry = package.model_selection._NATIVE
    if type(entry) is BuiltinFunctionType:
        entry(None, 'family_msel_parallel_initialize', (), context)
