# SPDX-License-Identifier: Apache-2.0
# Native owner qualification added 2026-10-21. These are new lanes: no existing
# lane, fixture or accepted digest is revised. Execution, fault restoration,
# installed-wheel admission and cross-vendor references remain separately owed.
# Python here is the independent verification controller, never product code.


@lane("native-schedule-series")
def _(ml, X, yc, yr, Xh=None):
    """Consume both series APIs, including warmup/phase edges and empty series."""
    T = ml.training
    schedules = [T.ConstantLR(.003, warmup_steps=3),
                 T.WarmupLinearLR(.003, warmup_steps=3, total_steps=12, min_lr=.0001),
                 T.WarmupCosineLR(.003, warmup_steps=3, total_steps=12, min_lr=.0001),
                 ml.StepLR(.003, step_size=3, gamma=.8), ml.ExponentialLR(.003, gamma=.8),
                 ml.OneCycleLR(.003, total_steps=12),
                 ml.OneCycleLR(.003, total_steps=12, anneal_strategy="linear", three_phase=True)]
    parts = {}
    for i, schedule in enumerate(schedules):
        for count in (0, 1, 3, 4, 12):
            values = np.asarray(schedule.lr_values(count), dtype=np.float32)
            parts[f"values-{i}-{count}"] = _h(values)
            if i >= 3:
                parts[f"list-{i}-{count}"] = _h(np.asarray(schedule.lrs(count), dtype=np.float32))
    return _fit(parts)


@lane("optim-owner-lifecycle")
def _(ml, X, yc, yr, Xh=None):
    """Replace both moments, resume all three bare owners, and consume +0 bytes."""
    T = ml.training
    parts = {}
    for name in ("SGD", "Adam", "AdamW"):
        cls = getattr(T, name)
        options = dict(lr=.003, numeric_mode="identical")
        if name == "SGD":
            options.update(momentum=.75, dampening=.25)
        params = [np.ascontiguousarray(X[:5, :3], dtype=np.float32).copy(),
                  np.ascontiguousarray(X[5:9, 0], dtype=np.float32).copy()]
        grads = [np.ascontiguousarray(X[10:15, :3], dtype=np.float32).copy(),
                 np.ascontiguousarray(X[15:19, 0], dtype=np.float32).copy()]
        owner = cls(params, **options)
        owner.step(grads)
        n = sum(p.size for p in params)
        mean = np.linspace(-.125, .25, n, dtype=np.float32)
        variance = np.linspace(.125, .5, n, dtype=np.float32)
        owner.exp_avg = mean
        owner.exp_avg_sq = variance
        saved = owner.state_dict()
        parts[name + "-set"] = _h(np.asarray(saved["exp_avg"]), np.asarray(saved["exp_avg_sq"]))
        resumed_params = [p.copy() for p in params]
        resumed = cls(resumed_params, **options)
        resumed.load_state_dict(saved)
        restored = resumed.state_dict()
        parts[name + "-load"] = _h(np.asarray([restored["t"]], dtype=np.int64),
                                  np.asarray(restored["exp_avg"]), np.asarray(restored["exp_avg_sq"]),
                                  np.asarray(restored["buf_initialized"]))
        resumed.step([g.copy() for g in grads])
        owner.step([g.copy() for g in grads])
        parts[name + "-continued"] = _h(*params)
        parts[name + "-resumed"] = _h(*resumed_params)
        # Signed zero, NaN and infinity must all become positive zero bytes.
        zeroed = [np.asarray([np.nan, np.inf, -np.inf, -0., 1., -2.], dtype=np.float32),
                  np.ascontiguousarray(grads[0])]
        resumed.zero_grad(zeroed)
        parts[name + "-zero"] = _h(*zeroed)
    return _fit(parts)


@lane("optim-io-capability")
def _(ml, X, yc, yr, Xh=None):
    """Host IO query and native transfer refusal; the same disabled-owner contract on GPU."""
    owner = ml.training.AdamW(np.ascontiguousarray(X[:3, :2], dtype=np.float32).copy(),
                              numeric_mode="identical", resident=False)
    gradient = np.ones((3, 2), dtype=np.float32)
    observed = [int(owner.params_resident_available()), int(owner.params_to_host() is owner.params)]
    for operation, args in ((owner.params_to_device, ()),
                            (owner.grads_to_device, ([gradient],)),
                            (owner.grads_to_host, ([gradient],))):
        try:
            operation(*args)
        except RuntimeError as error:
            observed.append(int("device-resident parameter" in str(error)))
        else:
            observed.append(0)
    return _fit(dict(contract=_h(np.asarray(observed, dtype=np.int32)), gradient=_h(gradient)))


@lane("optim-resident-io")
def _(ml, X, yc, yr, Xh=None):
    """Consume every supported GPU parameter/gradient IO operation, including device output."""
    params = [np.ascontiguousarray(X[:5, :3], dtype=np.float32).copy(),
              np.ascontiguousarray(X[5:9, 0], dtype=np.float32).copy()]
    gradients = [np.ascontiguousarray(X[10:15, :3], dtype=np.float32).copy(),
                 np.ascontiguousarray(X[15:19, 0], dtype=np.float32).copy()]
    owner = ml.training.AdamW(params, lr=.003, numeric_mode="identical", resident=True)
    if not owner.params_resident_available():
        raise RuntimeError("optim-resident-io requires the supported resident optimizer ABI")
    owner.params_to_device()
    # Overwrite caller buffers so a missing download cannot look like success.
    for p in params:
        p.fill(-17.)
    owner.params_to_host()
    parts = dict(params=_h(*params))
    owner.grads_to_device(gradients)
    downloaded = [np.full_like(g, -19.) for g in gradients]
    owner.grads_to_host(downloaded)
    parts["gradients"] = _h(*downloaded)
    outputs = [owner.param_device(j) for j in range(len(params))]
    parts["device-outputs"] = _h(*[np.asarray(p.to_host()) for p in outputs])
    for output in outputs:
        output.__del__()
    owner.step_device()
    owner.params_to_host()
    parts["stepped"] = _h(*params)
    return _fit(parts)


def _native_owner_closed(binding, handle, n):
    # The independent raw ABI checks the pool, rather than trusting the
    # wrapper's cleared field. No handle/address enters a portable digest.
    output = np.empty(n, dtype=np.float32)
    try:
        binding.seq_tensor_download(handle, output.ctypes.data, n)
    except Exception as error:
        if "not open" in str(error):
            return 1
        raise
    return 0


def _native_owner_failure_cleanup(binding, source, operation, owner=None):
    # Testing only: substitute a native builtin whose argument-count check
    # raises before touching inputs. This forces the actual native allocation
    # cleanup arm without a Python callback, bad pointer or compiler change.
    from mojolearn._x_sequence_device import _SEQ_CONTEXT, to_device
    entries = _SEQ_CONTEXT["sequence_bindings"]["identical"]
    probe = binding.seq_tensor_alloc(source.size, 0)
    binding.seq_tensor_free(probe)
    original = entries[operation]
    refused = 0
    try:
        entries[operation] = len
        try:
            if owner is None:
                to_device(source, numeric_mode="identical")
            else:
                owner.copy()
        except TypeError:
            refused = 1
    finally:
        entries[operation] = original
    return refused, _native_owner_closed(binding, probe, source.size)


@lane("sequence-device-release")
def _(ml, X, yc, yr, Xh=None):
    """Observe actual pool closure, view retention, double finalization and GC release."""
    import gc
    import weakref
    from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device
    source = np.ascontiguousarray(X[:5, :3], dtype=np.float32)
    owner = to_device(source, numeric_mode="identical")
    if type(owner) is not SequenceDeviceTensor:
        raise RuntimeError("sequence-device-release requires the supported resident tensor ABI")
    binding, handle = owner.b, owner.h
    view = owner.reshape((source.size,))
    retained = weakref.ref(owner)
    del owner
    gc.collect()
    observed = [int(retained() is not None)]
    view.__del__()
    observed.append(_native_owner_closed(binding, handle, source.size))
    values = np.asarray(view.numpy())
    owner = retained()
    del view
    owner.__del__()
    observed.append(_native_owner_closed(binding, handle, source.size))
    owner.__del__()
    observed.append(_native_owner_closed(binding, handle, source.size))
    orphan = to_device(source, numeric_mode="identical")
    orphan_handle = orphan.h
    del orphan
    gc.collect()
    observed.append(_native_owner_closed(binding, orphan_handle, source.size))
    observed.extend(_native_owner_failure_cleanup(binding, source, "seq_tensor_upload"))
    copy_source = to_device(source, numeric_mode="identical")
    observed.extend(_native_owner_failure_cleanup(binding, source, "seq_tensor_copy", copy_source))
    copy_source.__del__()
    return _fit(dict(lifecycle=_h(np.asarray(observed, dtype=np.int32)), values=_h(values)))


GPU_ONLY_LANES["optim-resident-io"] = "the CPU training binding has no resident parameter or gradient buffers"
GPU_ONLY_LANES["sequence-device-release"] = "the CPU sequence binding has no resident tensor pool to release"
_batch_decl("n/a:owner-lifecycle (series, checkpoints, buffer IO and release have no prediction batch axis)",
            "native-schedule-series", "optim-owner-lifecycle", "optim-io-capability",
            "optim-resident-io", "sequence-device-release")
