# SPDX-License-Identifier: Apache-2.0
"""The manifold estimators on both devices: the fit-level rows of `device=`.

Construction, refusals, `set_params`, clone and pickle of the parameter itself are covered for every class that takes
`device=` by `test_device_parameter.py`. Here, for UMAP, TSNE, Isomap, LocallyLinearEmbedding, MDS and ClassicalMDS: on
a GPU installation `device="cpu"` runs the CPU binaries beside the GPU ones in one process (`vendor_used() == "cpu"`,
nothing launched on the GPU; the bits of each side are the verify lanes umap, x-decomp-umap-options, x-ann-tsne-pca,
x-decomp-manifold); an estimator moved with
`set_params` runs on its new device. What needs two devices is skipped by name on a CPU-only installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both as cb
from mojolearn.manifold import TSNE, UMAP, ClassicalMDS, Isomap, LocallyLinearEmbedding, MDS

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")

# (id, class, constructor keywords, has transform)
CASES = [
    ("umap", UMAP, dict(n_neighbors=8, n_epochs=30, random_state=3, init="random"), True),
    ("umap-pca", UMAP, dict(n_neighbors=8, n_epochs=30, random_state=3, init="pca"), True),
    ("tsne", TSNE, dict(perplexity=8.0, max_iter=60, random_state=3), False),
    ("isomap", Isomap, dict(n_neighbors=8), True),
    ("lle", LocallyLinearEmbedding, dict(n_neighbors=8, random_state=3), True),
    ("mds", MDS, dict(max_iter=40, random_state=3), False),
    ("classical-mds", ClassicalMDS, dict(), False),
]
case = pytest.mark.parametrize("spec", CASES, ids=[row[0] for row in CASES])


def data(rows=72, cols=5, seed=9):
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols), dtype=np.float32)


def build(spec, **extra):
    return spec[1](**dict(spec[2], **extra))


@case
def test_default_device_is_this_installations_device(spec):
    x = data()
    model = build(spec)
    embedding = np.asarray(model.fit_transform(x))
    assert embedding.shape[0] == len(x)
    assert model.vendor_used() == mojolearn.vendor()
    assert model.get_params()["device"] is None


@case
def test_cpu_device_runs_the_cpu_binary(spec):
    x = data()
    model = build(spec, device="cpu")
    assert np.asarray(model.fit_transform(x)).shape[0] == len(x)
    assert model.vendor_used() == "cpu"
    if spec[3]:
        assert np.asarray(model.transform(x[:7])).shape[0] == 7


@two_devices
@case
def test_cpu_device_does_nothing_on_the_gpu(spec):
    x = data()
    build(spec, device="cpu").fit_transform(x)           # every route and image is connected before the record
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    model = build(spec, device="cpu")
    model.fit_transform(x)
    if spec[3]:
        model.transform(x[:7])
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    assert activity["launches"] == 0, activity["launch_names"]
    assert activity["upload_calls"] == 0 and activity["download_calls"] == 0


@two_devices
@case
def test_default_device_still_runs_on_the_gpu_beside_a_cpu_estimator(spec):
    x = data()
    cpu = build(spec, device="cpu")
    gpu = build(spec)
    cpu.fit_transform(x)
    gpu.fit_transform(x)
    assert cpu.vendor_used() == "cpu" and gpu.vendor_used() == mojolearn.vendor() != "cpu"
    if not cb.activity_begin(mojolearn):
        pytest.skip("this installation has no activity record")
    third = build(spec)
    third.fit_transform(x)
    activity = cb.activity_end(mojolearn)
    assert activity is not None
    # The record counts launches at the NAMED launch sites only (core/activity_record.mojo: "a family whose
    # device work passes none of these doors reads zero here"); the manifold kernels are not among them, so a
    # zero is "not counted on this path". What a GPU fit beside a CPU estimator must show is the route it ran
    # and that nothing it did was counted at a host inference entry.
    assert third.vendor_used() == mojolearn.vendor() != "cpu"
    assert activity["host_entries"] == 0, activity["host_names"]
    if activity["launch_names"]:
        assert activity["launches"] > 0


@two_devices
@case
def test_cpu_device_never_runs_a_fast_binary(spec):
    model = build(spec, device="cpu", numeric_mode="fast")
    try:
        model.fit_transform(data())
    except ValueError as refusal:
        # where `fast` names its own binaries (Apple): refused in words
        assert "device='cpu' runs in identical mode only" in str(refusal)
    except ImportError:
        # where no fast binaries are installed, the mode's own refusal stands
        pass
    else:
        # where the identical set answers `fast` (core/mode_answer.mojo; the form of
        # test_family_device_boundary and test_scaler_device_boundary): the CPU identical set ran
        assert model.vendor_used() == "cpu"


@two_devices
@case
def test_set_params_moves_the_estimator_to_the_other_device(spec):
    x = data()
    model = build(spec)
    model.fit_transform(x)
    assert model.vendor_used() == mojolearn.vendor()
    assert model.set_params(device="cpu") is model
    model.fit_transform(x)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None)
    model.fit_transform(x)
    assert model.vendor_used() == mojolearn.vendor()


@case
def test_the_check_takes_the_device_form(spec):
    model = build(spec)
    assert cb.takes_device(model)
    if TWO_DEVICES:
        twin = cb.on_device(model, "cpu", fitted=False)
        assert type(twin) is spec[1] and twin is not model and twin.get_params()["device"] == "cpu"
