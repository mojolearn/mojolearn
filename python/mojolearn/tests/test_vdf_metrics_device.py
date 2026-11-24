# SPDX-License-Identifier: Apache-2.0
"""`device=` of every public metric (lane vdf-metrics): the boundary words only.

The device word is read FIRST in the metric's native door (bindings/metrics_device_door.mojo),
before any argument is admitted, so a refused word is refused whatever the other arguments are:
the calls below pass None for every required argument. No numerical check here (the digests
are verify's).
"""
import inspect

import pytest

import mojolearn
from mojolearn import metrics

# The public metrics: `mojolearn.metrics.__all__` and the two reports the module re-exports.
NAMES = sorted(set(metrics.__all__) | {"regression_report", "ranking_report"})
GPU_INSTALL = mojolearn.vendor() != "cpu"


def _call(name, device):
    function = getattr(metrics, name)
    positional = []
    keywords = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[parameter.name] = None
        else:
            positional.append(None)
    return function(*positional, device=device, **keywords)


def test_every_public_metric_is_listed():
    assert len(NAMES) == 71


@pytest.mark.parametrize("name", NAMES)
def test_device_is_keyword_only_and_defaults_to_none(name):
    parameter = inspect.signature(getattr(metrics, name)).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", ["tpu", 0])
def test_unknown_device_is_refused_in_the_shared_words(name, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        _call(name, bad)


@pytest.mark.parametrize("name", NAMES)
def test_explicit_gpu_on_a_cpu_only_installation_is_refused(name):
    # Holds on a CPU-only installation only (mojolearn.vendor() == "cpu"); on a GPU installation
    # every metric has a GPU implementation and the word is admitted (the call then refuses the
    # None inputs in the metric's own words, which this file does not check).
    if GPU_INSTALL:
        try:
            _call(name, "gpu")
        except Exception as error:  # the metric's own refusal of the None inputs
            assert "device must be" not in str(error) and "needs a GPU build" not in str(error)
    else:
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build"):
            _call(name, "gpu")


# The metrics that take a 2-D X: the only metric inputs a DeviceRows fold view can stand for.
X_METRICS = ["silhouette_score", "silhouette_samples", "calinski_harabasz_score", "davies_bouldin_score",
             "trustworthiness"]


@pytest.mark.parametrize("name", X_METRICS)
def test_device_resident_input_with_cpu_is_refused_in_unpack(name):
    # A device-resident input (DeviceRows) with an explicit device='cpu' is refused before any work,
    # whatever the installation; the gather callable must never run.
    from mojolearn import _arena_io
    from mojolearn._array import Array

    def forbidden(*args):
        raise AssertionError("the fold gather ran")

    rows = _arena_io.DeviceRows(Array.from_list([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "<f4"),
                                Array.from_list([0, 2], "<i8"), forbidden)
    function = getattr(metrics, name)
    second = rows if name == "trustworthiness" else [0, 1]
    with pytest.raises(ValueError, match="device='cpu' cannot take a device-resident input"):
        function(rows, second, device="cpu")
