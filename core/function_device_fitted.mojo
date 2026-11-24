# SPDX-License-Identifier: Apache-2.0
"""The device of a public function that QUERIES an already fitted estimator
(lane vdf-misc, 2026-11-24): `hdbscan.approximate_predict`,
`hdbscan.membership_vector`, `hdbscan.all_points_membership_vectors`.

Such a function runs where its fitted model lives: on the model's own route
(its `device`, resolved once per estimator). Its `device=` (admitted by the
shared rule, core/function_device.mojo) only states the side the caller
expects, and a disagreement is refused in words before any work; the query
never moves the fitted state to the other side.

* None: the model's route, unchanged (no read beyond the identity compare
  the door already made).
* An explicit word and a model constructed with a different explicit
  `device`: refused in `function_device_member`'s words.
* "cpu" beside a GPU set (`host_beside`) and a model constructed with
  `device=None` (it ran on this installation's device, the GPU set):
  refused, naming the function and the model's class.
* Every other combination is the side the model already runs on.

Pure rule: no Python, no image; checked by
checks/native_tests/function_device_fitted.mojo.
"""
from core.device_route import DEVICE_INSTALL
from core.function_device import FunctionDevice, function_device_member


def function_device_fitted(device: FunctionDevice, member: Int, name: String, member_class: String) raises:
    """Refuse a query function's `device` that disagrees with the side its
    fitted model runs on. `member` is the model's own device code
    (core/device_route.mojo, DEVICE_INSTALL for None)."""
    if device.request == DEVICE_INSTALL:
        return
    _ = function_device_member(device.request, member, name, member_class)
    if device.host_beside and member == DEVICE_INSTALL:
        raise Error("ValueError: " + name + " was given device='cpu' but its " + member_class
            + " was constructed with device=None and runs on this installation's GPU; a fitted model is queried "
            + "where it was fitted: construct it with device='cpu', or leave device=None")
