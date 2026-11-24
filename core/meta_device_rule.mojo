# SPDX-License-Identifier: Apache-2.0
"""The member device rule of a meta-estimator that may have been HANDED its
device (lane vdf-meta, 2026-11-24; bindings/trees_meta_device_door.mojo).

A meta-estimator's explicit `device` is handed to a member that names none;
a member that names another explicit device is refused before anything runs.
When the meta-estimator was given the word itself (`origin` empty or its own
name) the words are the shared ones (core/function_device.mojo
`function_device_member`); when an outer meta-estimator handed it down, the
words name that one: "<origin> was given device='cpu', handed to <meta>,
but its estimator <child> was constructed with device='gpu'; ...".

Pure rule: no Python, no image; checked by checks/native_tests/meta_device_rule.mojo.
"""
from core.device_route import DEVICE_INSTALL, device_name
from core.function_device import function_device_member


def meta_member_device(request: Int, own: Int, meta: String, origin: String, child: String) raises -> Int:
    """The device a meta-estimator's member runs on (codes of
    core/device_route.mojo, DEVICE_INSTALL for None), or the refusal."""
    if origin.byte_length() == 0 or origin == meta:
        return function_device_member(request, own, meta, child)
    if request == DEVICE_INSTALL:
        return own
    if own == DEVICE_INSTALL or own == request:
        return request
    raise Error("ValueError: " + origin + " was given device='" + device_name(request) + "', handed to " + meta
        + ", but its estimator " + child + " was constructed with device='" + device_name(own)
        + "'; give both the same device, or leave one of them None")
