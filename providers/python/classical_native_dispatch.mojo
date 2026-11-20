# SPDX-License-Identifier: Apache-2.0
# The classical doors' descriptor dispatch, fixed at build time (lane
# flag-classical-dispatch-gate, 2026-11-15). Before, the run-time environment
# gate MOJOLEARN_CLASSICAL_NATIVE_DISPATCH (default "0", added 2026-10-27 by
# 32d1319e93 / f36744285f / d14bcf8251, before v0.8.66) chose the route per
# public operation in three readers: the SVC/SVR provider admission
# (bindings/svc_kernel_native_door.mojo `_svc_provider`), the x_neighbors
# module admission (providers/python/x_neighbors_descriptor_pyapi_door.mojo
# `xn_admit`) and the SVM / x_neighbors mode readback
# (providers/python/estimator_binding_pyapi_door.mojo `estimator_checked_mode`).
#
# Default (A, the route every board row of 0.8.66 and of fcc934a37c ran, since
# the gate was never set by any harness): SVC/SVR call their compiled entries
# by name and keep the fitted state device resident (`svc_predict_resident`,
# `svm_resident_lease`) and X on the device for a fit (`device_x`); an
# x_neighbors MODULE handed to `xn_admit` (fast mode, explicit internal
# callers) keeps its by-name packed entries. Identical x_neighbors public calls
# are unaffected by this switch: `estimator_checked_mode` admits their native
# snapshot unconditionally (d2b70e3047, 2026-10-28).
#
# B arm `-D MOJOLEARN_CLASSICAL_NATIVE_DISPATCH` (never timed; RUN OWED,
# bench/results/paper_compare/owed-pre0867.json "classical-native-dispatch",
# docs/apple-fast/EXPERIMENTS.md): the SVC/SVR descriptor table (no CPython
# entry call per pair, fold or epilogue, but no device-resident predict and no
# device-X fit), native admission of x_neighbors modules, and no SVM mode
# readback (the provider checks mode itself). Both arms run the same numerical
# owners on the same words (experiments/classical_shared_provider_20261027,
# experiments/x_neighbors_provider_20261027): no bit change.
from std.sys import is_defined

comptime CLASSICAL_NATIVE_DISPATCH = is_defined["MOJOLEARN_CLASSICAL_NATIVE_DISPATCH"]()
