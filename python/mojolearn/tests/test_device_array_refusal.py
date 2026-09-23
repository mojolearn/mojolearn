# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A device array must REFUSE BY NAME, and host input must be untouched.

DEVIATION 2692. Before this, a cupy or torch-CUDA array had no host buffer,
so `_materialize` fell into the nested-list branch and raised
`mojolearn: X holds a ndarray, not a number` -- an error about ELEMENT TYPES
for input whose actual problem is that the memory lives on a GPU. Nothing in
it mentioned the device.

THE NEGATIVE CONTROLS ARE THE POINT OF THIS FILE. The refusal is a new branch
placed BEFORE the buffer path, so the real risk is not that it fails to fire,
it is that it fires on something it should not. A CPU torch tensor exports
DLPack *and* the buffer protocol; if the DLPack arm did not ask
`__dlpack_device__` for the device type, host tensors would start refusing
and every `fit` in the library would break on them. `test_cpu_dlpack_*` is
that guard, and it asserts the ZERO-COPY path specifically (`copied=False`),
not merely that the call succeeded.

No GPU is required and none is touched: detection reads protocol attributes
only, never a device pointer.
"""

import array
import unittest

from .. import _buffer


class _CupyLike:
    """Exports `__cuda_array_interface__`, no host buffer."""

    def __init__(self):
        self.__cuda_array_interface__ = {
            "shape": (4, 3),
            "typestr": "<f4",
            "data": (140234567890, False),
            "version": 3,
        }


class _TorchCudaLike:
    """Exports DLPack reporting kDLCUDA (2), no host buffer."""

    def __dlpack__(self, *args, **kwargs):
        raise RuntimeError("not called: detection must not read the pointer")

    def __dlpack_device__(self):
        return (2, 0)


class _TorchCpuLike(array.array):
    """A CPU tensor: DLPack reporting kDLCPU (1) AND a host buffer.

    An array.array subclass rather than a class with `__buffer__`: a
    pure-Python `__buffer__` is honored from 3.12 only, so on 3.10 and
    3.11 the old fake exported no buffer at all and `_materialize` read it
    as a scalar (red on the 2026-09-23 pod, green on 3.14). A real host
    tensor exports its buffer from C on every Python; so does this.
    """

    def __new__(cls):
        return super().__new__(cls, "f", [1.0, 2.0, 3.0, 4.0])

    def __dlpack__(self, *args, **kwargs):
        raise RuntimeError("not called: a host tensor takes the buffer path")

    def __dlpack_device__(self):
        return (1, 0)


class _BadDlpackDevice:
    """`__dlpack_device__` raises; must not be treated as a device array."""

    def __dlpack_device__(self):
        raise RuntimeError("broken exporter")


class DeviceArrayRefusalTest(unittest.TestCase):
    def test_cuda_array_interface_refuses_by_name(self):
        with self.assertRaises(TypeError) as caught:
            _buffer._materialize(_CupyLike(), "X")
        message = str(caught.exception)
        self.assertIn("DEVICE array", message)
        self.assertIn("__cuda_array_interface__", message)
        self.assertIn("HOST memory only", message)
        self.assertIn("x.get()", message)
        # The OLD misleading message must be gone.
        self.assertNotIn("not a number", message)

    def test_dlpack_cuda_refuses_and_names_the_device_type(self):
        with self.assertRaises(TypeError) as caught:
            _buffer._materialize(_TorchCudaLike(), "X")
        message = str(caught.exception)
        self.assertIn("DEVICE array", message)
        self.assertIn("__dlpack__", message)
        self.assertIn("x.cpu()", message)

    def test_refusal_names_the_argument(self):
        with self.assertRaises(TypeError) as caught:
            _buffer._materialize(_CupyLike(), "sample_weight")
        self.assertIn("sample_weight", str(caught.exception))

    # ---------------------------------------------- negative controls

    def test_cpu_dlpack_tensor_keeps_the_zero_copy_path(self):
        materialized, copied = _buffer._materialize(_TorchCpuLike(), "X")
        self.assertFalse(copied, "a host DLPack tensor must not be copied")
        self.assertIsNotNone(materialized)

    def test_host_buffer_is_untouched(self):
        materialized, copied = _buffer._materialize(array.array("f", [1.0, 2.0]), "X")
        self.assertFalse(copied)
        self.assertIsNotNone(materialized)

    def test_nested_list_is_untouched(self):
        materialized, copied = _buffer._materialize([[1.0, 2.0], [3.0, 4.0]], "X")
        self.assertTrue(copied)
        self.assertIsNotNone(materialized)

    def test_broken_dlpack_exporter_is_not_a_device_array(self):
        self.assertIsNone(_buffer._device_array_kind(_BadDlpackDevice()))

    # ------------------------------------------- the address chokepoint

    def test_addr_path_also_refuses_by_name(self):
        """`view`/`addr`/`addr_ro` bypass `_materialize` and reach `Buf`.

        Before DEVIATION 2692's second half these said "does not support the
        buffer protocol" -- true, but silent about the GPU. No caller reaches
        them with user input today (ARIMA converts through `_series_major`
        first), so this is defence in depth against a future one.
        """
        for entry in ("view", "addr", "addr_ro"):
            function = getattr(_buffer, entry)
            with self.assertRaises(TypeError) as caught:
                function(_CupyLike(), name="X")
            message = str(caught.exception)
            self.assertIn("DEVICE array", message, entry)
            self.assertNotIn("does not support the buffer protocol", message)

    def test_addr_path_still_accepts_host_memory(self):
        storage = array.array("f", [1.0, 2.0, 3.0])
        self.assertIsInstance(_buffer.addr_ro(storage, name="X"), int)

    def test_kind_discriminates(self):
        self.assertEqual(
            _buffer._device_array_kind(_CupyLike()), "__cuda_array_interface__"
        )
        self.assertIn("__dlpack__", _buffer._device_array_kind(_TorchCudaLike()))
        self.assertIsNone(_buffer._device_array_kind(_TorchCpuLike()))
        self.assertIsNone(_buffer._device_array_kind(array.array("f", [1.0])))
        self.assertIsNone(_buffer._device_array_kind([[1.0]]))


if __name__ == "__main__":
    unittest.main()
