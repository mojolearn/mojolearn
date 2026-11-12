# SPDX-License-Identifier: Apache-2.0
"""RUN half of `providers/python/autograd_weight_frontend_pyapi.mojo` (wave 8
run split): a weight dtype name's native code and element size. Typed Mojo
only: no Python import."""


def _dtype(dtype: String) raises -> Tuple[Int, Int]:
    if dtype == "F16" or dtype == "torch.float16":
        return (23, 2)
    if dtype == "F32" or dtype == "torch.float32":
        return (11, 4)
    if dtype == "F64" or dtype == "torch.float64":
        return (12, 8)
    if dtype == "BF16" or dtype == "torch.bfloat16":
        return (4096, 2)
    raise Error("TypeError: a weight must be float16, bfloat16, float32 or float64; got " + dtype)
