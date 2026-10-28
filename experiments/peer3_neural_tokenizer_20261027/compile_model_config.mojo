# SPDX-License-Identifier: Apache-2.0
# Compile-only root for every public model configuration/plan operation.
from std.python._cpython import PyObjectPtr
from bindings.causal_model_config import model_config_c


@export
def peer3_model_config_compile(op: Int, py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    if op == 0:
        return model_config_c[0](py_self, args)
    if op == 1:
        return model_config_c[1](py_self, args)
    if op == 2:
        return model_config_c[2](py_self, args)
    if op == 3:
        return model_config_c[3](py_self, args)
    if op == 4:
        return model_config_c[4](py_self, args)
    if op == 5:
        return model_config_c[5](py_self, args)
    if op == 6:
        return model_config_c[6](py_self, args)
    if op == 7:
        return model_config_c[7](py_self, args)
    return model_config_c[8](py_self, args)
