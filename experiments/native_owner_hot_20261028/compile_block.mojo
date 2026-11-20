from bindings.neural_block_public_door import neural_block_forward_binding, neural_block_allocate_binding, neural_block_backward_binding
from bindings.native_loaded_module_door import NativeModules
from std.python import PythonObject

def main() raises:
    var mods = NativeModules()
    var value = PythonObject(None)
    _ = neural_block_forward_binding(mods, value, value, value, value, value, value)
    _ = neural_block_allocate_binding(mods, value, value, value, value)
    _ = neural_block_backward_binding(mods, value, value, value, value)
