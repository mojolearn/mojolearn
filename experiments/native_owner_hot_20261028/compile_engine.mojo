from bindings.causal_lm_public import causal_lm_forward_binding, causal_lm_generate_binding
from bindings.native_loaded_module import NativeModules
from std.python import PythonObject

def main() raises:
    var mods = NativeModules()
    var value = PythonObject(None)
    _ = causal_lm_forward_binding(mods, value, value, value, value)
    _ = causal_lm_generate_binding(mods, value, value, value, value)
