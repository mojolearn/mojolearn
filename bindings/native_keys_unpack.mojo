# SPDX-License-Identifier: Apache-2.0
# The native family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

# Three dispatchers share this table and their names collide ("initialize",
# "close"), so each has its own `ops_<door>` dictionary and
# `NATIVE_<DOOR>_OPN_*` codes (the bw2-parallel form): codec_c
# (native_codec_door), process_c (native_process_pool_door) and
# transport_c (native_transport_door).
# `codec` dispatcher codes (`ops_codec`).
comptime NATIVE_CODEC_OPN_INITIALIZE = 0
comptime NATIVE_CODEC_OPN_PACK = 1
comptime NATIVE_CODEC_OPN_UNPACK = 2
# `process` dispatcher codes (`ops_process`).
comptime NATIVE_PROCESS_OPN_INITIALIZE = 0
comptime NATIVE_PROCESS_OPN_CREATE = 1
comptime NATIVE_PROCESS_OPN_MAP = 2
comptime NATIVE_PROCESS_OPN_CLOSE = 3
comptime NATIVE_PROCESS_OPN_WORKER_ECHO = 4
comptime NATIVE_PROCESS_OPN_ENVIRONMENT = 5
comptime NATIVE_PROCESS_OPN_MIRROR = 6
comptime NATIVE_PROCESS_OPN_MODULE = 7
# `transport` dispatcher codes (`ops_transport`).
comptime NATIVE_TRANSPORT_OPN_INITIALIZE = 0
comptime NATIVE_TRANSPORT_OPN_IDNA = 1
comptime NATIVE_TRANSPORT_OPN_LISTEN = 2
comptime NATIVE_TRANSPORT_OPN_ACCEPT = 3
comptime NATIVE_TRANSPORT_OPN_CONNECT = 4
comptime NATIVE_TRANSPORT_OPN_ADDRESS = 5
comptime NATIVE_TRANSPORT_OPN_PORT = 6
comptime NATIVE_TRANSPORT_OPN_TIMEOUT = 7
comptime NATIVE_TRANSPORT_OPN_SEND = 8
comptime NATIVE_TRANSPORT_OPN_RECEIVE = 9
comptime NATIVE_TRANSPORT_OPN_CLOSE = 10
comptime NATIVE_TRANSPORT_OPN_SLEEP = 11
comptime NATIVE_TRANSPORT_OPN_SPAWN = 12
comptime NATIVE_TRANSPORT_OPN_WAIT = 13
comptime NATIVE_TRANSPORT_OPN_TERMINATE = 14



struct NativeKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops_codec: PythonObject
    var ops_process: PythonObject
    var ops_transport: PythonObject
    var k_ARRAY_TYPE: PythonObject
    var k_ClassLabels: PythonObject
    var k_DispatchKey: PythonObject
    var k_MOJOLEARN_NUMERIC_MODE: PythonObject
    var k_MOJOLEARN_VENDOR: PythonObject
    var k_Parameter: PythonObject
    var k_PreDispatch: PythonObject
    var k_Tensor: PythonObject
    var k_TensorBase: PythonObject
    var k_ZERO_STORES: PythonObject
    var k__ARRAY_CONTEXT: PythonObject
    var k__TensorBase: PythonObject
    var k__TorchDispatchModeKey: PythonObject
    var k___bases__: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___dictoffset__: PythonObject
    var k___flags__: PythonObject
    var k___getattr__: PythonObject
    var k___getattribute__: PythonObject
    var k___module__: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k___setattr__: PythonObject
    var k___slots__: PythonObject
    var k___version__: PythonObject
    var k___weakrefoffset__: PythonObject
    var k__addr: PythonObject
    var k__base: PythonObject
    var k__dispatch_tls_is_dispatch_key_included: PythonObject
    var k__get_dispatch_mode: PythonObject
    var k__get_promotion_state: PythonObject
    var k__len_torch_dispatch_stack: PythonObject
    var k__len_torch_function_stack: PythonObject
    var k__mv: PythonObject
    var k__readonly: PythonObject
    var k__store: PythonObject
    var k_array: PythonObject
    var k_array_class: PythonObject
    var k_array_storage_dtypes: PythonObject
    var k_array_storage_slots: PythonObject
    var k_array_type: PythonObject
    var k_begin: PythonObject
    var k_bits: PythonObject
    var k_bool: PythonObject
    var k_buffer_owner_types: PythonObject
    var k_bytearray: PythonObject
    var k_bytes: PythonObject
    var k_close_seconds: PythonObject
    var k_close_signal: PythonObject
    var k_closed: PythonObject
    var k_codes: PythonObject
    var k_complex: PythonObject
    var k_contiguous: PythonObject
    var k_cooperative: PythonObject
    var k_cpu: PythonObject
    var k_data_ptr: PythonObject
    var k_detach: PythonObject
    var k_device: PythonObject
    var k_devices: PythonObject
    var k_dict: PythonObject
    var k_driver_library: PythonObject
    var k_dtype: PythonObject
    var k_exponents: PythonObject
    var k_float: PythonObject
    var k_from_buffer_slots: PythonObject
    var k_identity: PythonObject
    var k_int: PythonObject
    var k_integer_types: PythonObject
    var k_item: PythonObject
    var k_items: PythonObject
    var k_itemsize: PythonObject
    var k_kind: PythonObject
    var k_labels_integral_type: PythonObject
    var k_labels_real_type: PythonObject
    var k_list: PythonObject
    var k_memoryview: PythonObject
    var k_modules: PythonObject
    var k_name: PythonObject
    var k_native_scalar_promotion_getter: PythonObject
    var k_nbytes: PythonObject
    var k_ndarray: PythonObject
    var k_ndim: PythonObject
    var k_numel: PythonObject
    var k_numpy: PythonObject
    var k_obj: PythonObject
    var k_object: PythonObject
    var k_order: PythonObject
    var k_ordinal: PythonObject
    var k_owner_array_slots: PythonObject
    var k_owner_builtin_function_class: PythonObject
    var k_owner_capsule_class: PythonObject
    var k_owner_dtype_classes: PythonObject
    var k_owner_dtype_scalars: PythonObject
    var k_owner_getweakrefs: PythonObject
    var k_owner_method_class: PythonObject
    var k_owner_mmap_class: PythonObject
    var k_owner_module_class: PythonObject
    var k_owner_module_dict: PythonObject
    var k_owner_native_types: PythonObject
    var k_owner_sys_modules: PythonObject
    var k_owner_weakref_callback: PythonObject
    var k_owner_weakref_class: PythonObject
    var k_pci_bus_id: PythonObject
    var k_pid: PythonObject
    var k_ppid: PythonObject
    var k_readonly: PythonObject
    var k_scalar_types: PythonObject
    var k_shape: PythonObject
    var k_size: PythonObject
    var k_start_seconds: PythonObject
    var k_str: PythonObject
    var k_stride: PythonObject
    var k_strides: PythonObject
    var k_timing: PythonObject
    var k_to: PythonObject
    var k_token: PythonObject
    var k_trees_integral_type: PythonObject
    var k_trees_real_type: PythonObject
    var k_tuple: PythonObject
    var k_type: PythonObject
    var k_uuid: PythonObject
    var k_value: PythonObject
    var k_vendor: PythonObject
    var k_visibility: PythonObject
    var k_workers: PythonObject
    var k__mojolearn_typed_context_labels: PythonObject
    var k__mojolearn_typed_context_trees: PythonObject
    var k__mojolearn_typed_context_both: PythonObject
    var k___new__: PythonObject
    var k__pin: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops_codec = PythonObject(None)
        self.ops_process = PythonObject(None)
        self.ops_transport = PythonObject(None)
        self.k_ARRAY_TYPE = PythonObject(None)
        self.k_ClassLabels = PythonObject(None)
        self.k_DispatchKey = PythonObject(None)
        self.k_MOJOLEARN_NUMERIC_MODE = PythonObject(None)
        self.k_MOJOLEARN_VENDOR = PythonObject(None)
        self.k_Parameter = PythonObject(None)
        self.k_PreDispatch = PythonObject(None)
        self.k_Tensor = PythonObject(None)
        self.k_TensorBase = PythonObject(None)
        self.k_ZERO_STORES = PythonObject(None)
        self.k__ARRAY_CONTEXT = PythonObject(None)
        self.k__TensorBase = PythonObject(None)
        self.k__TorchDispatchModeKey = PythonObject(None)
        self.k___bases__ = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___dictoffset__ = PythonObject(None)
        self.k___flags__ = PythonObject(None)
        self.k___getattr__ = PythonObject(None)
        self.k___getattribute__ = PythonObject(None)
        self.k___module__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k___setattr__ = PythonObject(None)
        self.k___slots__ = PythonObject(None)
        self.k___version__ = PythonObject(None)
        self.k___weakrefoffset__ = PythonObject(None)
        self.k__addr = PythonObject(None)
        self.k__base = PythonObject(None)
        self.k__dispatch_tls_is_dispatch_key_included = PythonObject(None)
        self.k__get_dispatch_mode = PythonObject(None)
        self.k__get_promotion_state = PythonObject(None)
        self.k__len_torch_dispatch_stack = PythonObject(None)
        self.k__len_torch_function_stack = PythonObject(None)
        self.k__mv = PythonObject(None)
        self.k__readonly = PythonObject(None)
        self.k__store = PythonObject(None)
        self.k_array = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_storage_dtypes = PythonObject(None)
        self.k_array_storage_slots = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_begin = PythonObject(None)
        self.k_bits = PythonObject(None)
        self.k_bool = PythonObject(None)
        self.k_buffer_owner_types = PythonObject(None)
        self.k_bytearray = PythonObject(None)
        self.k_bytes = PythonObject(None)
        self.k_close_seconds = PythonObject(None)
        self.k_close_signal = PythonObject(None)
        self.k_closed = PythonObject(None)
        self.k_codes = PythonObject(None)
        self.k_complex = PythonObject(None)
        self.k_contiguous = PythonObject(None)
        self.k_cooperative = PythonObject(None)
        self.k_cpu = PythonObject(None)
        self.k_data_ptr = PythonObject(None)
        self.k_detach = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_devices = PythonObject(None)
        self.k_dict = PythonObject(None)
        self.k_driver_library = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_exponents = PythonObject(None)
        self.k_float = PythonObject(None)
        self.k_from_buffer_slots = PythonObject(None)
        self.k_identity = PythonObject(None)
        self.k_int = PythonObject(None)
        self.k_integer_types = PythonObject(None)
        self.k_item = PythonObject(None)
        self.k_items = PythonObject(None)
        self.k_itemsize = PythonObject(None)
        self.k_kind = PythonObject(None)
        self.k_labels_integral_type = PythonObject(None)
        self.k_labels_real_type = PythonObject(None)
        self.k_list = PythonObject(None)
        self.k_memoryview = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_name = PythonObject(None)
        self.k_native_scalar_promotion_getter = PythonObject(None)
        self.k_nbytes = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_ndim = PythonObject(None)
        self.k_numel = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_obj = PythonObject(None)
        self.k_object = PythonObject(None)
        self.k_order = PythonObject(None)
        self.k_ordinal = PythonObject(None)
        self.k_owner_array_slots = PythonObject(None)
        self.k_owner_builtin_function_class = PythonObject(None)
        self.k_owner_capsule_class = PythonObject(None)
        self.k_owner_dtype_classes = PythonObject(None)
        self.k_owner_dtype_scalars = PythonObject(None)
        self.k_owner_getweakrefs = PythonObject(None)
        self.k_owner_method_class = PythonObject(None)
        self.k_owner_mmap_class = PythonObject(None)
        self.k_owner_module_class = PythonObject(None)
        self.k_owner_module_dict = PythonObject(None)
        self.k_owner_native_types = PythonObject(None)
        self.k_owner_sys_modules = PythonObject(None)
        self.k_owner_weakref_callback = PythonObject(None)
        self.k_owner_weakref_class = PythonObject(None)
        self.k_pci_bus_id = PythonObject(None)
        self.k_pid = PythonObject(None)
        self.k_ppid = PythonObject(None)
        self.k_readonly = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_size = PythonObject(None)
        self.k_start_seconds = PythonObject(None)
        self.k_str = PythonObject(None)
        self.k_stride = PythonObject(None)
        self.k_strides = PythonObject(None)
        self.k_timing = PythonObject(None)
        self.k_to = PythonObject(None)
        self.k_token = PythonObject(None)
        self.k_trees_integral_type = PythonObject(None)
        self.k_trees_real_type = PythonObject(None)
        self.k_tuple = PythonObject(None)
        self.k_type = PythonObject(None)
        self.k_uuid = PythonObject(None)
        self.k_value = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_visibility = PythonObject(None)
        self.k_workers = PythonObject(None)
        self.k__mojolearn_typed_context_labels = PythonObject(None)
        self.k__mojolearn_typed_context_trees = PythonObject(None)
        self.k__mojolearn_typed_context_both = PythonObject(None)
        self.k___new__ = PythonObject(None)
        self.k__pin = PythonObject(None)


comptime _NATIVE_KEYS = SharedSlot[NativeKeys, "MojolearnNativeKeysV1"]


def _native_keys_fill(nk: UnsafePointer[NativeKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_ARRAY_TYPE = py_key("ARRAY_TYPE")
    nk[].k_ClassLabels = py_key("ClassLabels")
    nk[].k_DispatchKey = py_key("DispatchKey")
    nk[].k_MOJOLEARN_NUMERIC_MODE = py_key("MOJOLEARN_NUMERIC_MODE")
    nk[].k_MOJOLEARN_VENDOR = py_key("MOJOLEARN_VENDOR")
    nk[].k_Parameter = py_key("Parameter")
    nk[].k_PreDispatch = py_key("PreDispatch")
    nk[].k_Tensor = py_key("Tensor")
    nk[].k_TensorBase = py_key("TensorBase")
    nk[].k_ZERO_STORES = py_key("ZERO_STORES")
    nk[].k__ARRAY_CONTEXT = py_key("_ARRAY_CONTEXT")
    nk[].k__TensorBase = py_key("_TensorBase")
    nk[].k__TorchDispatchModeKey = py_key("_TorchDispatchModeKey")
    nk[].k___bases__ = py_key("__bases__")
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___dictoffset__ = py_key("__dictoffset__")
    nk[].k___flags__ = py_key("__flags__")
    nk[].k___getattr__ = py_key("__getattr__")
    nk[].k___getattribute__ = py_key("__getattribute__")
    nk[].k___module__ = py_key("__module__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k___setattr__ = py_key("__setattr__")
    nk[].k___slots__ = py_key("__slots__")
    nk[].k___version__ = py_key("__version__")
    nk[].k___weakrefoffset__ = py_key("__weakrefoffset__")
    nk[].k__addr = py_key("_addr")
    nk[].k__base = py_key("_base")
    nk[].k__dispatch_tls_is_dispatch_key_included = py_key("_dispatch_tls_is_dispatch_key_included")
    nk[].k__get_dispatch_mode = py_key("_get_dispatch_mode")
    nk[].k__get_promotion_state = py_key("_get_promotion_state")
    nk[].k__len_torch_dispatch_stack = py_key("_len_torch_dispatch_stack")
    nk[].k__len_torch_function_stack = py_key("_len_torch_function_stack")
    nk[].k__mv = py_key("_mv")
    nk[].k__readonly = py_key("_readonly")
    nk[].k__store = py_key("_store")
    nk[].k_array = py_key("array")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_storage_dtypes = py_key("array_storage_dtypes")
    nk[].k_array_storage_slots = py_key("array_storage_slots")
    nk[].k_array_type = py_key("array_type")
    nk[].k_begin = py_key("begin")
    nk[].k_bits = py_key("bits")
    nk[].k_bool = py_key("bool")
    nk[].k_buffer_owner_types = py_key("buffer_owner_types")
    nk[].k_bytearray = py_key("bytearray")
    nk[].k_bytes = py_key("bytes")
    nk[].k_close_seconds = py_key("close_seconds")
    nk[].k_close_signal = py_key("close_signal")
    nk[].k_closed = py_key("closed")
    nk[].k_codes = py_key("codes")
    nk[].k_complex = py_key("complex")
    nk[].k_contiguous = py_key("contiguous")
    nk[].k_cooperative = py_key("cooperative")
    nk[].k_cpu = py_key("cpu")
    nk[].k_data_ptr = py_key("data_ptr")
    nk[].k_detach = py_key("detach")
    nk[].k_device = py_key("device")
    nk[].k_devices = py_key("devices")
    nk[].k_dict = py_key("dict")
    nk[].k_driver_library = py_key("driver_library")
    nk[].k_dtype = py_key("dtype")
    nk[].k_exponents = py_key("exponents")
    nk[].k_float = py_key("float")
    nk[].k_from_buffer_slots = py_key("from_buffer_slots")
    nk[].k_identity = py_key("identity")
    nk[].k_int = py_key("int")
    nk[].k_integer_types = py_key("integer_types")
    nk[].k_item = py_key("item")
    nk[].k_items = py_key("items")
    nk[].k_itemsize = py_key("itemsize")
    nk[].k_kind = py_key("kind")
    nk[].k_labels_integral_type = py_key("labels_integral_type")
    nk[].k_labels_real_type = py_key("labels_real_type")
    nk[].k_list = py_key("list")
    nk[].k_memoryview = py_key("memoryview")
    nk[].k_modules = py_key("modules")
    nk[].k_name = py_key("name")
    nk[].k_native_scalar_promotion_getter = py_key("native_scalar_promotion_getter")
    nk[].k_nbytes = py_key("nbytes")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_ndim = py_key("ndim")
    nk[].k_numel = py_key("numel")
    nk[].k_numpy = py_key("numpy")
    nk[].k_obj = py_key("obj")
    nk[].k_object = py_key("object")
    nk[].k_order = py_key("order")
    nk[].k_ordinal = py_key("ordinal")
    nk[].k_owner_array_slots = py_key("owner_array_slots")
    nk[].k_owner_builtin_function_class = py_key("owner_builtin_function_class")
    nk[].k_owner_capsule_class = py_key("owner_capsule_class")
    nk[].k_owner_dtype_classes = py_key("owner_dtype_classes")
    nk[].k_owner_dtype_scalars = py_key("owner_dtype_scalars")
    nk[].k_owner_getweakrefs = py_key("owner_getweakrefs")
    nk[].k_owner_method_class = py_key("owner_method_class")
    nk[].k_owner_mmap_class = py_key("owner_mmap_class")
    nk[].k_owner_module_class = py_key("owner_module_class")
    nk[].k_owner_module_dict = py_key("owner_module_dict")
    nk[].k_owner_native_types = py_key("owner_native_types")
    nk[].k_owner_sys_modules = py_key("owner_sys_modules")
    nk[].k_owner_weakref_callback = py_key("owner_weakref_callback")
    nk[].k_owner_weakref_class = py_key("owner_weakref_class")
    nk[].k_pci_bus_id = py_key("pci_bus_id")
    nk[].k_pid = py_key("pid")
    nk[].k_ppid = py_key("ppid")
    nk[].k_readonly = py_key("readonly")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_shape = py_key("shape")
    nk[].k_size = py_key("size")
    nk[].k_start_seconds = py_key("start_seconds")
    nk[].k_str = py_key("str")
    nk[].k_stride = py_key("stride")
    nk[].k_strides = py_key("strides")
    nk[].k_timing = py_key("timing")
    nk[].k_to = py_key("to")
    nk[].k_token = py_key("token")
    nk[].k_trees_integral_type = py_key("trees_integral_type")
    nk[].k_trees_real_type = py_key("trees_real_type")
    nk[].k_tuple = py_key("tuple")
    nk[].k_type = py_key("type")
    nk[].k_uuid = py_key("uuid")
    nk[].k_value = py_key("value")
    nk[].k_vendor = py_key("vendor")
    nk[].k_visibility = py_key("visibility")
    nk[].k_workers = py_key("workers")
    nk[].k__mojolearn_typed_context_labels = py_key("_mojolearn_typed_context_labels")
    nk[].k__mojolearn_typed_context_trees = py_key("_mojolearn_typed_context_trees")
    nk[].k__mojolearn_typed_context_both = py_key("_mojolearn_typed_context_both")
    nk[].k___new__ = py_key("__new__")
    nk[].k__pin = py_key("_pin")
    var ops_codec = Python.dict()
    py_op_add(ops_codec, "initialize", NATIVE_CODEC_OPN_INITIALIZE)
    py_op_add(ops_codec, "pack", NATIVE_CODEC_OPN_PACK)
    py_op_add(ops_codec, "unpack", NATIVE_CODEC_OPN_UNPACK)
    nk[].ops_codec = ops_codec
    var ops_process = Python.dict()
    py_op_add(ops_process, "initialize", NATIVE_PROCESS_OPN_INITIALIZE)
    py_op_add(ops_process, "create", NATIVE_PROCESS_OPN_CREATE)
    py_op_add(ops_process, "map", NATIVE_PROCESS_OPN_MAP)
    py_op_add(ops_process, "close", NATIVE_PROCESS_OPN_CLOSE)
    py_op_add(ops_process, "worker_echo", NATIVE_PROCESS_OPN_WORKER_ECHO)
    py_op_add(ops_process, "environment", NATIVE_PROCESS_OPN_ENVIRONMENT)
    py_op_add(ops_process, "mirror", NATIVE_PROCESS_OPN_MIRROR)
    py_op_add(ops_process, "module", NATIVE_PROCESS_OPN_MODULE)
    nk[].ops_process = ops_process
    var ops_transport = Python.dict()
    py_op_add(ops_transport, "initialize", NATIVE_TRANSPORT_OPN_INITIALIZE)
    py_op_add(ops_transport, "idna", NATIVE_TRANSPORT_OPN_IDNA)
    py_op_add(ops_transport, "listen", NATIVE_TRANSPORT_OPN_LISTEN)
    py_op_add(ops_transport, "accept", NATIVE_TRANSPORT_OPN_ACCEPT)
    py_op_add(ops_transport, "connect", NATIVE_TRANSPORT_OPN_CONNECT)
    py_op_add(ops_transport, "address", NATIVE_TRANSPORT_OPN_ADDRESS)
    py_op_add(ops_transport, "port", NATIVE_TRANSPORT_OPN_PORT)
    py_op_add(ops_transport, "timeout", NATIVE_TRANSPORT_OPN_TIMEOUT)
    py_op_add(ops_transport, "send", NATIVE_TRANSPORT_OPN_SEND)
    py_op_add(ops_transport, "receive", NATIVE_TRANSPORT_OPN_RECEIVE)
    py_op_add(ops_transport, "close", NATIVE_TRANSPORT_OPN_CLOSE)
    py_op_add(ops_transport, "sleep", NATIVE_TRANSPORT_OPN_SLEEP)
    py_op_add(ops_transport, "spawn", NATIVE_TRANSPORT_OPN_SPAWN)
    py_op_add(ops_transport, "wait", NATIVE_TRANSPORT_OPN_WAIT)
    py_op_add(ops_transport, "terminate", NATIVE_TRANSPORT_OPN_TERMINATE)
    nk[].ops_transport = ops_transport
    nk[].ready = True


@always_inline
def native_keys() raises -> UnsafePointer[NativeKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _NATIVE_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _native_keys_fill(nk)
    return nk
