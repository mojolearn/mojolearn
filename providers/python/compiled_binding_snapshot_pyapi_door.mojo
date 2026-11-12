# SPDX-License-Identifier: Apache-2.0
# Operation-local metadata, not another execution provider. The existing common
# provider owns dispatch/module lifetime; this immutable-by-convention admission
# record carries its freshly read scalar metadata through one public operation.
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider
from core.shared_slot import SharedSlot


struct CompiledBindingSnapshot(Movable, Writable):
    var provider: PythonObject
    var module: PythonObject
    var abi: String
    var mode: Int
    var capabilities: List[Int]

    def __init__(out self, provider: PythonObject, module: PythonObject, var abi: String,
                 mode: Int, var capabilities: List[Int]) raises:
        var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
        if owner[].module is not module or owner[].name != abi:
            raise Error("compiled binding snapshot requires its admitted provider and module")
        self.provider = provider
        self.module = module
        self.abi = abi^
        self.mode = mode
        self.capabilities = capabilities^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CompiledBindingSnapshot(", self.abi, ", mode=", self.mode, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct CompiledBindingSnapshots(Defaultable, Movable):
    # Admitted snapshots kept for the process (lane perf-provider-cache,
    # 2026-11-06). A snapshot's mode and capability words are compile-time
    # constants of its loaded image, so re-reading them through a freshly
    # connected provider on every public operation (one capsule, one provider
    # object and two descriptor calls per call) repeated the same answer.
    # Each record holds its module and factory strongly: a loaded extension
    # module is never unloaded, so this retention costs nothing, and the
    # strong reference makes the identity compare below sound (an address
    # cannot be reused while it is held). A replaced module or factory entry
    # misses the cache and runs the full admission again.
    var modules: List[PythonObject]
    var factories: List[PythonObject]
    var abis: List[String]
    var snapshots: List[PythonObject]

    def __init__(out self):
        self.modules = List[PythonObject]()
        self.factories = List[PythonObject]()
        self.abis = List[String]()
        self.snapshots = List[PythonObject]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever CompiledBindingSnapshots' layout changes.
comptime COMPILED_BINDING_SNAPSHOTS = SharedSlot[CompiledBindingSnapshots, "MojolearnCompiledBindingSnapshotsV2"]


def compiled_binding_retained(module: PythonObject, factory: PythonObject, abi: String) raises -> PythonObject:
    """The snapshot admitted earlier for this exact module, factory entry and
    ABI name, or None. The caller has read `factory` from the module's live
    namespace in this operation; a replaced entry is a different object and
    misses here."""
    var g = COMPILED_BINDING_SNAPSHOTS.get_or_create_ptr()
    for i in range(len(g[].modules)):  # small-loop(admitted compiled providers): a few extension modules per process, never data
        if g[].modules[i] is module and g[].factories[i] is factory and g[].abis[i] == abi:
            return g[].snapshots[i]
    return PythonObject(None)


def compiled_binding_retain(module: PythonObject, factory: PythonObject, abi: String, snapshot: PythonObject) raises:
    """Record a snapshot that just passed the full admission (mode and
    capability reads) for reuse by later operations."""
    var g = COMPILED_BINDING_SNAPSHOTS.get_or_create_ptr()
    for i in range(len(g[].modules)):  # small-loop(admitted compiled providers): a few extension modules per process, never data
        if g[].modules[i] is module and g[].abis[i] == abi:
            # Same module, replaced factory entry: the new admission replaces it.
            g[].factories[i] = factory
            g[].snapshots[i] = snapshot
            return
    if len(g[].modules) >= 64:
        g[].modules.clear()
        g[].factories.clear()
        g[].abis.clear()
        g[].snapshots.clear()
    g[].modules.append(module)
    g[].factories.append(factory)
    g[].abis.append(abi)
    g[].snapshots.append(snapshot)
