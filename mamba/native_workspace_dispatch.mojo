# SPDX-License-Identifier: Apache-2.0
# Optional private storage resource for ordinary, mutable public Mamba2 step.
from std.ffi import _Global
from std.memory import alloc
from std.atomic import Atomic, Ordering
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_mamba_workspace_abi import MAMBA2_WORKSPACE_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from mamba.impl.mamba_shell_impl import Mamba2PublicWorkspace, _mamba2_workspace_step


struct Mamba2WorkspaceOwner(Movable):
    var busy: List[Int64]
    var workspace: Mamba2PublicWorkspace

    def __init__(out self):
        self.busy = List[Int64](length=1, fill=0)
        self.workspace = Mamba2PublicWorkspace()


def mamba2_workspace_v1(
    op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError,
) abi("C") -> Int:
    try:
        if result == 0:
            raise Error("Mamba2 workspace ABI: null result")
        var output = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if op == 0:
            if na != 0 or np != 0 or nf != 0:
                raise Error("Mamba2 workspace create: unexpected arguments")
            var owner = alloc[Mamba2WorkspaceOwner](1)
            owner.init_pointee_move(Mamba2WorkspaceOwner())
            output[] = Int(owner)
            return 0
        if op != 1 and op != 2:
            raise Error("Mamba2 workspace ABI: unknown operation")
        if na != (1 if op == 1 else 17) or np != (0 if op == 1 else 3) or nf != (0 if op == 1 else 2):
            raise Error("Mamba2 workspace ABI: invalid argument counts")
        var words = native_descriptor_words(aa, na)
        if words[0] == 0:
            raise Error("Mamba2 workspace ABI: null owner")
        # Private, admitted resource: the provider retains the creating image
        # and opaque owner through the complete synchronous dispatch. Close is
        # called only after those per-operation pins have been released.
        var owner = MutPointer[Mamba2WorkspaceOwner, MutUntrackedOrigin](unsafe_from_address=words[0])
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](owner[].busy.unsafe_ptr())
        var expected = Int64(0)
        var claimed = Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL, failure_ordering=Ordering.RELAXED,
        ](lock, expected, Int64(1))
        if op == 1:
            if not claimed:
                raise Error("Mamba2 workspace close: resource is in use")
            try:
                owner[].workspace.clear()
            finally:
                owner.destroy_pointee()
                owner.free()
            output[] = 0
            return 0
        try:
            var a = List[Int](capacity=16)
            for i in range(1, 17):  # small-loop(sixteen admitted native public-step addresses)
                if words[i] == 0:
                    raise Error("Mamba2 workspace step: null buffer address")
                a.append(words[i])
            var p = native_descriptor_words(pp, np)
            if ff == 0:
                raise Error("Mamba2 workspace step: null scalar span")
            var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            if claimed:
                try:
                    output[] = _mamba2_workspace_step(owner[].workspace, a, p[0], p[1], p[2], Float32(f[0]), Float32(f[1]))
                except problem:
                    try:
                        owner[].workspace.clear()
                    except:
                        pass  # Storage is dropped in clear's finally; preserve the operation error.
                    raise problem^
            else:
                # No waiting/reentrant alias to mutable cached device storage.
                # A second call owns its own buffers and refreshes live bytes.
                var temporary = Mamba2PublicWorkspace()
                try:
                    output[] = _mamba2_workspace_step(temporary, a, p[0], p[1], p[2], Float32(f[0]), Float32(f[1]))
                except problem:
                    try:
                        temporary.clear()
                    except:
                        pass  # Preserve the operation error after invalidating temporary storage.
                    raise problem^
                temporary.clear()
            return 0
        finally:
            if claimed:
                Atomic.store[ordering=Ordering.RELEASE](lock, Int64(0))
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeMamba2WorkspaceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA2_WORKSPACE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(mamba2_workspace_v1)


comptime NATIVE_MAMBA2_WORKSPACE_API = _Global[StorageType=NativeMamba2WorkspaceAPI,
    name="MojoMamba2PublicWorkspaceAPIv1", init_fn=NativeMamba2WorkspaceAPI.__init__]
