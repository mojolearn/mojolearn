# SPDX-License-Identifier: Apache-2.0
# Resident model and optimizer operations in the shared primitive table. The
# existing public bindings and this table use identical native numerical owners.
# Reservation/validation operations run under the provider admission GIL; no
# Python object or callback enters this execution module.
from core.hostptr import f32_ptr, i32_ptr
from max.gpu.host import DeviceContext
from training.neural_identical_experiments import IDN_SAMBA_RESIDENT_STEP
from training.samba_resident import SambaResidentSession, samba_resident_forward, samba_resident_train_step, samba_resident_accumulated_step
from training.optimizer_public import OPT_POOL, opt_pool_handle, resident_open, resident_close, resident_download, resident_upload, per_call_step, resident_step, resident_step_io


def native_resident_arity(op: Int, na: Int, np: Int, nf: Int) raises:
    var valid = False
    if op == 12:
        valid = na == 1 and np >= 8 and nf == 1
    elif op == 13:
        valid = na == 1 and np == 1 and nf == 0
    elif op == 14 or op == 15:
        valid = na == 0 and np == 1 and nf == 0
    elif op == 16:
        valid = na == 3 and np == 3 and nf == 0
    elif op == 17:
        valid = na == 7 and np == 8 and nf == 8
    elif op == 18:
        valid = na == 7 and np == 10 and nf == 8
    elif op == 19:
        valid = na == 0 and np == 2 and nf == 0
    elif op == 20:
        valid = na == 0 and np == 1 and nf == 0
    elif op == 21:
        valid = na == 0 and np == 1 and nf == 0
    elif op == 22 or op == 23:
        valid = na == 2 and np == 2 and nf == 0
    elif op == 24:
        valid = na == 0 and np == 0 and nf == 0
    elif op == 25:
        valid = na == 7 and np == 5 and nf == 8
    elif op == 26:
        valid = na == 5 and np == 6 and nf == 8
    elif op == 27:
        valid = na == 5 and np == 7 and nf == 8
    elif op == 28:
        valid = na == 0 and np == 0 and nf == 0
    if not valid:
        raise Error("resident primitive ABI: invalid operation or descriptor arity")


def native_resident_operation(ctx: DeviceContext, op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 24:
        return Int(IDN_SAMBA_RESIDENT_STEP)
    # Optimizer registry validation intentionally precedes model reservation.
    if op == 19:
        return opt_pool_handle(p[0], p[1])
    if op == 20:
        return resident_open(ctx, p[0])
    if op == 21:
        return resident_close(ctx, p[0])
    if op == 22:
        return resident_download(ctx, p[0], a[0], a[1], p[1])
    if op == 23:
        return resident_upload(ctx, p[0], a[0], a[1], p[1])
    if op == 25:
        return per_call_step(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
            i32_ptr(a[4]), i32_ptr(a[5]), f32_ptr(a[6]), p[0], p[1], p[2], p[3],
            Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]), p[4] != 0)
    if op == 26:
        return resident_step(ctx, p[0], f32_ptr(a[0]), f32_ptr(a[1]), i32_ptr(a[2]), i32_ptr(a[3]), f32_ptr(a[4]),
            p[1], p[2], p[3], p[4], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]), p[5] != 0)
    if op == 27:
        return resident_step_io(ctx, p[0], p[1], f32_ptr(a[0]), f32_ptr(a[1]), i32_ptr(a[2]), i32_ptr(a[3]), f32_ptr(a[4]),
            p[2], p[3], p[4], p[5], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]), p[6] != 0)
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("Samba resident operation is disabled in this build")
    else:
        if op == 12:
            var kinds = List[Int]()
            for i in range(8, len(p)):  # driver-loop: over admitted model layer kind descriptors
                kinds.append(p[i])
            var session = SambaResidentSession(ctx, kinds, p[0], p[1], p[2], p[3], p[4], p[5], p[6] != 0, Float32(f[0]), p[7], f32_ptr(a[0]))
            var owner = alloc[SambaResidentSession](1)
            owner.init_pointee_move(session^)
            return Int(owner)
        var handle = a[0] if op == 13 else p[0]
        if handle == 0:
            raise Error("Samba resident operation requires a live owner")
        var owner = MutPointer[SambaResidentSession, MutUntrackedOrigin](unsafe_from_address=handle)
        if op == 13:
            if p[0] != 12:
                raise Error("Samba resident resource belongs to another registry")
            if owner[].busy:
                raise Error("Samba resident session busy")
            owner.destroy_pointee()
            owner.free()
            return 0
        if op == 14:
            if owner[].busy:
                raise Error("Samba resident session busy")
            owner[].busy = True
            return 0
        if op == 15:
            owner[].busy = False
            return 0
        if op == 16:
            return samba_resident_forward(owner[], f32_ptr(a[0]), i32_ptr(a[1]), f32_ptr(a[2]), p[1], p[2])
        var pool = OPT_POOL.get_or_create_ptr()
        var h = p[1]
        if op == 17:
            return samba_resident_train_step(owner[], f32_ptr(a[0]), i32_ptr(a[1]), i32_ptr(a[2]), f32_ptr(a[6]),
                pool[].m[h], pool[].v[h], pool[].sp[h], pool[].sg[h], i32_ptr(a[3]), i32_ptr(a[4]), f32_ptr(a[5]),
                p[2], p[3], p[4], p[5], p[6], p[7], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]),
                Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]))
        if op == 18:
            return samba_resident_accumulated_step(owner[], f32_ptr(a[0]), i32_ptr(a[1]), i32_ptr(a[2]), f32_ptr(a[6]),
                pool[].m[h], pool[].v[h], pool[].sp[h], pool[].sg[h], i32_ptr(a[3]), i32_ptr(a[4]), f32_ptr(a[5]),
                p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]),
                Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]))
        raise Error("Samba resident ABI: unknown operation")
