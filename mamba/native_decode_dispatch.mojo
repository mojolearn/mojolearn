# SPDX-License-Identifier: Apache-2.0
# Three resident Mamba families share the existing block descriptor table and
# common provider reservation/GIL lifecycle. Numerical helpers are unchanged.
from mamba.impl.mamba_shell_impl import (
    Mamba1DecodeSession,
    _m1_require_session_open,
    _m1_session_open_run,
    _m1_session_step_run,
    _m1_session_export_run,
    _m1_session_load_run,
    Mamba2DecodeSession,
    _m2_require_session_open,
    _m2_session_open_run,
    _m2_session_step_run,
    _m2_session_export_run,
    _m2_session_load_run,
    Mamba3DecodeSession,
    _m3_require_session_open,
    _m3_session_open_run,
    _m3_session_step_run,
    _m3_session_export_run,
    _m3_session_copy_in,
)


def mamba_decode_operation(kind: Int, op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if len(words) == 0 or words[0] == 0:
        raise Error("Mamba decode: missing native session owner")
    var p = List[Int]()
    for i in range(1, len(words)):  # driver-loop: over admitted decode scalar words
        p.append(words[i])
    if kind == 1:
        var owner = MutPointer[Mamba1DecodeSession, MutUntrackedOrigin](unsafe_from_address=words[0])
        if op == 0 or op == 1 or op == 2 or op == 7:
            if len(p) != 0 or len(a) != 0 or len(f) != 0:
                raise Error("Mamba decode lifecycle: invalid descriptor")
            if op == 7:
                owner[].close()
                return 0
            if op == 2:
                owner[].busy = False
                return 0
            if op == 0:
                if owner[].busy or owner[].is_open():
                    raise Error("mamba1 session: busy or already open")
            else:
                _m1_require_session_open(owner[])
            owner[].busy = True
            return 0
        if not owner[].busy:
            raise Error("Mamba decode execution requires a reserved owner")
        var valid = (op == 3 and len(a) == 12 and len(p) == 2 and len(f) == 0) or (op == 4 and len(a) == 2 and len(p) == 0 and len(f) == 0) or (op == 5 and len(a) == 2 and len(p) == 0 and len(f) == 0) or (op == 6 and len(a) == 2 and len(p) == 0 and len(f) == 0)
        if not valid:
            raise Error("mamba1 decode: invalid operation or descriptor arity")
        for address in a:  # driver-loop: over fixed admitted decode buffer addresses
            if address == 0:
                raise Error("mamba1 decode: null buffer address")
        if op == 3:
            if p[0] <= 0:
                raise Error("mamba1 session open: positive shape required")
            owner[].usable = False
            try:
                var expanded: List[Int] = [0]
                for address in a:  # driver-loop: existing Mamba1 open address layout
                    expanded.append(address)
                _m1_session_open_run(owner[], expanded, p[0], p[1], 0)
            except problem:
                owner[].release()
                raise problem
            owner[].usable = True
            return 0
        try:
            if op == 4:
                _m1_session_step_run(owner[], a[0], a[1])
                return 0
            if op == 5:
                _m1_session_export_run(owner[], a[0], a[1])
                return 0
            _m1_session_load_run(owner[], a[0], a[1])
            return 0
        except problem:
            owner[].usable = False
            raise problem
    elif kind == 2:
        var owner = MutPointer[Mamba2DecodeSession, MutUntrackedOrigin](unsafe_from_address=words[0])
        if op == 0 or op == 1 or op == 2 or op == 7:
            if len(p) != 0 or len(a) != 0 or len(f) != 0:
                raise Error("Mamba decode lifecycle: invalid descriptor")
            if op == 7:
                owner[].close()
                return 0
            if op == 2:
                owner[].busy = False
                return 0
            if op == 0:
                if owner[].busy or owner[].is_open():
                    raise Error("mamba2 session: busy or already open")
            else:
                _m2_require_session_open(owner[])
            owner[].busy = True
            return 0
        if not owner[].busy:
            raise Error("Mamba decode execution requires a reserved owner")
        var valid = (op == 3 and len(a) == 13 and len(p) == 3 and len(f) == 2) or (op == 4 and len(a) == 3 and len(p) == 0 and len(f) == 0) or (op == 5 and len(a) == 4 and len(p) == 0 and len(f) == 0) or (op == 6 and len(a) == 4 and len(p) == 1 and len(f) == 0)
        if not valid:
            raise Error("mamba2 decode: invalid operation or descriptor arity")
        for address in a:  # driver-loop: over fixed admitted decode buffer addresses
            if address == 0:
                raise Error("mamba2 decode: null buffer address")
        if op == 3:
            if p[0] <= 0 or p[1] <= 0:
                raise Error("mamba2 session open: positive shape required")
            owner[].usable = False
            try:
                _m2_session_open_run(owner[], a, p[0], p[1], p[2], Float32(f[0]), Float32(f[1]))
            except problem:
                owner[].release()
                raise problem
            owner[].usable = True
            return 0
        try:
            if op == 4:
                return _m2_session_step_run(owner[], a[0], a[1], a[2])
            if op == 5:
                return _m2_session_export_run(owner[], a)
            _m2_session_load_run(owner[], a, p[0])
            return 0
        except problem:
            owner[].usable = False
            raise problem
    elif kind == 3:
        var owner = MutPointer[Mamba3DecodeSession, MutUntrackedOrigin](unsafe_from_address=words[0])
        if op == 0 or op == 1 or op == 2 or op == 7:
            if len(p) != 0 or len(a) != 0 or len(f) != 0:
                raise Error("Mamba decode lifecycle: invalid descriptor")
            if op == 7:
                owner[].close()
                return 0
            if op == 2:
                owner[].busy = False
                return 0
            if op == 0:
                if owner[].busy or owner[].is_open():
                    raise Error("mamba3 session: busy or already open")
            else:
                _m3_require_session_open(owner[])
            owner[].busy = True
            return 0
        if not owner[].busy:
            raise Error("Mamba decode execution requires a reserved owner")
        var valid = (op == 3 and len(a) == 19 and len(p) == 4 and len(f) == 0) or (op == 4 and len(a) == 6 and len(p) == 0 and len(f) == 0) or (op == 5 and len(a) == 10 and len(p) == 0 and len(f) == 0) or (op == 6 and len(a) == 10 and len(p) == 2 and len(f) == 0)
        if not valid:
            raise Error("mamba3 decode: invalid operation or descriptor arity")
        for address in a:  # driver-loop: over fixed admitted decode buffer addresses
            if address == 0:
                raise Error("mamba3 decode: null buffer address")
        if op == 3:
            if p[0] <= 0 or p[1] <= 0:
                raise Error("mamba3 session open: positive shape required")
            owner[].usable = False
            try:
                _m3_session_open_run(owner[], a, p[0], p[1], p[2], p[3])
            except problem:
                owner[].release()
                raise problem
            owner[].usable = True
            return 0
        try:
            if op == 4:
                return _m3_session_step_run(owner[], a)
            if op == 5:
                var q = _m3_session_export_run(owner[], a)
                return 2 * q + Int(owner[].state.value().pending)
            _m3_session_copy_in(owner[], a, p[0], p[1])
            return 0
        except problem:
            owner[].usable = False
            raise problem
    raise Error("Mamba decode: unknown family")
