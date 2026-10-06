# SPDX-License-Identifier: Apache-2.0
"""Neural-only host binding for the selected same-version GEMM profile.

OFF delegates to the exact incumbent API. NN03/NN04 select the common Mojo
scalar graph used by neural device forwards and backwards. Classical callers
continue importing gemm_host_rows/identical_gemm and cannot reach this route.
No compilation or verification was run for this integration.
"""
from gemm.host.gemm_oracle import (
    gemm_oracle as _incumbent_oracle,
    gemm_oracle_right_zero_padded as _incumbent_padded,
)
from gemm.host.gemm_host_rows import (
    gemm_host_rows as _incumbent_rows,
    gemm_host_rows_into as _incumbent_rows_into,
    gemm_host_rows_right_zero_padded as _incumbent_rows_padded,
    GhrPtr,GHR_G,GHR_SERIAL_FMAS,ghr_pack_a,ghr_pack_b,ghr_panel_count,ghr_tile,
)
from gemm.contract import GEMM_ORACLE_HOST_SABOTAGE,gemm_oracle_sabotage_value_flip
from gemm.contract import OP_NN,OP_NT,OP_TN
from gemm.experiments.neural_profile import (
    NEURAL_PROFILE_CHANGED,NEURAL_LEAF,NEURAL_CHAINS,
    neural_partition,neural_strides,neural_validate,neural_cell,
)


def _neural_host_cell(a: GhrPtr,b: GhrPtr,row: Int,col: Int,k: Int,leaf: Int,leaves: Int,
                     asi: Int,asp: Int,bsp: Int,bsj: Int,real_k: Int = -1) -> Float32:
    var value = neural_cell[NEURAL_CHAINS](
        a.unsafe_origin_cast[MutAnyOrigin](),b.unsafe_origin_cast[MutAnyOrigin](),
        row,col,k,leaf,leaves,asi,asp,bsp,bsj,real_k)
    comptime if GEMM_ORACLE_HOST_SABOTAGE:
        # Retain a guaranteed observable negative control in the new profile.
        # This output perturbation is a deliberate new-profile sabotage site.
        return gemm_oracle_sabotage_value_flip(value)
    return value


def neural_gemm_host_into(c: GhrPtr,a: GhrPtr,b: GhrPtr,m: Int,n: Int,k: Int,op: Int,
                         real_k: Int = -1) raises:
    neural_validate(m,n,k,op)
    comptime if not NEURAL_PROFILE_CHANGED:
        _incumbent_rows_into(a,b,c,op,m,n,k,False,real_k)
        return
    var part = neural_partition[NEURAL_LEAF](k)
    var st = neural_strides(op,m,n,k)
    for row in range(m):
        for col in range(n):
            c.unsafe_store(row*n+col,_neural_host_cell(a,b,row,col,k,part[0],part[1],st[0],st[1],st[2],st[3],real_k))


def gemm_oracle(a: List[Float32],b: List[Float32],op: Int,m: Int,n: Int,k: Int) -> List[Float32]:
    comptime if not NEURAL_PROFILE_CHANGED:
        return _incumbent_oracle(a,b,op,m,n,k)
    if ((op!=OP_NN and op!=OP_NT and op!=OP_TN) or m<0 or n<0 or k<0
        or len(a)<m*k or len(b)<n*k):
        # Preserve the incumbent invalid-input path rather than turning its
        # checked List access into an unchecked raw-pointer read.
        return _incumbent_oracle(a,b,op,m,n,k)
    var out = List[Float32](length=max(0,m*n),fill=Float32(0))
    var part = neural_partition[NEURAL_LEAF](k)
    var st = neural_strides(op,m,n,k)
    for row in range(m):
        for col in range(n):
            out[row*n+col] = _neural_host_cell(rebind[GhrPtr](a.unsafe_ptr()),rebind[GhrPtr](b.unsafe_ptr()),
                row,col,k,part[0],part[1],st[0],st[1],st[2],st[3])
    return out^


def gemm_host_rows(a: List[Float32],b: List[Float32],op: Int,m: Int,n: Int,k: Int,
                   force_redo: Bool = False) -> List[Float32]:
    comptime if not NEURAL_PROFILE_CHANGED:
        return _incumbent_rows(a,b,op,m,n,k,force_redo)
    return gemm_oracle(a,b,op,m,n,k)


def gemm_host_rows_into(a: GhrPtr,b: GhrPtr,c: GhrPtr,op: Int,m: Int,n: Int,k: Int,
                        force_redo: Bool = False,real_k: Int = -1) raises:
    comptime if not NEURAL_PROFILE_CHANGED:
        _incumbent_rows_into(a,b,c,op,m,n,k,force_redo,real_k)
        return
    if real_k < -1 or real_k > k:
        raise Error("neural host GEMM padded length outside [-1,k]")
    # A logical +0 suffix is generated without reading masked input storage.
    # Every declared FMA is retained, including signed-zero transitions.
    neural_gemm_host_into(c,a,b,m,n,k,op,real_k)


def gemm_oracle_right_zero_padded(a: List[Float32],b: List[Float32],op: Int,
                                  m: Int,n: Int,k: Int,real_k: Int) raises -> List[Float32]:
    if real_k<0 or real_k>k:
        raise Error("gemm_oracle_right_zero_padded: real_k must be in [0,k]")
    comptime if not NEURAL_PROFILE_CHANGED:
        return _incumbent_padded(a,b,op,m,n,k,real_k)
    var out = List[Float32](length=max(0,m*n),fill=Float32(0))
    neural_gemm_host_into(rebind[GhrPtr](out.unsafe_ptr()),rebind[GhrPtr](a.unsafe_ptr()),
        rebind[GhrPtr](b.unsafe_ptr()),m,n,k,op,real_k)
    return out^


def gemm_host_rows_right_zero_padded(a: List[Float32],b: List[Float32],op: Int,
    m: Int,n: Int,k: Int,real_k: Int,force_redo: Bool = False) raises -> List[Float32]:
    comptime if not NEURAL_PROFILE_CHANGED:
        return _incumbent_rows_padded(a,b,op,m,n,k,real_k,force_redo)
    return gemm_oracle_right_zero_padded(a,b,op,m,n,k,real_k)
