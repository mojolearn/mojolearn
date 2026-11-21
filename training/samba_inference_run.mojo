# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba inference door (lane rs-neural, wave 8 run split,
from bindings/samba_inference_native.mojo): the Mamba-3 carried-state piece
shapes and names (the host route's primitive entry is held by the stack's
device route, bindings/samba_route_unpack.mojo). Typed Mojo only, no
Python import (tools/python_free_dirs.sh). Same bodies."""


def _m3_shapes(b: Int, h: Int) -> List[List[Int]]:
    var result = List[List[Int]]()
    result.append([b, h, 32])
    result.append([b, h, 64, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 64])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, h, 128])
    result.append([b, h, 64])
    return result^


def _m3_names() -> List[String]:
    return [String("theta"), String("h"), String("buffer_qrot"), String("buffer_krot"),
            String("buffer_v"), String("buffer_dt"), String("buffer_sig"), String("buffer_adt"),
            String("pending_k"), String("pending_v")]


