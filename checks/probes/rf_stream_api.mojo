# SPDX-License-Identifier: Apache-2.0
"""Probe real runtime stream capabilities, without simulating GPU overlap.

Build/run under tools/with_build_lock.sh pixi run mojo run <this file>.
Metal is an explicit unsupported result; on CUDA, verifies whether creating
DeviceStream also registers a selectable DeviceContext stream view.
"""
from max.gpu.host import DeviceContext

def main() raises:
    var ctx = DeviceContext()
    print("api", ctx.api(), "initial_streams", ctx.num_streams())
    if ctx.api() != "cuda":
        print("UNSUPPORTED: RF CUDA stream probe requires CUDA")
        return
    var initial = ctx.num_streams()
    var created = ctx.create_stream()
    var after = ctx.num_streams()
    print("after_create", after)
    if after <= initial:
        created.synchronize()
        raise Error("create_stream did not expose a new selectable DeviceContext view")
    var other = ctx.select_stream(after - 1)
    other.enqueue_wait_for(ctx)
    other.synchronize()
    created.synchronize()
    print("PASS: newly created stream is selectable; kernel ownership check still required")
