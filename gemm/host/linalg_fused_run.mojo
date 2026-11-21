# SPDX-License-Identifier: Apache-2.0
"""Operation 15 of the host linalg descriptor table (gemm/host/linalg_dispatch.mojo).
`gemm_fused` is the Apple GPU fast epilogue arm (MOJOLEARN_AFN_GEMM_EPILOGUE,
gemm/linalg_fused_run.mojo); the host image has no such arm and refuses it in
the device binding's words. Typed Mojo only, no Python import."""


def gemm_fused_native(addrs: List[Int], params: List[Int]) raises -> Int:
    raise Error("gemm_fused: this binding was built without MOJOLEARN_AFN_GEMM_EPILOGUE")
