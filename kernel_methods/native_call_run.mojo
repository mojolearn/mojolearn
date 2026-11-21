# SPDX-License-Identifier: Apache-2.0
# RUN of the kernel methods door (bindings/kernel_methods_native_door.mojo):
# typed requests for the named operations of the kernel methods provider
# table (kernel_methods/descriptor_schema.mojo) and the one call that runs
# them. No Python object and no Python import: UNPACK
# (bindings/kernel_methods_boundary_unpack.mojo) reads every scalar and pins
# every buffer, PACK (bindings/kernel_methods_boundary_pack.mojo) allocates
# the public arrays, and this file receives addresses and words only.
#
# A request is the three vectors the provider entry takes: addresses, integer
# words and Float64 words, the scalar slots split by the operation's mask
# exactly as the packed admission split them (a masked slot is read from the
# Float64 vector, any other from the integer vector; the unread half is 0).
from core.neural_descriptor_run import neural_descriptor_record_admit, neural_descriptor_record_call
from core.scale_gamma_exact import scale_gamma_from_limbs, SGX_BAD
from kernel_methods.descriptor_schema import kernel_method_scalar_mask

comptime KM_OP_KRR_FIT = 0
comptime KM_OP_KRR_PREDICT = 1
comptime KM_OP_KRR_FIT_PTR = 2
comptime KM_OP_KRR_PREDICT_PTR = 3
comptime KM_OP_KRR_RESIDENT_PREPARE = 4
comptime KM_OP_KRR_PREDICT_RESIDENT = 6
comptime KM_OP_NYS_FIT_TRANSFORM_PTR = 7
comptime KM_OP_NYS_FIT = 8
comptime KM_OP_NYS_FIT_PTR = 9
comptime KM_OP_NYS_TRANSFORM = 10
comptime KM_OP_RBF_FIT = 11
comptime KM_OP_RBF_TRANSFORM = 12
comptime KM_OP_SCALE_GAMMA_LIMBS = 16
# A request only a list entry takes (no provider operation).
comptime KM_OP_LISTED_ONLY = -1


@fieldwise_init
struct KmKernel(ImplicitlyCopyable, Movable):
    """The kernel words of one operation: the kernel code, degree, gamma and
    coef0, read once by UNPACK."""
    var kernel: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64


struct KmCall(Movable):
    """One typed request. `mask` is the provider schema (the slots the engine
    reads as Float64); `listed` is the set of slots a list entry of a module
    without the provider receives as floats (the same set, plus a slot the
    schema declares integer but the list entry was always handed as a
    float)."""
    var operation: Int
    var mask: Int
    var listed: Int
    var a: List[Int]
    var p: List[Int]
    var f: List[Float64]

    def __init__(out self, operation: Int, count: Int, mask: Int):
        self.operation = operation
        self.mask = mask
        self.listed = mask
        self.a = List[Int]()
        self.p = List[Int](length=count, fill=0)
        self.f = List[Float64](length=count, fill=0.0)

    def address(mut self, value: Int):
        self.a.append(value)

    def word(mut self, slot: Int, value: Int) raises:
        if (self.mask & (1 << slot)) != 0:
            raise Error("kernel methods provider: packed admission does not match operation")
        self.p[slot] = value

    def real(mut self, slot: Int, value: Float64) raises:
        if (self.mask & (1 << slot)) == 0:
            raise Error("kernel methods provider: packed admission does not match operation")
        self.f[slot] = value

    def real_listed(mut self, slot: Int, value: Float64):
        """A real the schema of this operation may declare integer (the
        ridge alpha of the pointer and resident predicts, which no predict
        reads): the Float64 word where the schema is real, its integer part
        otherwise; a list entry receives the float either way."""
        self.listed = self.listed | (1 << slot)
        self.f[slot] = value
        if (self.mask & (1 << slot)) == 0:
            self.p[slot] = Int(value)


def km_call(operation: Int) raises -> KmCall:
    """An empty request with the scalar layout of `operation`."""
    var schema = kernel_method_scalar_mask(operation)
    return KmCall(operation, schema[0], schema[1])


def km_call_admit(provider_id: Int, call: KmCall) raises:
    """The packed admission's size test, then the table's own admission
    (refuses an operation outside the table and counts the attempt). Under
    the caller's lock, before `km_call_run`."""
    var buffers = len(call.a)
    if call.operation == KM_OP_KRR_PREDICT_RESIDENT:
        buffers -= 1  # the resident handle leads the addresses
    if call.operation < 0 or buffers > 8:
        raise Error("kernel methods provider: packed admission does not match operation")
    neural_descriptor_record_admit(provider_id, call.operation)


def km_call_run(provider_id: Int, call: KmCall) raises -> Int:
    """One synchronous provider call. Safe with the GIL released."""
    return neural_descriptor_record_call(provider_id, call.operation, call.a, call.p, call.f)


def _km_kernel_words(mut call: KmCall, first: Int, kernel: KmKernel) raises:
    call.word(first, kernel.kernel)
    call.word(first + 1, kernel.degree)
    call.real(first + 2, kernel.gamma)
    call.real(first + 3, kernel.coef0)


def km_krr_fit_call(pointer: Bool, x: Int, y: Int, dual: Int, scalars: Int,
                    n: Int, d: Int, t: Int, kernel: KmKernel, alpha: Float64) raises -> KmCall:
    """ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_fit_binding:
    x, y, dual_out, scalars_out[, sw]; n, d, t, kernel, degree, gamma, coef0,
    alpha. The pointer entry (kernel_ridge_fit_ptr_binding) takes nine
    addresses: the caller appends sw (0: none), x_fit_out (0: none) and the
    fold's store ids (-1, 0, -1: none)."""
    var call = km_call(KM_OP_KRR_FIT_PTR if pointer else KM_OP_KRR_FIT)
    call.address(x)
    call.address(y)
    call.address(dual)
    call.address(scalars)
    call.word(0, n)
    call.word(1, d)
    call.word(2, t)
    _km_kernel_words(call, 3, kernel)
    call.real(7, alpha)
    return call^


def km_krr_predict_call(operation: Int, n: Int, d: Int, t: Int, kernel: KmKernel, alpha: Float64,
                        info: Int, q: Int) raises -> KmCall:
    """The ten words of every KernelRidge predict (kernel_ridge_predict_binding):
    n, d, t, kernel, degree, gamma, coef0, alpha, info, q. The caller appends
    the addresses of its entry: x_fit, dual, x_new, out (per call; the
    pointer entry's fold adds its store ids after a 0 x_new), or the resident
    handle, x_new, out."""
    var call = km_call(operation)
    call.word(0, n)
    call.word(1, d)
    call.word(2, t)
    _km_kernel_words(call, 3, kernel)
    call.real_listed(7, alpha)
    call.word(8, info)
    call.word(9, q)
    return call^


def km_krr_prepare_call(x_fit: Int, dual: Int, n: Int, d: Int, t: Int) raises -> KmCall:
    """kernel_ridge_resident_prepare: x_fit, dual; n, d, t. Answers the handle."""
    var call = km_call(KM_OP_KRR_RESIDENT_PREPARE)
    call.address(x_fit)
    call.address(dual)
    call.word(0, n)
    call.word(1, d)
    call.word(2, t)
    return call^


def km_nys_fit_call(operation: Int, x: Int, components: Int, indices: Int, normalization: Int,
                    eigenvalues: Int, eigenvectors: Int, scalars: Int,
                    n: Int, d: Int, kernel: KmKernel, q: Int, seed: Int) raises -> KmCall:
    """ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_fit_binding:
    x, components_out, indices_out, normalization_out, eigenvalues_out,
    eigenvectors_out, scalars_out; n, d, kernel, degree, gamma, coef0, q,
    seed. nystroem_fit_transform_ptr_binding: the caller appends the n x q
    transform. Answers the sweep count."""
    var call = km_call(operation)
    call.address(x)
    call.address(components)
    call.address(indices)
    call.address(normalization)
    call.address(eigenvalues)
    call.address(eigenvectors)
    call.address(scalars)
    call.word(0, n)
    call.word(1, d)
    _km_kernel_words(call, 2, kernel)
    call.word(6, q)
    call.word(7, seed)
    return call^


def km_nys_transform_call(components: Int, indices: Int, normalization: Int, eigenvalues: Int,
                          eigenvectors: Int, x: Int, dst: Int,
                          q: Int, d: Int, kernel: KmKernel, seed: Int, sweeps: Int, m: Int) raises -> KmCall:
    """ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_transform_binding:
    components, indices, normalization, eigenvalues, eigenvectors, x, out;
    q, d, kernel, degree, gamma, coef0, seed, sweeps, m."""
    var call = km_call(KM_OP_NYS_TRANSFORM)
    call.address(components)
    call.address(indices)
    call.address(normalization)
    call.address(eigenvalues)
    call.address(eigenvectors)
    call.address(x)
    call.address(dst)
    call.word(0, q)
    call.word(1, d)
    _km_kernel_words(call, 2, kernel)
    call.word(6, seed)
    call.word(7, sweeps)
    call.word(8, m)
    return call^


def km_rbf_fit_call(weights: Int, offset: Int, scalars: Int, d: Int, q: Int, gamma: Float64,
                    seed: Int) raises -> KmCall:
    """ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_fit_binding:
    weights_out, offset_out, scalars_out (sigma, scale); d, q, gamma, seed."""
    var call = km_call(KM_OP_RBF_FIT)
    call.address(weights)
    call.address(offset)
    call.address(scalars)
    call.word(0, d)
    call.word(1, q)
    call.real(2, gamma)
    call.word(3, seed)
    return call^


def km_rbf_transform_call(weights: Int, offset: Int, x: Int, dst: Int, d: Int, q: Int, gamma: Float64,
                          seed: Int, sigma: Float64, scale: Float64, m: Int) raises -> KmCall:
    """ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_transform_binding:
    weights, offset, x, out; d, q, gamma, seed, sigma, scale, m."""
    var call = km_call(KM_OP_RBF_TRANSFORM)
    call.address(weights)
    call.address(offset)
    call.address(x)
    call.address(dst)
    call.word(0, d)
    call.word(1, q)
    call.real(2, gamma)
    call.word(3, seed)
    call.real(4, sigma)
    call.real(5, scale)
    call.word(6, m)
    return call^


def km_rbf_joint_call(weights: Int, offset: Int, scalars: Int, x: Int, dst: Int, d: Int, q: Int,
                      gamma: Float64, seed: Int, m: Int) raises -> KmCall:
    """ORDER MATCHES kernel_methods/rbf_resident.mojo::rbf_sampler_fit_transform_binding
    (lane apple-fast-w2-kfeat MOJOLEARN_KM_FAST_RBF_RESIDENT): weights_out,
    offset_out, scalars_out, x, out; d, q, gamma, seed, m. A list entry only."""
    var call = KmCall(KM_OP_LISTED_ONLY, 5, 4)
    call.address(weights)
    call.address(offset)
    call.address(scalars)
    call.address(x)
    call.address(dst)
    call.word(0, d)
    call.word(1, q)
    call.real(2, gamma)
    call.word(3, seed)
    call.word(4, m)
    return call^


def km_scale_gamma_words() -> List[Int64]:
    """The binding's fixed limb words of gamma='scale', owned by RUN."""
    return List[Int64](length=SGX_BAD + 1, fill=Int64(0))


def km_scale_gamma_run(provider_id: Int, x: Int, cells: Int, mut words: List[Int64]) raises:
    """The provider's exact limbs of the float32 cells of X into `words`.
    Safe with the GIL released (the caller admitted operation 16)."""
    var a: List[Int] = [x, Int(words.unsafe_ptr())]
    var p: List[Int] = [cells]
    var f: List[Float64] = [0.0]
    _ = neural_descriptor_record_call(provider_id, KM_OP_SCALE_GAMMA_LIMBS, a, p, f)
    _ = words[0]  # the provider wrote through the address above


def km_scale_gamma_value(words: List[Int64], cells: Int, d: Int) raises -> Float64:
    """`_scale_gamma.scale_gamma_x`: the one rounding, in exact integer math."""
    return scale_gamma_from_limbs(words, cells, d)
