# SPDX-License-Identifier: Apache-2.0
"""The GaussianMixture descriptor (core/mixture_native_api.mojo) over the
unchanged typed bodies. Transport only; no Python."""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_string import unsafe_string_from_utf8_cstr
from core.mixture_native_api import (
    MIXTURE_NATIVE_MODE, MIXTURE_NATIVE_CAPABILITIES, MIXTURE_NATIVE_FIT, MIXTURE_NATIVE_SCORE_SAMPLES,
    MIXTURE_NATIVE_SCORE_BIC_AIC,
)


trait GmmNativeBackend:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def fit(source: Int, rows: Int, features: Int, components: Int, covariance_type: String,
            tol: Float64, reg_covar: Float64, max_iter: Int, init_params: String, random_state: Int,
            weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int, scalars: Int,
            labels: Int) raises -> Int: ...
    @staticmethod
    def score(operation: Int, weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int,
              source: Int, destination: Int, components: Int, features: Int, n_iter: Int, converged: Bool,
              lower_bound: Float64, rows: Int) raises: ...


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def _float(address: Int, index: Int) -> Float64:
    return Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def _name(address: Int) raises -> String:
    if address == 0:
        raise Error("GaussianMixture native ABI: missing option name")
    return unsafe_string_from_utf8_cstr(Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address))


def gmm_native_dispatch[K: GmmNativeBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
            raise Error("GaussianMixture native ABI: missing request or result span")
        var answer = 0
        if operation == MIXTURE_NATIVE_MODE or operation == MIXTURE_NATIVE_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("GaussianMixture native ABI: metadata takes no arguments")
            answer = K.mode() if operation == MIXTURE_NATIVE_MODE else K.capabilities()
        elif operation == MIXTURE_NATIVE_FIT:
            if na != 10 or np != 5 or nf != 2:
                raise Error("GaussianMixture native ABI: invalid fit request")
            # The two names are decoded in the binding's order (covariance
            # type, then init params), as gmm_fit_binding decoded them.
            var covariance_type = _name(_word(aa, 8))
            var init_params = _name(_word(aa, 9))
            answer = K.fit(_word(aa, 0), _word(pp, 0), _word(pp, 1), _word(pp, 2), covariance_type,
                           _float(ff, 0), _float(ff, 1), _word(pp, 3), init_params, _word(pp, 4),
                           _word(aa, 1), _word(aa, 2), _word(aa, 3), _word(aa, 4), _word(aa, 5), _word(aa, 6),
                           _word(aa, 7))
        elif operation >= MIXTURE_NATIVE_SCORE_SAMPLES and operation <= MIXTURE_NATIVE_SCORE_BIC_AIC:
            if na != 7 or np != 5 or nf != 1 or _word(aa, 6) == 0:
                raise Error("GaussianMixture native ABI: invalid scoring request")
            K.score(operation, _word(aa, 0), _word(aa, 1), _word(aa, 2), _word(aa, 3), _word(aa, 4),
                    _word(aa, 5), _word(aa, 6), _word(pp, 0), _word(pp, 1), _word(pp, 2), _word(pp, 3) != 0,
                    _float(ff, 0), _word(pp, 4))
        else:
            raise Error("GaussianMixture native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
