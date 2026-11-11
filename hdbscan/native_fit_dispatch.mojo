# SPDX-License-Identifier: Apache-2.0
"""Typed dispatch into the dedicated HDBSCAN fit/preparation implementations."""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.kmeans_variants_api import VariantsSpan
from core.hdbscan_fit_api import (
    HdbscanFitOptions, HdbscanFitRequest, HdbscanFitState, HdbscanPredictionState,
    HDBSCAN_FIT_MODE, HDBSCAN_FIT_HOST, HDBSCAN_FIT_RUN, HDBSCAN_FIT_PREDICTION,
)


trait HdbscanFitBackend:
    @staticmethod
    def mode() -> Int: ...

    @staticmethod
    def host() -> Bool: ...

    @staticmethod
    def fit(request: HdbscanFitRequest, mut state: HdbscanFitState) raises -> Int: ...

    @staticmethod
    def prediction(mut state: HdbscanPredictionState, rows: Int, edges: Int,
        condensed: Int, clusters: Int, deaths: Int, selected: Int,
        offsets: Int, index_into: Int) raises -> Int: ...


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def hdbscan_fit_dispatch[K: HdbscanFitBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
            raise Error("HDBSCAN fit ABI: missing request or result span")
        var answer = 0
        if operation == HDBSCAN_FIT_MODE or operation == HDBSCAN_FIT_HOST:
            if na != 0 or np != 0 or nf != 0:
                raise Error("HDBSCAN fit ABI: metadata takes no arguments")
            answer = K.mode() if operation == HDBSCAN_FIT_MODE else Int(K.host())
        elif operation == HDBSCAN_FIT_RUN:
            if na != 5 or np != 10 or nf != 2 or _word(aa, 4) == 0:
                raise Error("HDBSCAN fit ABI: invalid fit request")
            var floats = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var o = HdbscanFitOptions(_word(pp, 1), _word(pp, 2), _word(pp, 3),
                _word(pp, 4), _word(pp, 5), _word(pp, 6) != 0, _word(pp, 7),
                _word(pp, 8), _word(pp, 9) != 0, floats[0], floats[1])
            var request = HdbscanFitRequest(VariantsSpan(_word(aa, 0), _word(pp, 0)),
                VariantsSpan(_word(aa, 1), o.rows), VariantsSpan(_word(aa, 2), o.rows),
                VariantsSpan(_word(aa, 3), o.rows), o)
            var state = MutPointer[HdbscanFitState, MutUntrackedOrigin](unsafe_from_address=_word(aa, 4))
            if len(state[].info) != (5 if o.prediction_data else 4):
                raise Error("HDBSCAN fit ABI: info owner does not match prediction-data mode")
            if K.host() and len(state[].host_source) != max(0, o.rows * o.features):
                raise Error("HDBSCAN fit ABI: host source snapshot is not prepared")
            if o.prediction_data and (len(state[].tree_parents) != 2 * o.rows
                    or len(state[].tree_children) != 2 * o.rows or len(state[].tree_lambdas) != 2 * o.rows
                    or len(state[].tree_sizes) != 2 * o.rows or len(state[].inverse) != o.rows):
                raise Error("HDBSCAN fit ABI: tree scratch is not prepared")
            answer = K.fit(request, state[])
        elif operation == HDBSCAN_FIT_PREDICTION:
            if na != 5 or np != 4 or nf != 0 or _word(aa, 0) == 0:
                raise Error("HDBSCAN fit ABI: invalid prediction preparation request")
            var rows = _word(pp, 0)
            var edges = _word(pp, 1)
            var condensed = _word(pp, 2)
            var clusters = _word(pp, 3)
            var state = MutPointer[HdbscanPredictionState, MutUntrackedOrigin](unsafe_from_address=_word(aa, 0))
            if len(state[].labels) != rows or len(state[].parents) != edges or len(state[].children) != edges
                    or len(state[].lambdas) != edges or len(state[].sizes) != edges
                    or len(state[].inverse) != clusters or len(state[].exemplars) != rows:
                raise Error("HDBSCAN fit ABI: prediction snapshot is not prepared")
            answer = K.prediction(state[], rows, edges, condensed, clusters,
                _word(aa, 1), _word(aa, 2), _word(aa, 3), _word(aa, 4))
        else:
            raise Error("HDBSCAN fit ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
