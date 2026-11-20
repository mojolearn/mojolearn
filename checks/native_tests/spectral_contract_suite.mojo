# SPDX-License-Identifier: Apache-2.0
"""Typed Spectral transport with independent prescribed wire/field checks.

Independent cases run across host cores. Sentinel addresses are integers only;
no fake address reaches a numerical kernel or is dereferenced. This suite
checks transport, not numerical admission, fitting or resident ownership.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.spectral_native_api import SpectralInput, SpectralStateView, SpectralFitOptions, SpectralFitRequest, SpectralEmbeddingOptions, SpectralEmbeddingRequest, SpectralQueryShape, SpectralQueryRequest, SpectralPrepareRequest, SpectralResidentQueryRequest, SpectralNativeKernels, SpectralDriver
from core.spectral_native_provider import SpectralProviderAPI, spectral_native_v1

comptime SPECTRAL_CONTRACT_FIXTURES = 58
comptime WIDE_HANDLE = 9007199254740993


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def input(route: Int) -> SpectralInput:
    return SpectralInput(11, 13, 17, 19, route, 120, 2, 431)


def state() -> SpectralStateView:
    return SpectralStateView(31, 37, 41, 43)


def shape(affinity: Int) -> SpectralQueryShape:
    return SpectralQueryShape(120, 7, 2, 4, 3, 8, affinity)


def fitted(route: Int, keep: Bool) -> SpectralFitRequest:
    return SpectralFitRequest(input(route), 23, 29, state(),
        SpectralFitOptions(3, 4, 10, 8, 0.10000000000000002, 4294967295, keep))


def embedded(route: Int, normalize: Bool, drop: Bool) -> SpectralEmbeddingRequest:
    return SpectralEmbeddingRequest(input(route), 29,
        SpectralEmbeddingOptions(4, 3, 8, normalize, drop, 4294967295, 0.10000000000000002))


def check_input(value: SpectralInput) raises:
    require(value.source == 11 and value.graph_rows == 13 and value.graph_columns == 17
        and value.graph_values == 19 and value.route >= 0 and value.route <= 2
        and value.rows == 120 and value.features == 2 and value.nonzeros == 431,
        "Spectral typed source/graph/route/dimension fields")


def check_state(value: SpectralStateView) raises:
    require(value.eigenvalues == 31 and value.eigenvectors == 37 and value.diagonal == 41
        and value.centroids == 43, "Spectral typed eigen/diagonal/centroid addresses")


def check_shape(value: SpectralQueryShape) raises:
    require(value.training_rows == 120 and value.rows == 7 and value.features == 2
        and value.components == 4 and value.clusters == 3 and value.neighbors == 8
        and (value.affinity == 0 or value.affinity == 1), "Spectral typed query dimensions and affinity")


struct PrescribedSpectral(SpectralNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return 0

    @staticmethod
    def capabilities() raises -> Int:
        return 3

    @staticmethod
    def resident_count() raises -> Int:
        return WIDE_HANDLE

    @staticmethod
    def fit(r: SpectralFitRequest) raises -> Int:
        check_input(r.input)
        check_state(r.state)
        require(r.labels == 23 and r.embedding == 29 and r.options.clusters == 3
            and r.options.components == 4 and r.options.initializations == 10
            and r.options.neighbors == 8 and r.options.seed == 4294967295
            and r.options.tolerance == Float64(0.10000000000000002), "Spectral decoded fit fields")
        return WIDE_HANDLE + 2 * r.input.route + Int(r.options.keep_state)

    @staticmethod
    def embed(r: SpectralEmbeddingRequest) raises -> Int:
        check_input(r.input)
        require(r.embedding == 29 and r.options.lanczos_components == 4
            and r.options.output_components == 3 and r.options.neighbors == 8
            and r.options.seed == 4294967295 and r.options.tolerance == Float64(0.10000000000000002),
            "Spectral decoded embedding distinct Lanczos/output widths and tolerance")
        return WIDE_HANDLE + 4 * r.input.route + 2 * Int(r.options.normalize_laplacian) + Int(r.options.drop_first)

    @staticmethod
    def query(r: SpectralQueryRequest) raises -> Int:
        check_state(r.state)
        check_shape(r.shape)
        require(r.source == 47 and r.training == 11 and r.labels == 23 and r.embedding == 29,
            "Spectral decoded query addresses")
        return WIDE_HANDLE + 4 + r.shape.affinity

    @staticmethod
    def prepare(r: SpectralPrepareRequest) raises -> Int:
        check_state(r.state)
        check_shape(r.shape)
        require(r.training == 11, "Spectral decoded resident preparation training address")
        return WIDE_HANDLE + 5 + r.shape.affinity

    @staticmethod
    def query_resident(r: SpectralResidentQueryRequest) raises -> Int:
        check_shape(r.shape)
        require(r.handle == WIDE_HANDLE and r.source == 47 and r.labels == 23 and r.embedding == 29,
            "Spectral decoded exact resident handle and query addresses")
        return WIDE_HANDLE + 6 + r.shape.affinity

    @staticmethod
    def release(handle: Int) raises:
        require(handle == WIDE_HANDLE, "Spectral release retains exact integer above binary64 precision")


def wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
               result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(result != 0, "Spectral owned result slot")
        var value = 0
        if operation == 0 or operation == 1 or operation == 8:
            require(na == 0 and np == 0 and nf == 0 and aa == 0 and pp == 0 and ff == 0,
                "Spectral empty metadata envelope")
            value = 0 if operation == 0 else (3 if operation == 1 else WIDE_HANDLE)
        elif operation == 7:
            require(na == 0 and np == 1 and nf == 0 and pp != 0, "Spectral release envelope")
            require(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[] == WIDE_HANDLE,
                "Spectral exact release handle transport")
        else:
            require(aa != 0 and pp != 0, "Spectral borrowed inline request spans")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if operation == 2 or operation == 3:
                require(na == (10 if operation == 2 else 5) and np == 10 and nf == 1 and ff != 0,
                    "Spectral fit/embed fixed envelope")
                var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
                require(a[0] == 11 and a[1] == 13 and a[2] == 17 and a[3] == 19,
                    "Spectral prescribed source/graph address order")
                require(p[0] >= 0 and p[0] <= 2 and p[1] == 120 and p[2] == 2 and p[3] == 431,
                    "Spectral prescribed route/row/feature/nonzero order")
                require(f[0] == Float64(0.10000000000000002), "Spectral tolerance retains binary64 precision")
                if operation == 2:
                    require(a[4] == 23 and a[5] == 29 and a[6] == 31 and a[7] == 37
                        and a[8] == 41 and a[9] == 43, "Spectral fit output/state order")
                    require(p[4] == 3 and p[5] == 4 and p[6] == 10 and p[7] == 8
                        and p[8] == 4294967295 and (p[9] == 0 or p[9] == 1), "Spectral fit options and Boolean encoding")
                    value = WIDE_HANDLE + 2 * p[0] + p[9]
                else:
                    require(a[4] == 29 and p[4] == 4 and p[5] == 3 and p[6] == 8
                        and (p[7] == 0 or p[7] == 1) and (p[8] == 0 or p[8] == 1)
                        and p[9] == 4294967295, "Spectral embedding widths/flags/seed order")
                    value = WIDE_HANDLE + 4 * p[0] + 2 * p[7] + p[8]
            elif operation == 4 or operation == 5 or operation == 6:
                require(nf == 0 and ff == 0 and np == (8 if operation == 6 else 7)
                    and na == (8 if operation == 4 else (5 if operation == 5 else 3)), "Spectral query/preparation envelope")
                var offset = 1 if operation == 6 else 0
                require(p[offset] == 120 and p[offset+1] == 7 and p[offset+2] == 2
                    and p[offset+3] == 4 and p[offset+4] == 3 and p[offset+5] == 8
                    and (p[offset+6] == 0 or p[offset+6] == 1), "Spectral query dimensions/affinity order")
                if operation == 4:
                    require(a[0] == 47 and a[1] == 11 and a[2] == 31 and a[3] == 37
                        and a[4] == 41 and a[5] == 43 and a[6] == 23 and a[7] == 29, "Spectral direct query pointers")
                elif operation == 5:
                    require(a[0] == 11 and a[1] == 31 and a[2] == 37 and a[3] == 41 and a[4] == 43,
                        "Spectral resident preparation pointers")
                else:
                    require(p[0] == WIDE_HANDLE and a[0] == 47 and a[1] == 23 and a[2] == 29,
                        "Spectral resident exact integer handle and output pointers")
                value = WIDE_HANDLE + operation + p[offset+6]
            else:
                raise Error("Spectral unexpected operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def refused_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                  result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("Spectral prescribed provider refusal"), context, sink)
    return 1


def operation_case(driver: SpectralDriver, operation: Int) raises:
    if operation == 0 or operation == 1 or operation == 8:
        var expected = 0 if operation == 0 else (3 if operation == 1 else WIDE_HANDLE)
        require(driver.metadata(operation) == expected, "Spectral metadata exact result")
    elif operation == 2:
        require(driver.fit(fitted(2, True)) == WIDE_HANDLE + 5, "Spectral provider fit result")
    elif operation == 3:
        require(driver.embed(embedded(1, False, True)) == WIDE_HANDLE + 5, "Spectral provider embedding result")
    elif operation == 4:
        require(driver.query(SpectralQueryRequest(47, 11, state(), 23, 29, shape(0))) == WIDE_HANDLE + 4,
            "Spectral provider direct query result")
    elif operation == 5:
        require(driver.prepare(SpectralPrepareRequest(11, state(), shape(1))) == WIDE_HANDLE + 6,
            "Spectral provider prepare exact integer result")
    elif operation == 6:
        require(driver.query_resident(SpectralResidentQueryRequest(WIDE_HANDLE, 47, 23, 29, shape(0))) == WIDE_HANDLE + 6,
            "Spectral provider resident query result")
    else:
        driver.release(WIDE_HANDLE)


def record_error(context: Int, message: Int, size: Int) abi("C"):
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=context)
    dst[0] += 1
    dst[1] = Int(message != 0 and size > 0)


def malformed(unit: Int) raises:
    var a: Array[Int, 10] = [11,13,17,19,23,29,31,37,41,43]
    var p: Array[Int, 10] = [2,120,2,431,3,4,10,8,4294967295,1]
    var f: Array[Float64, 1] = [0.10000000000000002]
    var result = List[Int](length=1, fill=-73)
    var error = List[Int](length=2, fill=0)
    var operation = 2
    var aa = Int(a.unsafe_ptr())
    var pp = Int(p.unsafe_ptr())
    var ff = Int(f.unsafe_ptr())
    var na = 10
    var np = 10
    var nf = 1
    var rp = Int(result.unsafe_ptr())
    if unit == 0:
        rp = 0
    elif unit == 1:
        operation = 9
    elif unit == 2:
        operation = 1
    elif unit == 3:
        na = 9
    elif unit == 4:
        np = 9
    elif unit == 5:
        nf = 0
    elif unit == 6:
        operation = 3
        na = 5
        pp = 0
    elif unit == 7:
        operation = 4
        na = 8
        np = 7
        nf = 0
        aa = 0
    elif unit == 8:
        operation = 5
        na = 5
        np = 7
    elif unit == 9:
        operation = 6
        na = 3
        np = 8
        nf = 0
        pp = 0
    else:
        operation = 7
        na = 0
        np = 1
        nf = 0
        pp = 0
    try:
        var status = spectral_native_v1[PrescribedSpectral](operation, aa, na, pp, np, ff, nf,
            rp, Int(error.unsafe_ptr()), record_error)
        require(status == 1 and error[0] == 1 and error[1] == 1,
            "Spectral malformed request reports exactly one native error")
        require(result[0] == -73, "Spectral refusal does not publish a result")
    finally:
        _ = a^
        _ = p^
        _ = f^
        _ = result^
        _ = error^


def fixture(unit: Int) raises:
    var driver = SpectralDriver(wire_entry)
    if unit < 3:
        operation_case(driver, unit if unit < 2 else 8)
    elif unit < 9:
        var variant = unit - 3
        require(driver.fit(fitted(variant // 2, variant % 2 != 0)) == WIDE_HANDLE + variant,
            "Spectral all fit route/retained-state variants")
    elif unit < 21:
        var variant = unit - 9
        require(driver.embed(embedded(variant // 4, (variant & 2) != 0, (variant & 1) != 0)) == WIDE_HANDLE + variant,
            "Spectral all embedding route/Laplacian/drop-first variants")
    elif unit < 27:
        var variant = unit - 21
        var affinity = variant % 2
        if variant < 2:
            require(driver.query(SpectralQueryRequest(47,11,state(),23,29,shape(affinity))) == WIDE_HANDLE + 4 + affinity,
                "Spectral direct query affinity transport")
        elif variant < 4:
            require(driver.prepare(SpectralPrepareRequest(11,state(),shape(affinity))) == WIDE_HANDLE + 5 + affinity,
                "Spectral preparation affinity transport")
        else:
            require(driver.query_resident(SpectralResidentQueryRequest(WIDE_HANDLE,47,23,29,shape(affinity))) == WIDE_HANDLE + 6 + affinity,
                "Spectral resident query affinity transport")
    elif unit == 27:
        driver.release(WIDE_HANDLE)
    elif unit < 37:
        var provider = SpectralDriver(spectral_native_v1[PrescribedSpectral])
        operation_case(provider, unit - 28)
    elif unit < 46:
        var rejected = SpectralDriver(refused_entry)
        var did_refuse = False
        try:
            operation_case(rejected, unit - 37)
        except error:
            require(String(error) == "Spectral prescribed provider refusal", "Spectral exact native error propagation")
            did_refuse = True
        require(did_refuse, "Spectral provider refusal lost")
    elif unit < 57:
        malformed(unit - 46)
    else:
        var api = SpectralProviderAPI[PrescribedSpectral]()
        require(api.name == "mojolearn.spectral.native.v1" and len(api.entries) == 1,
            "Spectral named provider table with one fixed entry")
        var provider = SpectralDriver(api.entries[0])
        operation_case(provider, 8)
        _ = api^


def run_checks() raises:
    var failures = List[Int](length=SPECTRAL_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL spectral_contract fixture", unit, String(error))
    host_parallelize(task, SPECTRAL_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "spectral_contract failed " + String(failed) + " fixtures")
    print("PASS spectral_contract:", SPECTRAL_CONTRACT_FIXTURES, "typed request and error fixtures")
