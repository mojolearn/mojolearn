# SPDX-License-Identifier: Apache-2.0
"""PCS typed plans, scalar admission and prescribed native descriptor fields.

Independent contracts run in parallel. Fake addresses are compared as integer
fields only; they never reach kernels or pointer dereferences.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neighbors_native_driver import XnDriver
from core.pcs_native_api import PCSOptions, PCSFitPlan, PCSNativeDriver, pcs_options, pcs_seed_word, pcs_feature_width
from x_neighbors.descriptor_schema import xn_operation_id

comptime PCS_CONTRACT_FIXTURES = 34


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
               result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(result != 0, "PCS native result room")
        var value = 0
        if operation == xn_operation_id("capabilities"):
            require(na == 0 and np == 0 and nf == 0, "PCS capability has empty spans")
            value = 24
        else:
            require(aa != 0 and pp != 0, "PCS owned request spans")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if operation == xn_operation_id("pcs_draw"):
                require(na == 2 and np == 3 and nf == 0, "PCS draw envelope")
                require(a[0] == 13 and a[1] == 17 and p[0] == 4294967295 and p[1] == 32 and p[2] == 21,
                    "PCS seed/component/hash-count fields")
            elif operation == xn_operation_id("pcs") or operation == xn_operation_id("pcs_resident"):
                require(na == 4 and np == 5 and nf == 2 and ff != 0, "PCS query envelope")
                var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
                var old = operation == xn_operation_id("pcs")
                require(a[0] == (11 if old else 23) and a[1] == 13 and a[2] == 17 and a[3] == 19,
                    "PCS route and address fields")
                require(p[0] == 60 and p[1] == 6 and p[2] == 7 and p[3] == 32 and p[4] == 3,
                    "PCS row/feature/augmented/component/degree fields")
                require(f[0] == Float64(0.10000000149011612) and f[1] == Float64(0),
                    "PCS query Float32 scalar round trip after Float64 augmentation")
            else:
                raise Error("PCS unexpected native operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def missing_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                  result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 7
    return 0


def rejected_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("PCS wire rejection"), context, sink)
    return 1


def fixture(unit: Int) raises:
    if unit < 5:
        var degree = 1 if unit == 0 else (3 if unit == 1 else (7 if unit == 2 else (1 if unit == 3 else 2)))
        var components = 1 if unit == 0 else (32 if unit == 1 else (17 if unit == 2 else (4097 if unit == 3 else 1)))
        var options = pcs_options(degree, components)
        require(options.degree == degree and options.components == components, "PCS admitted exact positive sizes")
    elif unit < 9:
        var refused = False
        try:
            _ = pcs_options(0 if unit == 5 else (-1 if unit == 6 else 3),
                0 if unit == 7 else (-1 if unit == 8 else 32))
        except error:
            require(String(error) == "degree and n_components must be >= 1", "PCS size refusal")
            refused = True
        require(refused, "PCS nonpositive size admitted")
    elif unit < 17:
        var features = 0 if unit < 13 else 6
        var variant = (unit - 9) % 4
        var coefficient = Float64(0) if variant == 0 else (Float64(-0.0) if variant == 1 else (Float64(1.0) if variant == 2 else Float64(1e-50)))
        var augmented = features if variant < 2 else features + 1
        var plan = PCSFitPlan(pcs_options(3, 32), features, coefficient)
        require(plan.features == features and plan.augmented == augmented, "PCS unrounded augmentation decision")
        var hash_shape = plan.hash_shape()
        var output_shape = plan.output_shape(0 if features == 0 else 60)
        require(len(hash_shape) == 2 and hash_shape[0] == 3 and hash_shape[1] == augmented, "PCS rank-two hash shape")
        require(len(output_shape) == 2 and output_shape[0] == (0 if features == 0 else 60) and output_shape[1] == 32,
            "PCS rank-two output shape including empty rows")
        var draw = plan.draw(7, 13, 17)
        require(draw.total == 3 * augmented and draw.seed == 7 and draw.components == 32 and draw.indices == 13 and draw.signs == 17,
            "PCS fitted draw count and borrowed storage")
    elif unit < 23:
        var raw = UInt64(0)
        var expected = 0
        if unit == 18:
            raw = 1
            expected = 1
        elif unit == 19:
            raw = 0xFFFFFFFF
            expected = 4294967295
        elif unit == 20:
            raw = 0x100000000
        elif unit == 21:
            raw = 0x12345678ABCDEF01
            expected = 2882400001
        elif unit == 22:
            raw = 0xFFFFFFFFFFFFFFFF
            expected = 4294967295
        require(pcs_seed_word(raw) == expected, "PCS seed low-word mask")
    elif unit < 27:
        var plan = PCSFitPlan(pcs_options(3, 32), 6, 1.0)
        if unit == 23:
            plan.admit_hash_widths(7, 7)
            pcs_feature_width(6, 6)
        else:
            var refused = False
            try:
                if unit == 26:
                    pcs_feature_width(7, 6)
                else:
                    plan.admit_hash_widths(6 if unit == 24 else 7, 6)
            except error:
                var expected = String("Number of features of test samples does not match that of training samples.") if unit == 26 else String("ValueError: PolynomialCountSketch hash tables do not match the fitted features")
                require(String(error) == expected, "PCS fitted-width refusal")
                refused = True
            require(refused, "PCS mismatched widths admitted")
    elif unit < 31:
        var driver = PCSNativeDriver(XnDriver(wire_entry))
        var plan = PCSFitPlan(pcs_options(3, 32), 6, 1e-50)
        if unit == 27:
            driver.draw(plan.draw(4294967295, 13, 17))
        elif unit == 30:
            driver.require_draw()
        else:
            var old = unit == 28
            driver.query(plan.query(11 if old else 23, 13, 17, 19, 60, 0.1, 1e-50, old))
    elif unit < 33:
        var refused = False
        try:
            if unit == 31:
                PCSNativeDriver(XnDriver(missing_entry)).require_draw()
            else:
                PCSNativeDriver(XnDriver(rejected_entry)).require_draw()
        except error:
            var expected = String("PCS wire rejection") if unit == 32 else String("NotImplementedError: mojolearn PolynomialCountSketch: this binary lacks x_neighbors_kfeat_pcs_draw_idn (rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")
            require(String(error) == expected, "PCS capability/native error propagation")
            refused = True
        require(refused, "PCS missing provider capability admitted")
    else:
        var first = PCSFitPlan(pcs_options(3, 32), 6, 1.0)
        var second = first.copy()
        second.options.degree = 5
        second.features = 9
        second.augmented = 10
        require(first.options.degree == 3 and first.features == 6 and first.augmented == 7,
            "PCS copied plan has independent typed fields")


def run_checks() raises:
    var failures = List[Int](length=PCS_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL pcs_contract fixture", unit, String(error))
    host_parallelize(task, PCS_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pcs_contract failed " + String(failed) + " fixtures")
    print("PASS pcs_contract:", PCS_CONTRACT_FIXTURES, "native typed plan/request/refusal fixtures")
