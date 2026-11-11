# SPDX-License-Identifier: Apache-2.0
"""Native cases for the linear expansion door's Python-free RUN pieces
(x_linear/door_run.mojo): the fit request words, the [coef | intercept]
block, the joins, slices and part concatenation, the seed halves, the
L-BFGS work size and the SGD class-weight parameters. Analytic answers over
Mojo buffers; independent fixtures run through the shared host worker pool.
The operations-entry calls (xl_run_call / xl_run_fit / xl_run_fit_fold) need
a published x_linear table and are covered by checks/native_tests/x_linear_ops_*;
the cv split RUNs (xl_cv_fold_ids_run / xl_cv_roles_run) need the base helper
table and are covered with the installed wheel's linear CV cases."""
from core.host_parallel import host_parallelize
from x_linear.door_run import (
    xl_run_fit_ints, xl_run_word, xl_run_copy, xl_run_join, xl_run_parts, xl_parts_total, xl_block_dims,
    xl_run_block, xl_seed_halves, xl_lbfgs_work, xl_sgd_class_floats, xl_run_status_refuse, xl_cv_roles_limit,
)

comptime XL_DOOR_FIXTURES = 11


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def fit_ints_fixture() raises:
    var header: List[Int] = [8, 5, 3]
    var ip: List[Int] = [1, 0, 4]
    var words = xl_run_fit_ints(header, ip)
    require(len(words) == 6 and words[0] == 8 and words[2] == 3 and words[3] == 1 and words[5] == 4,
            "fit words are not header then route ints")
    require(len(header) == 3, "the header was modified")


def block_fixture() raises:
    # two classes, three coefficients each
    var coef: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var intercept: List[Float32] = [-1.0, -2.0]
    var dims = xl_block_dims(len(coef), len(intercept))
    require(dims[0] == 2 and dims[1] == 3, "block dims changed")
    var block = List[Float32](length=8, fill=0.0)
    xl_run_block(Int(block.unsafe_ptr()), Int(coef.unsafe_ptr()), Int(intercept.unsafe_ptr()), dims[0], dims[1])
    var expected: List[Float32] = [1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, -2.0]
    for i in range(8):
        require(block[i] == expected[i], "block word " + String(i) + " changed")
    var zero = xl_block_dims(4, 0)
    require(zero[0] == 0 and zero[1] == 4, "an empty intercept must not divide by zero")
    _ = coef^
    _ = intercept^
    _ = block^


def join_fixture() raises:
    var a: List[Float32] = [1.0, 2.0]
    var b: List[Float32] = [3.0, 4.0, 5.0]
    var out = List[Float32](length=5, fill=0.0)
    xl_run_join(Int(out.unsafe_ptr()), Int(a.unsafe_ptr()), 2, Int(b.unsafe_ptr()), 3)
    for i in range(5):
        require(out[i] == Float32(i + 1), "join word " + String(i) + " changed")
    var only = List[Float32](length=2, fill=9.0)
    xl_run_join(Int(only.unsafe_ptr()), Int(a.unsafe_ptr()), 2, 0, 0)
    require(only[0] == 1.0 and only[1] == 2.0, "a join with an empty second part changed")
    _ = a^
    _ = b^
    _ = out^
    _ = only^


def copy_fixture() raises:
    var src: List[Float32] = [0.5, 1.5, 2.5, 3.5]
    var dst = List[Float32](length=2, fill=0.0)
    xl_run_copy(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 1, 2)
    require(dst[0] == 1.5 and dst[1] == 2.5, "slice copy changed")
    xl_run_copy(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 3, 0)
    require(dst[0] == 1.5, "a zero-count slice wrote")
    require(xl_run_word(Int(src.unsafe_ptr()), 3) == 3.5, "word read changed")
    _ = src^
    _ = dst^


def parts_fixture() raises:
    var a: List[Float32] = [1.0]
    var b: List[Float32] = [2.0, 3.0]
    var addrs: List[Int] = [Int(a.unsafe_ptr()), 0, Int(b.unsafe_ptr())]
    var counts: List[Int] = [1, 0, 2]
    require(xl_parts_total(counts) == 3, "parts total changed")
    var out = List[Float32](length=3, fill=0.0)
    xl_run_parts(Int(out.unsafe_ptr()), addrs, counts)
    require(out[0] == 1.0 and out[1] == 2.0 and out[2] == 3.0, "parts concatenation changed")
    _ = a^
    _ = b^
    _ = out^


def seed_fixture() raises:
    var small = xl_seed_halves(42, 0)
    require(small[0] == 42 and small[1] == 0, "a small seed changed")
    var wide = xl_seed_halves(4294967295, 2147483648)
    require(wide[0] == -1 and wide[1] == -2147483648, "the signed halves changed")


def class_floats_fixture(classes: Int) raises:
    var weights: List[Float32] = [0.25, 4.0, 2.0]
    var fp: List[Float64] = [9.0]
    xl_sgd_class_floats(fp, Int(weights.unsafe_ptr()), classes)
    if classes == 2:
        require(len(fp) == 3 and fp[1] == 4.0 and fp[2] == 0.25, "binary class weights changed")
    else:
        require(len(fp) == 7 and fp[1] == 0.25 and fp[3] == 2.0 and fp[4] == 1.0 and fp[6] == 1.0,
                "multiclass class weights changed")
    _ = weights^


def status_fixture() raises:
    xl_run_status_refuse(0, "Ridge")
    var word = List[Int32](length=1, fill=0)
    xl_run_status_refuse(Int(word.unsafe_ptr()), "Ridge")
    word[0] = 3
    var text = String("")
    try:
        xl_run_status_refuse(Int(word.unsafe_ptr()), "Ridge")
    except error:
        text = String(error)
    require(text == "Input X contains NaN or infinity (Ridge).", "finite refusal text changed: " + text)
    _ = word^


def roles_limit_fixture() raises:
    xl_cv_roles_limit(16777215, "RidgeCV")
    var text = String("")
    try:
        xl_cv_roles_limit(16777216, "RidgeCV")
    except error:
        text = String(error)
    require(text == "NotImplementedError: mojolearn RidgeCV: cv splits that are not a partition are limited "
            + "to 16,777,215 rows", "roles limit text changed: " + text)


def fixture(index: Int) raises:
    if index == 0:
        fit_ints_fixture()
    elif index == 1:
        block_fixture()
    elif index == 2:
        join_fixture()
    elif index == 3:
        copy_fixture()
    elif index == 4:
        parts_fixture()
    elif index == 5:
        seed_fixture()
    elif index == 6:
        require(xl_lbfgs_work(3) == 4 * 3 + 2 * 10 * 3 + 2 * 10, "L-BFGS work size changed")
    elif index == 7:
        class_floats_fixture(2)
    elif index == 8:
        class_floats_fixture(3)
    elif index == 9:
        status_fixture()
    else:
        roles_limit_fixture()


def run_checks() raises:
    var failures = List[Int](length=XL_DOOR_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(index: Int) {imm fp}:
        try:
            fixture(index)
        except error:
            fp.unsafe_store(index, 1)
            print("FAIL x_linear_door_run fixture", index, String(error))
    host_parallelize(task, XL_DOOR_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    if failed != 0:
        raise Error("x_linear_door_run: " + String(failed) + " fixture(s) failed")
    print("x_linear_door_run:", XL_DOOR_FIXTURES, "fixtures passed")
