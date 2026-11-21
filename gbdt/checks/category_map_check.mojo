# SPDX-License-Identifier: Apache-2.0
"""Native check of the category encoding (`gbdt/data/category_map.mojo`):
keys, the distinct set against a serial reference, dense codes, the unseen
code, the stored map searched in place, and the new cell kinds (pandas
Categorical codes, NumPy U / S cells, numeric text). Host only; no device."""
from gbdt.data.category_map import (
    CategoryMaps,
    CategoryKeyPtr,
    CategoryF32Ptr,
    CATEGORY_KEY_MISSING,
    category_key_integer,
    category_key_f32,
    category_key_f64,
    category_key_text,
    category_keys_from_f32,
    category_distinct_sorted,
    category_code_of,
    category_block_columns,
    category_block_feature,
    category_block_count,
    category_apply_column,
    category_keys_from_codes,
    category_values_from_codes,
    category_text_number,
    category_fixed_text,
    CATEGORY_TEXT_OK,
    CATEGORY_TEXT_INVALID,
    CATEGORY_TEXT_NOT_A_NUMBER,
)


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("category map check: " + what)


def _key_ptr(values: List[UInt64]) -> CategoryKeyPtr:
    return CategoryKeyPtr(unsafe_from_address=Int(values.unsafe_ptr()))


def _f32_ptr(values: List[Float32]) -> CategoryF32Ptr:
    return CategoryF32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def keys() raises:
    _need(category_key_f32(Float32(5.0)) == category_key_integer(5), "5.0 and 5 are one category")
    _need(category_key_f64(Float64(5.0)) == category_key_integer(5), "float 5.0 and int 5 are one category")
    _need(category_key_f32(Float32(-0.0)) == category_key_integer(0), "-0.0 and 0 are one category")
    _need(category_key_f32(Float32(-3.0)) < category_key_f32(Float32(2.0)), "whole keys keep numeric order")
    _need(category_key_f32(Float32(1.5)) != category_key_f32(Float32(1.0)), "1.5 is not category 1")
    _need(category_key_f64(Float64(1.5)) == category_key_f32(Float32(1.5)), "a fractional float keys as its Float32")
    var zero = Float32(0.0)
    _need(category_key_f32(zero / zero) == CATEGORY_KEY_MISSING, "NaN is the missing category")
    var site = String("site-a")
    var other = String("site-b")
    var site_bytes = site.as_bytes()
    var other_bytes = other.as_bytes()
    var a = category_key_text(Int(site_bytes.unsafe_ptr()), site.byte_length())
    var b = category_key_text(Int(other_bytes.unsafe_ptr()), other.byte_length())
    _need(a != b, "two texts are two categories")
    _need((a >> UInt64(63)) == UInt64(1), "a text key carries the text tag")
    _need((category_key_integer(-7) >> UInt64(62)) == UInt64(0), "a whole key carries no tag")
    _ = site
    _ = other


def _reference_distinct(values: List[UInt64]) -> List[UInt64]:
    """Serial insertion into an ascending distinct list: the oracle."""
    var out = List[UInt64]()
    for i in range(len(values)):
        var v = values[i]
        var pos = 0
        while pos < len(out) and out[pos] < v:
            pos += 1
        if pos < len(out) and out[pos] == v:
            continue
        out.append(v)
        var j = len(out) - 1
        while j > pos:
            out[j] = out[j - 1]
            j -= 1
        out[pos] = v
    return out^


def distinct(n: Int, modulus: UInt64) raises:
    var values = List[UInt64](capacity=n)
    var state = UInt64(0x9E3779B97F4A7C15)
    for _ in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        values.append((state >> UInt64(20)) % modulus)
    var got = category_distinct_sorted(_key_ptr(values), n)
    var want = _reference_distinct(values)
    _need(len(got) == len(want), "distinct count for n=" + String(n))
    for i in range(len(want)):
        _need(got[i] == want[i], "distinct key " + String(i) + " for n=" + String(n))
    for i in range(n):
        var code = category_code_of(_key_ptr(got), len(got), values[i])
        _need(code < len(got) and got[code] == values[i], "every fitted key has its rank")
    _need(category_code_of(_key_ptr(got), len(got), modulus + UInt64(1)) == len(got),
          "an unseen key takes the unseen code")
    _ = values


def distinct_wide(n: Int) raises:
    """Many tasks, nearly every key distinct: checked by its properties (the
    serial oracle is quadratic there)."""
    var values = List[UInt64](capacity=n)
    var state = UInt64(0x2545F4914F6CDD1D)
    for _ in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        values.append(state >> UInt64(12))
    var got = category_distinct_sorted(_key_ptr(values), n)
    _need(len(got) >= 1 and len(got) <= n, "wide distinct count")
    for i in range(1, len(got)):
        _need(got[i - 1] < got[i], "wide distinct keys ascend strictly")
    for i in range(n):
        var code = category_code_of(_key_ptr(got), len(got), values[i])
        _need(code < len(got) and got[code] == values[i], "every wide key has its rank")
    _ = values


def fit_and_apply() raises:
    # two categorical columns of a 3-column block, column-major, 6 rows
    var n = 6
    var x: List[Float32] = [
        40.0, 7.0, 40.0, -2.0, 7.0, 1.5,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        9.0, 9.0, 8.0, 8.0, 9.0, 8.0,
    ]
    var xp = _f32_ptr(x)
    var scratch = List[UInt64](length=n, fill=UInt64(0))
    var maps = CategoryMaps()
    category_keys_from_f32(xp, n, _key_ptr(scratch))
    maps.fit_column(0, _key_ptr(scratch), n, xp)
    category_keys_from_f32(xp + 2 * n, n, _key_ptr(scratch))
    maps.fit_column(2, _key_ptr(scratch), n, xp + 2 * n)
    # ascending keys: -2, 7, 40 (whole numbers in numeric order), then 1.5 (real tag)
    var want0: List[Float32] = [2.0, 1.0, 2.0, 0.0, 1.0, 3.0]
    var want2: List[Float32] = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    var numeric: List[Float32] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for r in range(n):
        _need(x[r] == want0[r], "column 0 code of row " + String(r))
        _need(x[2 * n + r] == want2[r], "column 2 code of row " + String(r))
        _need(x[n + r] == numeric[r], "the numeric column is not written")
    var words = maps.block_words()
    _need(words == 2 + 4 + 4 + 2, "stored map word count")
    var block = List[UInt64](length=words, fill=UInt64(0))
    maps.write_block(_key_ptr(block))
    _need(category_block_columns(_key_ptr(block), words, 3) == 2, "stored map column count")
    _need(category_block_feature(_key_ptr(block), 1) == 2, "stored map feature index")
    _need(category_block_count(_key_ptr(block), 0) == 4, "stored map column size")
    # apply: a fitted value, the same value as a whole float, and an unseen one
    var fresh: List[Float32] = [7.0, 40.0, 123.0, -2.0]
    var fresh_keys = List[UInt64](length=4, fill=UInt64(0))
    category_keys_from_f32(_f32_ptr(fresh), 4, _key_ptr(fresh_keys))
    category_apply_column(_key_ptr(block), 0, _key_ptr(fresh_keys), 4, _f32_ptr(fresh))
    var want_fresh: List[Float32] = [1.0, 2.0, 4.0, 0.0]
    for r in range(4):
        _need(fresh[r] == want_fresh[r], "applied code of row " + String(r))
    var refused = False
    try:
        _ = category_block_columns(_key_ptr(block), words - 1, 3)
    except:
        refused = True
    _need(refused, "a truncated stored map is refused")
    refused = False
    try:
        _ = category_block_columns(_key_ptr(block), words, 2)
    except:
        refused = True
    _need(refused, "a stored map naming a column past the model width is refused")
    _ = x
    _ = scratch
    _ = block
    _ = fresh
    _ = fresh_keys


def _text_key(text: String) -> UInt64:
    var bytes = text.as_bytes()
    var key = category_key_text(Int(bytes.unsafe_ptr()), text.byte_length())
    _ = text
    return key


def codes() raises:
    """A Categorical's codes mapped through its k category keys, in several
    tasks: every code width, -1 is missing, a corrupt code is refused."""
    var n = 100003
    var table: List[UInt64] = [_text_key("a"), _text_key("b"), category_key_integer(7)]
    var numbers: List[Float32] = [1.5, -2.0, 7.0]
    var c8 = List[Int8](length=n, fill=Int8(0))
    var c64 = List[Int64](length=n, fill=Int64(0))
    for r in range(n):
        var code = (r % 4) - 1  # -1, 0, 1, 2
        c8[r] = Int8(code)
        c64[r] = Int64(code)
    var got = List[UInt64](length=n, fill=UInt64(0))
    var wide = List[UInt64](length=n, fill=UInt64(0))
    var vals = List[Float32](length=n, fill=Float32(0.0))
    category_keys_from_codes(Int(c8.unsafe_ptr()), 1, 1, n, _key_ptr(table), 3, _key_ptr(got))
    category_keys_from_codes(Int(c64.unsafe_ptr()), 8, 8, n, _key_ptr(table), 3, _key_ptr(wide))
    category_values_from_codes(Int(c8.unsafe_ptr()), 1, 1, n, _f32_ptr(numbers), 3, _f32_ptr(vals))
    for r in range(n):
        var code = (r % 4) - 1
        var want = CATEGORY_KEY_MISSING if code < 0 else table[code]
        _need(got[r] == want and wide[r] == want, "a code maps to its category's key")
        if code < 0:
            _need(vals[r] != vals[r], "code -1 of a numeric feature is NaN")
        else:
            _need(vals[r] == numbers[code], "a code maps to its category's number")
    c8[n - 1] = Int8(3)
    var refused = False
    try:
        category_keys_from_codes(Int(c8.unsafe_ptr()), 1, 1, n, _key_ptr(table), 3, _key_ptr(got))
    except:
        refused = True
    _need(refused, "a code past the categories is refused")
    _ = table
    _ = numbers
    _ = c8
    _ = c64
    _ = got
    _ = wide
    _ = vals


def _number(text: String, mut ok: Bool) -> Float64:
    var bytes = text.as_bytes()
    var value = category_text_number(Int(bytes.unsafe_ptr()), text.byte_length(), ok)
    _ = text
    return value


def numbers() raises:
    var ok = False
    _need(_number(" 1.5 ", ok) == 1.5 and ok, "text 1.5 with spaces")
    _need(_number("-2e3", ok) == -2000.0 and ok, "exponent text")
    _need(_number("7", ok) == 7.0 and ok, "whole text")
    var nan = _number("NaN", ok)
    _need(ok and nan != nan, "nan text")
    _need(_number("-inf", ok) < -1.0e300 and ok, "-inf text")
    _need(_number("Infinity", ok) > 1.0e300 and ok, "infinity text")
    _ = _number("abc", ok)
    _need(not ok, "letters are not a number")
    _ = _number("", ok)
    _need(not ok, "empty text is not a number")
    _ = _number("1.5x", ok)
    _need(not ok, "trailing letters are not a number")


def fixed_text() raises:
    """A U array (UCS4) and an S array: the categorical column's keys equal
    the str / bytes keys of the same text, trailing NULs dropped; the
    numeric column parsed; a bad number and a surrogate refused with their
    cell. Row-major rows x 2 columns, column 0 categorical."""
    var n = 5000
    var chars = 4  # U4: 16-byte cells
    var u = List[UInt32](length=n * 2 * chars, fill=UInt32(0))
    var s8 = List[UInt8](length=n * 2 * chars, fill=UInt8(0))
    var texts: List[String] = ["a", "bc", "été", "1.5"]
    for r in range(n):
        var cat = texts[r % 3].copy()
        var i = 0
        for cp in cat.codepoints():
            u[(r * 2) * chars + i] = UInt32(Int(cp))
            i += 1
        var cb = cat.as_bytes()
        for b in range(min(cat.byte_length(), chars)):
            s8[(r * 2) * chars + b] = cb[b]
        # the numeric column: "1.5" (ASCII, so the same bytes in U and S)
        var num = texts[3].as_bytes()
        for b in range(3):
            u[(r * 2 + 1) * chars + b] = UInt32(num[b])
            s8[(r * 2 + 1) * chars + b] = num[b]
    var slots: List[Int] = [0, -1]
    var ku = List[UInt64](length=n, fill=UInt64(0))
    var ks = List[UInt64](length=n, fill=UInt64(0))
    var vu = List[Float32](length=2 * n, fill=Float32(0.0))
    var vs = List[Float32](length=2 * n, fill=Float32(0.0))
    var fu = category_fixed_text(Int(u.unsafe_ptr()), 4 * chars, True, 2 * 4 * chars, 4 * chars, n, 2,
                                 Int(slots.unsafe_ptr()), _key_ptr(ku), _f32_ptr(vu))
    var fs = category_fixed_text(Int(s8.unsafe_ptr()), chars, False, 2 * chars, chars, n, 2,
                                 Int(slots.unsafe_ptr()), _key_ptr(ks), _f32_ptr(vs))
    _need(fu[2] == CATEGORY_TEXT_OK and fs[2] == CATEGORY_TEXT_OK, "clean U and S arrays are admitted")
    for r in range(n):
        var want = _text_key(texts[r % 3])
        _need(ku[r] == want, "a U cell keys as the str of its text")
        if r % 3 != 2:
            _need(ks[r] == want, "an S cell keys as the bytes of its text")
        _need(vu[n + r] == Float32(1.5) and vs[n + r] == Float32(1.5), "the numeric column is parsed")
    # a letter in the numeric column of row 4321, a surrogate in row 77
    u[(4321 * 2 + 1) * chars] = UInt32(120)
    var bad = category_fixed_text(Int(u.unsafe_ptr()), 4 * chars, True, 2 * 4 * chars, 4 * chars, n, 2,
                                  Int(slots.unsafe_ptr()), _key_ptr(ku), _f32_ptr(vu))
    _need(bad[2] == CATEGORY_TEXT_NOT_A_NUMBER and bad[0] == 4321 and bad[1] == 1, "a text number is refused at its cell")
    u[(77 * 2) * chars] = UInt32(0xD800)
    var worse = category_fixed_text(Int(u.unsafe_ptr()), 4 * chars, True, 2 * 4 * chars, 4 * chars, n, 2,
                                    Int(slots.unsafe_ptr()), _key_ptr(ku), _f32_ptr(vu))
    _need(worse[2] == CATEGORY_TEXT_INVALID and worse[0] == 77 and worse[1] == 0, "the first refused cell in row order wins")
    _ = u
    _ = s8
    _ = texts
    _ = slots
    _ = ku
    _ = ks
    _ = vu
    _ = vs


def run_checks() raises:
    keys()
    # one task, several tasks, all distinct, heavy repeats
    distinct(1, UInt64(7))
    distinct(257, UInt64(5))
    distinct(4099, UInt64(1) << UInt64(40))
    distinct(300000, UInt64(1000))
    distinct_wide(2000000)
    fit_and_apply()
    codes()
    numbers()
    fixed_text()
