"""Compare a transformer identity trace card's stage dumps against the corpus.

WHAT THIS IS. `transformer/corpus/` (see its README.md) carries a float64
per-stage reference for a Llama-shaped decoder block, computed from
HuggingFace's `modeling_llama.py` and NOT from anything in this repository.
This tool reads a directory of raw little-endian float32 stage dumps that the
lane's driver wrote, one file `<stage>.f32` per contract-section-9 stage, and
compares each against that reference at a tolerance CALIBRATED PER CASE PER
STAGE from the corpus's own self test.

WHAT IT CLAIMS. Only this: a dump that passes is consistent with somebody
else's transformer at FP32-roundoff scale. That is the ONLY claim in this lane
that is not our code agreeing with our code. It is not a bitwise certificate,
it says nothing about cross-vendor identity, and it cannot see a summation
ORDER (reordering a float64 sum moves it by about 1e-16 relative, four orders
below any tolerance an FP32 dump can honestly be held to). Contract sections
5.3, 5.4 and 7.2's order clauses belong to the lane's own sabotages; the
corpus generator's `--verify` control 9 prints exactly which clauses this
instrument cannot reach.

WHAT WOULD FALSIFY IT. `--negative-control`, which corrupts the corpus's own
ref32 arm in named ways and demands each corruption be CAUGHT. A checker with
no negative control passes for ever. Every corruption that this instrument
does NOT catch on a given case is printed as a thing that case CANNOT CERTIFY,
which is how the mamba lane learned that its `base_b1_l1_d8` could not
certify a token-major to channel-major reindexing at all, because at L=1 the
two layouts are the same bytes.

FOUR THINGS THIS TOOL REFUSES TO CONFUSE.

  1. MISSING is not DIFFERENT. `cmp -s a b` reports "differ" when b does not
     exist; the mamba lane read an absent dump as a total divergence once.
     Every absent file is named as MISSING and every short file as WRONG SIZE.
  2. A TOLERANCE PASS IS NOT A SIGNED-ZERO PASS. `isclose` treats +0.0 and
     -0.0 as equal. Where the reference is exactly +-0.0 the dump's sign bit
     must match, and that check is separate and always on.
  3. AN FP32 OVERFLOW IS NOT A DEFECT. Contract 4.1(b) says a masked cell at
     an extreme score overflows to -inf in FP32 by construction. An infinity
     of the right sign where the reference exceeds the float32 maximum is a
     PASS, counted and reported, never silently absorbed.
  4. A DEFINITION MISMATCH IS NOT AN ARITHMETIC DEFECT. When a stage fails by
     a clean explainable term the tool NAMES the term rather than printing
     FAIL and stopping. The mamba corpus's `dt_proj.out` differed from the
     lane's by exactly its bias; the right move was to report the naming
     disagreement, not to fold the bias in and turn the check green, which
     would have broken the contract's stage split to satisfy a name.

Usage:
    python tools/transformer_corpus_check.py <case_dir> <dump_dir> [--rtol R] [--atol A]
    python tools/transformer_corpus_check.py <case_dir> --self-test
    python tools/transformer_corpus_check.py <case_dir> --negative-control

Exit codes: 0 every present stage passed; 1 a stage failed; 2 nothing was
compared; 3 an INPUT did not match bitwise, so no output difference can be
attributed to arithmetic; 4 a negative control was not caught.

Only numpy is needed, so this runs in the repository's pixi envs. torch is
required only by the generator.

NOTHING IN THIS FILE HAS BEEN EXECUTED BY ITS AUTHOR. Every statement about
behavior is CONSTRUCTION, not measurement.
"""

import argparse
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# THIS BLOCK IS DUPLICATED, ON PURPOSE, from transformer/corpus/gen_corpus.py,
# which imports torch and therefore cannot be imported here. The two copies
# MUST agree. COMPARE_VERSION is written into every case manifest and this
# tool REFUSES a manifest whose version it does not recognize, so a silent
# divergence between the copies cannot happen.
# --------------------------------------------------------------------------
COMPARE_VERSION = "transformer-corpus-compare-v1"
FP32_MAX = 3.4028234663852886e38
FP32_EPS = 1.1920928955078125e-07
RTOL_LADDER = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
DEFAULT_ATOL = 1e-6


def compare(dump, ref, rtol, atol):
    """Element comparison of an FP32 dump against a float64 reference.

    Passes an element when |dump - ref| <= atol + rtol*|ref|, or when it is an
    explained FP32 overflow (dump infinite, same sign, |ref| above the float32
    maximum). Signed zeros are compared by SIGN BIT separately; a sign-bit
    mismatch clears `ok` while leaving `tol_ok` set, so the two never get
    confused for one another."""
    d = np.asarray(dump, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    close = np.isclose(d, r, rtol=rtol, atol=atol, equal_nan=True)
    overflow = np.isinf(d) & (np.abs(r) > FP32_MAX) & (np.signbit(d) == np.signbit(r))
    ok_elem = close | overflow
    finite = np.isfinite(r) & np.isfinite(d)
    diff = np.where(finite, np.abs(d - r), 0.0)
    max_abs = float(diff.max()) if diff.size else 0.0
    nz = finite & (r != 0)
    max_rel = float((diff[nz] / np.abs(r[nz])).max()) if nz.any() else 0.0
    ref_zero = (r == 0)
    zero_sign_bad = ref_zero & (np.signbit(d) != np.signbit(r))
    tol_ok = bool(ok_elem.all())
    return dict(
        ok=tol_ok and not bool(zero_sign_bad.any()),
        tol_ok=tol_ok,
        max_abs=max_abs, max_rel=max_rel, n=int(d.size),
        n_overflow=int(overflow.sum()), n_ref_zero=int(ref_zero.sum()),
        n_zero_sign_bad=int(zero_sign_bad.sum()),
        n_ref_nan=int(np.isnan(r).sum()),
        first_bad=(None if tol_ok else int(np.argmin(ok_elem))),
        first_zero_sign_bad=(None if not zero_sign_bad.any() else int(np.argmax(zero_sign_bad))),
    )


def smallest_passing_rtol(dump, ref, atol):
    for rt in RTOL_LADDER:
        if compare(dump, ref, rt, atol)["tol_ok"]:
            return rt
    return None


# --------------------------------------------------------------------------
# Loading, with MISSING and WRONG SIZE kept apart from DIFFERENT
# --------------------------------------------------------------------------
MISSING, WRONG_SIZE, PRESENT = "MISSING", "WRONG SIZE", "PRESENT"


def load_manifest(case_dir):
    p = os.path.join(case_dir, "manifest.json")
    if not os.path.isfile(p):
        sys.exit(f"error: {p} not found (pass ONE corpus case directory, not the corpus root)")
    with open(p) as fh:
        m = json.load(fh)
    if m.get("compare_version") != COMPARE_VERSION:
        sys.exit(f"error: {p} was written by compare_version {m.get('compare_version')!r} and "
                 f"this tool is {COMPARE_VERSION!r}. The comparison logic is duplicated between "
                 f"gen_corpus.py and this file on purpose; a version skew means the two copies "
                 f"have drifted and the numbers below would not mean what they say. Regenerate "
                 f"the corpus or fix the copies.")
    return m


def read_raw(path, dtype, n_expected):
    """Returns (status, array_or_None, n_found). MISSING and WRONG SIZE are
    distinct outcomes and neither is ever reported as a value difference."""
    if not os.path.isfile(path):
        return MISSING, None, 0
    a = np.fromfile(path, dtype=dtype)
    if a.size != n_expected:
        return WRONG_SIZE, a, int(a.size)
    return PRESENT, a, int(a.size)


# --------------------------------------------------------------------------
# The explainers. A stage that fails by a clean term gets that term NAMED.
# --------------------------------------------------------------------------
def _named_factor(c, m):
    """Name a constant factor if it is one of the block's own quantities."""
    known = {
        "d_model": float(m["d_model"]),
        "1/d_model": 1.0 / m["d_model"],
        "head_dim": float(m["head_dim"]),
        "1/head_dim": 1.0 / m["head_dim"],
        "intermediate_size": float(m["intermediate_size"]),
        "attention scale": float(m["attention_scale"]),
        "1/attention scale": 1.0 / float(m["attention_scale"]),
        "n_heads": float(m["n_heads"]),
        "n_rep": float(m["n_rep"]),
        "-1 (a sign flip)": -1.0,
        "2": 2.0, "0.5": 0.5,
    }
    for name, v in known.items():
        if v != 0 and abs(c - v) <= 1e-6 * abs(v):
            return name
    return None


def explain(dump, ref, shape, m, rtol, atol):
    """Try to name the difference. Returns a list of human sentences.

    This exists because of the mamba corpus's `dt_proj.out`: the corpus
    included a bias the contract excludes, the difference was EXACTLY the
    bias, and reporting it as a naming disagreement was right where folding
    the bias in to turn the check green would have broken the contract's own
    stage split. A stage that differs by a clean, explainable term is
    reportable AS THAT."""
    out = []
    d = np.asarray(dump, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    fin = np.isfinite(d) & np.isfinite(r)
    if not fin.any():
        return ["every compared element is nonfinite; no term can be named"]
    df, rf = d[fin], r[fin]
    tol = atol + rtol * np.abs(rf)

    if np.allclose(df, -rf, rtol=rtol, atol=atol):
        out.append("the dump is the NEGATION of the reference, elementwise")

    b = float(np.median(df - rf))
    if b != 0 and np.all(np.abs(df - rf - b) <= tol):
        out.append(f"the dump differs from the reference by a CONSTANT OFFSET {b!r} "
                   f"everywhere; a missing or extra additive constant, not a rounding")

    nzr = rf != 0
    if nzr.any():
        c = float(np.median(df[nzr] / rf[nzr]))
        if c != 0 and np.all(np.abs(df - c * rf) <= tol * max(1.0, abs(c))):
            nm = _named_factor(c, m)
            out.append(f"the dump is the reference times a CONSTANT FACTOR {c!r}"
                       + (f", which is {nm}" if nm else "")
                       + ". A definition mismatch, not an arithmetic defect. The mamba lane's "
                         "norm sumsq-versus-mean trap has exactly this shape.")

    if len(shape) >= 2:
        rows = int(np.prod(shape[:-1]))
        cols = int(shape[-1])
        dm = np.asarray(dump, dtype=np.float64).reshape(rows, cols)
        rm = np.asarray(ref, dtype=np.float64).reshape(rows, cols)
        delta = dm - rm
        okd = np.isfinite(delta)
        if okd.all() and rows > 1:
            col = np.median(delta, axis=0)
            if np.all(np.abs(delta - col[None, :]) <= (atol + rtol * np.abs(rm))):
                out.append("the difference is a PER-COLUMN ADDITIVE VECTOR, constant down "
                           "every row. That is the shape of an included or omitted BIAS. "
                           "First values: " + repr(col[:min(6, cols)].tolist()))
            row = np.median(delta, axis=1)
            if np.all(np.abs(delta - row[:, None]) <= (atol + rtol * np.abs(rm))):
                out.append("the difference is a PER-ROW ADDITIVE VECTOR, constant across every "
                           "column. That is the shape of a per-token or per-head constant.")
        if len(shape) >= 2 and shape[-1] == shape[-2]:
            dt = np.asarray(dump, dtype=np.float64).reshape(shape)
            rt = np.asarray(ref, dtype=np.float64).reshape(shape)
            if np.allclose(np.swapaxes(dt, -1, -2), rt, rtol=rtol, atol=atol, equal_nan=True):
                out.append("the dump is the reference with its LAST TWO AXES TRANSPOSED; "
                           "this is a layout defect, not an arithmetic one")

    if df.size == rf.size and df.size > 1:
        sd, sr = np.sort(df), np.sort(rf)
        if not np.allclose(df, rf, rtol=rtol, atol=atol) and \
                np.allclose(sd, sr, rtol=rtol, atol=atol):
            out.append("the dump holds the SAME MULTISET OF VALUES in a DIFFERENT ORDER. A "
                       "permutation or an index map, not arithmetic. This is the failure that "
                       "hashed input values exist to make visible.")
    return out


# --------------------------------------------------------------------------
# Input verification, which runs BEFORE any output is looked at
# --------------------------------------------------------------------------
def check_inputs(case_dir, dump_dir, m):
    """Compare every input tensor the dump carries against the corpus BITWISE.

    This is first and it is not optional. When both sides produce byte
    identical inputs, an output disagreement is about ARITHMETIC. Without it,
    a difference could be about inputs and no amount of staring at stage
    tolerances will ever tell you which. The hash spec in the corpus README
    exists so a Mojo fixture can regenerate these bytes independently; this
    check is what makes that spec worth writing."""
    print("== inputs, compared BITWISE (not by tolerance) ==")
    names = sorted(m["tensors"])
    present, differ, missing, wrong = [], [], [], []
    for name in names:
        info = m["tensors"][name]
        n = int(np.prod(info["shape"]))
        dpath = os.path.join(dump_dir, f"{name}.f32")
        st, a, found = read_raw(dpath, "<f4", n)
        if st is MISSING:
            missing.append(name)
            continue
        if st is WRONG_SIZE:
            wrong.append((name, found, n))
            continue
        ref = np.fromfile(os.path.join(case_dir, info["file"]), dtype="<f4")
        same = a.tobytes() == ref.tobytes()
        present.append(name)
        if not same:
            nd = int((a.view(np.uint32) != ref.view(np.uint32)).sum())
            i = int(np.argmax(a.view(np.uint32) != ref.view(np.uint32)))
            differ.append((name, nd, n, i, float(a[i]), float(ref[i])))
    pos_note = ""
    ppath = os.path.join(dump_dir, "positions.i32")
    if os.path.isfile(ppath):
        got = np.fromfile(ppath, dtype="<i4").tolist()
        want = list(m["positions"])
        pos_note = f"  positions {'MATCH' if got == want else f'DIFFER got={got} want={want}'}"
    print(f"  {len(present)} tensor(s) compared, {len(differ)} DIFFERENT, "
          f"{len(missing)} MISSING, {len(wrong)} WRONG SIZE{pos_note}")
    if missing:
        print(f"  MISSING (not compared, NOT counted as a difference): {', '.join(missing)}")
    for name, found, want in wrong:
        print(f"  WRONG SIZE {name}: dump holds {found} float32, manifest says {want}")
    for name, nd, n, i, dv, rv in differ:
        print(f"  DIFFERENT {name}: {nd}/{n} elements differ in BITS; first at flat {i}, "
              f"dump={dv!r} corpus={rv!r}")
    bad = bool(differ or wrong)
    if bad:
        print("  the inputs are not the same bytes, so NO output difference below can be")
        print("  attributed to arithmetic. Fix the fixture against the corpus hash spec first.")
    if not present:
        print("  no input tensors in the dump; the arithmetic attribution above is UNPROVEN")
    return bad


def check_rope_anchor(case_dir, dump_dir, m):
    """Verify the rotary anchor before believing any rope.cos comparison.

    RoPE's angle is `position * inv_freq`, so a one-ulp difference in inv_freq
    becomes a position-ulp difference in the angle and `d(cos)/d(angle) = 1`.
    At the profile's ceiling that is an amplification of about 8000x. The
    corpus's rope.cos and rope.sin references are anchored to the reference's
    FP32 inv_freq bits (`ref32/rope.inv_freq.f32`), so if a dump's inv_freq
    differs at all, the cos and sin comparisons measure the anchor and not the
    cosine, and this tool says CONDITIONAL rather than PASS or FAIL."""
    info = m["stages"].get("rope.inv_freq")
    if info is None:
        return None
    apath = os.path.join(case_dir, m["rope_anchor"]["anchor_file"])
    dpath = os.path.join(dump_dir, "rope.inv_freq.f32")
    n = int(np.prod(info["shape"]))
    st, a, found = read_raw(dpath, "<f4", n)
    if st is not PRESENT:
        print(f"== rotary anchor ==\n  rope.inv_freq {st}; rope.cos and rope.sin are "
              f"CONDITIONAL (max|angle| in this case is "
              f"{m['rope_anchor']['max_abs_angle']:.4f})")
        return "unverified"
    anchor = np.fromfile(apath, dtype="<f4")
    same = a.tobytes() == anchor.tobytes()
    ulps = np.abs(a.astype(np.float64) - anchor.astype(np.float64)) / \
        np.maximum(np.spacing(np.abs(anchor.astype(np.float32))).astype(np.float64), 1e-300)
    print("== rotary anchor ==")
    print(f"  rope.inv_freq vs ref32 anchor: {'BITWISE EQUAL' if same else 'DIFFERENT'}; "
          f"max ulp distance {float(ulps.max()):.2f}; max|angle| in this case "
          f"{m['rope_anchor']['max_abs_angle']:.4f}; float32 eps {FP32_EPS:.3e}")
    if not same:
        amp = float(ulps.max()) * m["rope_anchor"]["max_abs_angle"]
        print(f"  the angle inherits about {amp:.3e} radians of that, and d(cos)/d(angle) = 1, "
              f"so rope.cos and rope.sin below are CONDITIONAL: they measure the anchor, not "
              f"the cosine. This is a finding about inv_freq (contract S6, portable_powf, "
              f"DEVIATION 809), not about S8.")
    return "equal" if same else "different"


# --------------------------------------------------------------------------
# The stage check
# --------------------------------------------------------------------------
def select_rope_rows(a, info, m, stage):
    """Accept a rotary table dumped over a FULL position range.

    The card's shape for rope.cos is [P_max, head_dim/2]; this corpus writes
    only the positions a case uses (DEVIATION 1041), because a full table at
    the far-position case would be 8192 rows of decoration. A dump that
    carries the full table is accepted by selecting the case's rows, and the
    tool SAYS it did so rather than quietly reshaping."""
    half = info["shape"][1]
    if a.size % half:
        return None, None
    rows = a.size // half
    pos = list(m["positions"])
    if rows == len(pos):
        return a.reshape(info["shape"]), None
    if rows >= max(pos) + 1:
        sel = a.reshape(rows, half)[np.asarray(pos, dtype=np.int64)]
        return sel, (f"dump carries {rows} table rows; selected rows {pos[0]}..{pos[-1]} "
                     f"by absolute position")
    return None, None


def run_check(case_dir, dump_dir, rtol_override, atol, allow_input_mismatch):
    if not os.path.isdir(dump_dir):
        sys.exit(f"error: dump directory {dump_dir} does not exist. That is a MISSING "
                 f"directory, which is not the same thing as a wrong answer, and this tool "
                 f"will not report it as one.")
    m = load_manifest(case_dir)
    print(f"case {m['name']}  B={m['B']} L={m['L']} pos0={m['pos0']} "
          f"d_model={m['d_model']} n_heads={m['n_heads']} n_kv={m['n_kv_heads']} "
          f"head_dim={m['head_dim']} inter={m['intermediate_size']} n_rep={m['n_rep']}")
    print(f"  corpus {m['corpus']}  profile {m['profile']}")
    if m.get("expect_lane_failure"):
        print("  NOTE: the corpus author PREDICTED the lane would fail this case. A failure "
              "here is expected information, not a surprise; read `cannot_certify` below.")
    inputs_bad = check_inputs(case_dir, dump_dir, m)
    if inputs_bad and not allow_input_mismatch:
        print("\nEXIT 3: inputs differ. Pass --inputs-differ-ok to proceed anyway, and if you "
              "do, treat every number below as uninterpretable.")
        return 3
    anchor = check_rope_anchor(case_dir, dump_dir, m)

    print("\n== stages ==")
    print(f"  {'stage':16s} {'verdict':10s} {'max_abs':>10s} {'max_rel':>10s} "
          f"{'ours':>8s} {'ref':>8s} {'gate':>8s}")
    compared, skipped, failed, conditional = [], [], [], []
    notes = []
    for s in m["stage_order"]:
        info = m["stages"][s]
        n = int(np.prod(info["shape"]))
        dpath = os.path.join(dump_dir, f"{s}.f32")
        st, a, found = read_raw(dpath, "<f4", n)
        if st is MISSING:
            skipped.append(s)
            continue
        note = None
        if st is WRONG_SIZE:
            if s in ("rope.cos", "rope.sin"):
                sel, note = select_rope_rows(a, info, m, s)
                if sel is None:
                    print(f"  {s:16s} {'WRONG SIZE':10s} dump holds {found} float32, "
                          f"manifest says {n} {info['shape']}")
                    failed.append(s)
                    continue
                a = sel
            else:
                print(f"  {s:16s} {'WRONG SIZE':10s} dump holds {found} float32, "
                      f"manifest says {n} {info['shape']}")
                failed.append(s)
                continue
        dump = np.asarray(a, dtype=np.float32).reshape(info["shape"])
        ref = np.fromfile(os.path.join(case_dir, info["ref64"]),
                          dtype="<f8").reshape(info["shape"])
        gate = rtol_override if rtol_override is not None else info["rtol_gate"]
        at = atol if atol is not None else info["atol"]
        c = compare(dump, ref, gate, at)
        ours = smallest_passing_rtol(dump, ref, at)
        ref_arm = info["rtol_torch_fp32"]
        compared.append(s)
        cond = (s in ("rope.cos", "rope.sin") and anchor in ("different", "unverified"))
        verdict = "PASS" if c["ok"] else "FAIL"
        if cond:
            verdict = "COND-" + ("PASS" if c["ok"] else "FAIL")
            conditional.append(s)
        elif not c["ok"]:
            failed.append(s)
        print(f"  {s:16s} {verdict:10s} {c['max_abs']:10.3e} {c['max_rel']:10.3e} "
              f"{('NONE' if ours is None else f'{ours:g}'):>8s} "
              f"{('NONE' if ref_arm is None else f'{ref_arm:g}'):>8s} {gate:8g}")
        if note:
            print(f"      note: {note}")
        if c["n_overflow"]:
            print(f"      {c['n_overflow']} element(s) are EXPLAINED FP32 OVERFLOWS (dump is an "
                  f"infinity of the right sign where the float64 reference exceeds the float32 "
                  f"maximum). Contract 4.1(b) says this is by construction.")
        if c["n_zero_sign_bad"]:
            i = c["first_zero_sign_bad"]
            print(f"      SIGNED ZERO MISMATCH: {c['n_zero_sign_bad']} of {c['n_ref_zero']} "
                  f"reference zeros have the wrong sign bit; first at flat {i}, "
                  f"dump={dump.ravel()[i]!r} ref={ref.ravel()[i]!r}. A tolerance pass cannot "
                  f"see this (isclose calls +0.0 and -0.0 equal), which is why it is separate.")
        if ours is not None and ref_arm is not None and ours < ref_arm:
            notes.append(f"{s}: ours ({ours:g}) is TIGHTER than the reference arm ({ref_arm:g}). "
                         f"That is the interesting result, not a problem.")
        if ours is not None and ref_arm is not None and ours > (gate or 1.0):
            notes.append(f"{s}: ours ({ours:g}) is LOOSER than the gate ({gate:g}). That is "
                         f"the finding.")
        if not c["ok"]:
            i = c["first_bad"]
            if i is not None:
                idx = tuple(int(v) for v in np.unravel_index(i, info["shape"]))
                print(f"      first failing element flat={i} index={idx} "
                      f"dump={dump.ravel()[i]!r} ref={ref.ravel()[i]!r}")
            for line in explain(dump, ref, info["shape"], m, gate, at):
                print(f"      EXPLAINED: {line}")

    unknown = sorted(f[:-4] for f in os.listdir(dump_dir)
                     if f.endswith(".f32") and f[:-4] not in m["stage_order"]
                     and f[:-4] not in m["tensors"])
    if skipped:
        print(f"\n  SKIPPED, no dump present (not a failure): {', '.join(skipped)}")
    if unknown:
        print(f"  dumps with no reference in this case: {', '.join(unknown)}")
    for n_ in notes:
        print(f"  NOTE {n_}")
    print(f"\n  what this case CANNOT certify: {m['cannot_certify']}")
    if m.get("sensitivity"):
        blind = [p for p, v in sorted(m["sensitivity"].items()) if not v.get("seen")]
        if blind:
            print(f"  MEASURED BLIND SPOTS of this case (perturbations it cannot see at all): "
                  f"{', '.join(blind)}")
    if not compared:
        print("\nEXIT 2: no stage dumps found; nothing was compared and nothing is claimed")
        return 2
    if failed:
        print(f"\nFAIL: {len(failed)}/{len(compared)} compared stages failed: "
              f"{', '.join(failed)}")
        return 1
    if conditional:
        print(f"\nPASS with {len(conditional)} CONDITIONAL stage(s) ({', '.join(conditional)}): "
              f"the rotary anchor did not match bitwise, so those stages measure the anchor.")
        return 0
    print(f"\nPASS: {len(compared)} stages compared, all within the calibrated gate, and every "
          f"reference zero's sign bit matched. Consistent with the reference algorithm at "
          f"FP32-roundoff scale. NOT a bitwise certificate and NOT a cross-vendor claim.")
    return 0


# --------------------------------------------------------------------------
# --self-test
# --------------------------------------------------------------------------
def run_self_test(case_dir):
    """The tolerance calibration for ONE case, read off the committed corpus.

    ref32 is a plain FP32 CPU run of the same algorithm. It is never a target.
    Its number is the FLOOR: holding a lane dump tighter than the reference
    arm's own rung makes the tolerance the defect the first time a stage's
    magnitudes move, which is exactly what happened to the mamba lane's
    adv_gate_saturation case at rtol 1e-7 on values of order 7.25e8."""
    m = load_manifest(case_dir)
    print(f"self-test {m['name']}: ref32 (plain FP32 CPU) vs ref64, "
          f"atol {m['stages'][m['stage_order'][0]]['atol']:g}")
    print(f"  {'stage':16s} {'max_abs':>10s} {'max_rel':>10s} {'fp32':>8s} {'best':>8s} "
          f"{'gate':>8s} {'zeros':>7s} {'ovf':>5s}")
    rc = 0
    for s in m["stage_order"]:
        i = m["stages"][s]
        ref = np.fromfile(os.path.join(case_dir, i["ref64"]), dtype="<f8").reshape(i["shape"])
        r32 = np.fromfile(os.path.join(case_dir, i["ref32"]), dtype="<f4").reshape(i["shape"])
        c = compare(r32, ref, RTOL_LADDER[-1], i["atol"])
        rt = i["rtol_torch_fp32"]
        br = i["rtol_correctly_rounded"]
        if rt is None:
            rc = 1
        rt_s = "NONE" if rt is None else f"{rt:g}"
        br_s = "NONE" if br is None else f"{br:g}"
        print(f"  {s:16s} {c['max_abs']:10.3e} {c['max_rel']:10.3e} "
              f"{rt_s:>8s} {br_s:>8s} "
              f"{i['rtol_gate']:8g} {i['n_ref_zero']:7d} {i['n_fp32_overflow']:5d}")
    print("\n  fp32 = the reference arm's smallest passing rung; best = the float64 value")
    print("  rounded once to float32, the floor no FP32 implementation beats; gate = this")
    print("  tool's default, one rung looser than fp32. fp32 far above best means the")
    print("  REFERENCE arm is the limit at that stage, not the lane.")
    print("  NONE means the reference arm cannot pass at the loosest rung. That is a")
    print("  reportable finding about the case, not a licence to widen the ladder.")
    return rc


# --------------------------------------------------------------------------
# --negative-control
# --------------------------------------------------------------------------
def _corrupt(name, a, ref, gate):
    """Return a corrupted copy of the stand-in dump, or None if not applicable."""
    b = np.array(a, dtype=np.float32, copy=True)
    if name == "scale_by_two":
        return (b.astype(np.float64) * 2.0).astype(np.float32)
    if name == "add_constant":
        k = max(1.0, float(np.abs(np.asarray(ref, dtype=np.float64)).max()))
        return (b.astype(np.float64) + k).astype(np.float32)
    if name == "perturb_one_element_beyond_gate":
        if b.size == 0:
            return None
        f = b.ravel()
        j = int(np.argmax(np.abs(f)))
        f = f.copy()
        f[j] = np.float32(float(f[j]) * (1.0 + 1000.0 * gate) + 1000.0 * DEFAULT_ATOL)
        return f.reshape(b.shape)
    if name == "flip_all_zero_signs":
        f = b.ravel().copy()
        z = (f == 0)
        if not z.any():
            return None
        f[z] = np.where(np.signbit(f[z]), np.float32(0.0), np.float32(-0.0))
        return f.reshape(b.shape)
    if name == "transpose_last_two_axes":
        if b.ndim < 2 or b.shape[-1] != b.shape[-2]:
            return None
        return np.ascontiguousarray(np.swapaxes(b, -1, -2))
    if name == "reverse_last_axis":
        if b.ndim < 1 or b.shape[-1] < 2:
            return None
        return np.ascontiguousarray(b[..., ::-1])
    if name == "roll_token_axis":
        if b.ndim < 1 or b.shape[0] < 2:
            return None
        return np.ascontiguousarray(np.roll(b, 1, axis=0))
    raise KeyError(name)


CORRUPTIONS = {
    "perturb_one_element_beyond_gate":
        "moves ONE element far past the gate. If this is not caught the check is VACUOUS.",
    "scale_by_two":
        "multiplies the stage by 2. Must be caught, and the explainer must name the factor.",
    "add_constant":
        "adds a constant. Must be caught, and the explainer must name the offset.",
    "flip_all_zero_signs":
        "flips the sign bit of every zero and changes NOTHING else. A tolerance check cannot "
        "see this at all; only the sign-bit check can. Not applicable where a stage has no "
        "zero, and that is reported rather than skipped silently.",
    "transpose_last_two_axes":
        "transposes the last two axes where they are square. Where this is NOT caught, the "
        "case cannot certify that stage's layout, which is the mamba base_b1_l1_d8 lesson.",
    "reverse_last_axis":
        "reverses the last axis. Inert where the last axis has one element, which is exactly "
        "how a small case silently stops certifying an ordering.",
    "roll_token_axis":
        "rolls the leading axis by one token. Inert at M == 1.",
}


def run_negative_control(case_dir):
    """Corrupt the corpus's own ref32 arm in named ways and demand each is caught.

    A checker with no negative control passes for ever. This runs today, with
    no lane code and no dump, because the stand-in dump is the corpus's own
    ref32. Two outcomes matter equally. A corruption that is CAUGHT proves the
    instrument moves. A corruption that is NOT caught is printed as a thing
    this case CANNOT CERTIFY, and that list is the honest coverage statement
    for the case."""
    m = load_manifest(case_dir)
    print(f"negative control on {m['name']}: the stand-in dump is this case's own ref32, so a")
    print("baseline PASS is expected first; every corruption after it must be CAUGHT.")
    rc = 0
    base_fail = []
    stages = m["stage_order"]
    refs, dumps, gates = {}, {}, {}
    for s in stages:
        i = m["stages"][s]
        refs[s] = np.fromfile(os.path.join(case_dir, i["ref64"]), dtype="<f8").reshape(i["shape"])
        dumps[s] = np.fromfile(os.path.join(case_dir, i["ref32"]),
                               dtype="<f4").reshape(i["shape"])
        gates[s] = i["rtol_gate"]
        if not compare(dumps[s], refs[s], gates[s], i["atol"])["ok"]:
            base_fail.append(s)
    print(f"  baseline: {len(stages) - len(base_fail)}/{len(stages)} stages pass at the "
          f"calibrated gate" + (f"; NOT passing: {', '.join(base_fail)}" if base_fail else ""))
    if base_fail:
        print("  a baseline failure means the gate is tighter than the reference arm it was")
        print("  calibrated from, which is a defect in the calibration, not in the lane.")
        rc = 1

    print(f"\n  {'corruption':34s} {'caught':>7s} {'inert':>6s} {'n/a':>5s}  stages NOT caught")
    blind = {}
    for cname in CORRUPTIONS:
        caught, inert, na = [], [], []
        for s in stages:
            i = m["stages"][s]
            bad = _corrupt(cname, dumps[s], refs[s], gates[s])
            if bad is None:
                na.append(s)
                continue
            if np.array_equal(bad, dumps[s]) and \
                    np.array_equal(np.signbit(bad), np.signbit(dumps[s])):
                inert.append(s)
                continue
            (caught if not compare(bad, refs[s], gates[s], i["atol"])["ok"]
             else inert).append(s)
        blind[cname] = inert
        print(f"  {cname:34s} {len(caught):7d} {len(inert):6d} {len(na):5d}  "
              f"{', '.join(inert[:6])}{' ...' if len(inert) > 6 else ''}")
        if not caught and not na:
            print(f"    NOT CAUGHT ANYWHERE. This case cannot see {cname} at any stage.")
            rc = max(rc, 4)

    print("\n  file-level classification, which is NOT a value comparison:")
    s0 = stages[0]
    i0 = m["stages"][s0]
    n0 = int(np.prod(i0["shape"]))
    st, _, found = read_raw(os.path.join(case_dir, "ref32", "no_such_stage.f32"), "<f4", n0)
    print(f"    an absent file classifies as {st} (never DIFFERENT; `cmp -s a b` says "
          f"'differ' when b does not exist, and the mamba lane read that as a total "
          f"divergence once)")
    rc = max(rc, 0 if st is MISSING else 4)
    tmp = os.path.join(case_dir, "ref32", f"{s0}.f32")
    st2, _, found2 = read_raw(tmp, "<f4", n0 + 1)
    print(f"    a file of the wrong length classifies as {st2} ({found2} elements found, "
          f"{n0 + 1} demanded), never as a value difference")
    rc = max(rc, 0 if st2 is WRONG_SIZE else 4)

    print("\n  WHAT THIS CASE CANNOT CERTIFY, measured rather than claimed:")
    for cname, inert in sorted(blind.items()):
        if inert:
            print(f"    {cname}: inert at {len(inert)} stage(s) -> {', '.join(inert)}")
    print(f"\n  and from the generator's own sensitivity table: {m['cannot_certify']}")
    print(f"\nnegative control exit code: {rc} "
          f"(0 every corruption was caught somewhere and the file classifications are right)")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir", help="ONE corpus case directory (holds manifest.json, ref64/)")
    ap.add_argument("dump_dir", nargs="?", help="directory of <stage>.f32 dumps from the lane")
    ap.add_argument("--rtol", type=float, default=None,
                    help="override the per-stage calibrated gate for every stage")
    ap.add_argument("--atol", type=float, default=None,
                    help="override the per-stage atol (default is the manifest's)")
    ap.add_argument("--self-test", action="store_true",
                    help="print this case's tolerance calibration and exit")
    ap.add_argument("--negative-control", action="store_true",
                    help="corrupt this case's own ref32 in named ways and demand each is "
                         "caught; prints what the case cannot certify")
    ap.add_argument("--inputs-differ-ok", action="store_true",
                    help="continue past a bitwise input mismatch; every number is then "
                         "uninterpretable and the tool says so")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(run_self_test(a.case_dir))
    if a.negative_control:
        sys.exit(run_negative_control(a.case_dir))
    if not a.dump_dir:
        ap.error("dump_dir is required unless --self-test or --negative-control")
    sys.exit(run_check(a.case_dir, a.dump_dir, a.rtol, a.atol, a.inputs_differ_ok))


if __name__ == "__main__":
    main()
