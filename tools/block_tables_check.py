#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every table that lists Sequential layer codes or layer names agrees, for EVERY code (lanes mamba2-nan and
layer-tables, 2026-11-23).

Where a layer code or name must appear for a layer to work end to end. Each place is checked below; "generic"
means the place reads the code through `is_block` and the model_blocks tables, so it needs no line per code, and
the check proves it still does (a literal code list there is how 42 to 46 went missing from load_state_dict).

  autograd/model_blocks.mojo (block layers, BLOCK_FIRST .. BLOCK_LAST: each code defined here, by a delegated
                              module, or in the never-assigned BLOCK_UNUSED_FIRST .. BLOCK_UNUSED_LAST)
    LAYER_* constants         the code itself; one constant per code, no gap, no value shared with model.mojo
    block_shape               construct and shape: argument validation and the output shape
    block_param_sizes         parameter layout: the tensors of the flat params_ vector
    block_param_names         state_dict / load_state_dict / checkpoint names of those tensors
    block_init_bounds         seeded init of each tensor
    block_tensor_dims         each tensor's shape for import, export and state_dict (a branch iff it has tensors)
    block_forward             forward (backward is the tape of the ops it records)
    block_reads               the codes that read an earlier layer's output (their reference is renumbered)
  delegated modules (DELEGATES below: autograd/nn_attention.mojo, nn_conv_blocks.mojo, ...): codes in
                              BLOCK_FIRST..BLOCK_LAST that a lane's own module defines and model_blocks' tables call
                              behind its admission function; the module's tables are checked like model_blocks'
  autograd/model.mojo         core codes 1..20 (LAYER_*); parse_model_v2 (block_shape, block_param_sizes),
                              model_init_params (block_init_bounds), _model_forward_v2 (block_forward), _absorbed_activation
                              (block_reads):
                              generic over block codes
  autograd/model_checkpoint.mojo  checkpoint save and load names: generic (block_param_sizes, block_param_names)
  autograd/weight_import.mojo     import, export, state_dict layouts: generic (block_param_sizes, block_tensor_dims)
  bindings/sequential_weights_pack.mojo
    sequential_parameter_layers   load_state_dict's tensor names: generic (is_block, block_param_names)
    sequential_export_weights     state_dict / parameters / named_parameters names: generic (is_block, names)
  bindings/sequential_description_pack.mojo
    sequential_add            construct: one `kind == "<name>"` branch per name that takes arguments
    _CODE_* / _emit           containers: the codes whose reference is renumbered equal model.mojo's and include
                              every code block_reads names
    sequential_layers         `model.layers`: reads the layer-name tuple below (generic)
  python/mojolearn/_x_sequence_model.py
    _LAYER                    name -> code, the table add() and recipes resolve a name through; every code named
    layer_names               code -> the name `model.layers` reports; one non-empty, unique name per code
    Sequential.<name>         a public method per name
    Sequential.add docstring  every name listed
  python/mojolearn/_verify_api_neural.py
    SEQUENTIAL_LAYER_FIXTURES verify's endpoint fixture per layer method, and the constructor table of the boundary
                              test python/mojolearn/tests/test_sequential_layers_boundary.py (fit one step,
                              state_dict -> load_state_dict, save_checkpoint -> from_checkpoint, export -> load):
                              every code needs one, except the parameter-free core codes CORE_NO_FIXTURE
  Generic, nothing per code: python/mojolearn/_digest_fields.py (digest_ reads `_layers`, the codes), the checkpoint
  and export files (they store codes), recipe serialisation (names through add() and _LAYER).

For model_blocks.mojo, per code: the number of tensors in block_param_sizes, block_param_names and block_init_bounds
is the same; block_tensor_dims has a branch exactly when the layer has parameters; no branch of the four tables is
empty; block_shape and block_forward name the code. A `for _ in range(K):` around an append counts K appends.
Source reading only: no build, no binding, no import of the package.

    python3 tools/block_tables_check.py [--root DIR | --tree SHA]    exit 0 when consistent, 1 with the list
    python3 tools/block_tables_check.py path/to/model_blocks.mojo    the model_blocks tables only
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "autograd/model_blocks.mojo"
BLOCKS = "autograd/model_blocks.mojo"
MODEL = "autograd/model.mojo"
CHECKPOINT = "autograd/model_checkpoint.mojo"
IMPORT = "autograd/weight_import.mojo"
WEIGHTS = "bindings/sequential_weights_pack.mojo"
DESCRIPTION = "bindings/sequential_description_pack.mojo"
PYMODEL = "python/mojolearn/_x_sequence_model.py"
FIXTURES = "python/mojolearn/_verify_api_neural.py"
TABLES = ("block_param_sizes", "block_param_names", "block_init_bounds", "block_tensor_dims")
NAMED = ("block_shape", "block_forward")
#: (file, function, the calls it must make): the generic consumers of the block tables
GENERIC = (
    (MODEL, "parse_model_v2", ("is_block(", "block_shape(", "block_param_sizes(")),
    (MODEL, "model_init_params", ("is_block(", "block_init_bounds(")),
    (MODEL, "_model_forward_v2", ("is_block(", "block_forward(")),
    (MODEL, "_absorbed_activation", ("is_block(", "block_reads(")),
    (CHECKPOINT, None, ("is_block(", "block_param_sizes(", "block_param_names(")),
    (IMPORT, None, ("is_block(", "block_param_sizes(", "block_tensor_dims(")),
    (WEIGHTS, "sequential_parameter_layers", ("is_block(", "block_param_names(")),
    (WEIGHTS, "sequential_export_weights", ("is_block(", "block_param_names(")),
)
#: delegated block codes (lane nn-integrate, 2026-11-23): a module that defines its own LAYER_* codes and tables,
#: which model_blocks' tables call behind its admission function. Per module: the file, the admission function
#: (its codes lie in BLOCK_FIRST..BLOCK_LAST, which is_block admits), and for each model_blocks table the module's function (None: not delegated). The
#: module's tables are checked like model_blocks' own: per code the same tensor count in sizes, names and inits, a
#: dims branch exactly when it has tensors, and the shape (and forward, when named) function names the code.
DELEGATES = (
    ("autograd/nn_attention.mojo", "nn_attention_is", {
        "block_param_sizes": "nn_attention_param_sizes", "block_param_names": "nn_attention_param_names",
        "block_init_bounds": "nn_attention_init_bounds", "block_tensor_dims": "nn_attention_tensor_dims",
        "block_shape": "nn_attention_shape", "block_forward": None}),
    ("autograd/nn_activations.mojo", "is_nn_activation", {
        "block_param_sizes": "nn_act_param_sizes", "block_param_names": "nn_act_param_names",
        "block_init_bounds": "nn_act_init_bounds", "block_tensor_dims": "nn_act_tensor_dims",
        "block_shape": "nn_act_shape", "block_forward": "nn_act_forward"}),
    ("autograd/nn_conv3d_layers.mojo", "is_conv3d_layer", {
        "block_param_sizes": "conv3d_layer_sizes", "block_param_names": "conv3d_layer_names",
        "block_init_bounds": "conv3d_layer_bounds", "block_tensor_dims": "conv3d_layer_dims",
        "block_shape": "conv3d_layer_shape", "block_forward": "conv3d_layer_forward"}),
    ("autograd/nn_conv_blocks.mojo", "is_nn_conv", {
        "block_param_sizes": "nnc_param_sizes", "block_param_names": "nnc_param_names",
        "block_init_bounds": "nnc_init_bounds", "block_tensor_dims": "nnc_tensor_dims",
        "block_shape": "nnc_shape", "block_forward": "nnc_forward"}),
)
#: core layers built by add() with no arguments (the description pack's final `takes no arguments` branch)
NO_ARGUMENTS = frozenset(("relu", "sigmoid", "tanh", "softmax", "flatten",
                          # nn-activations: block layers without arguments, the same final branch
                          "identity", "mish", "hardswish", "hardsigmoid"))
#: parameter-free core codes verify covers through its own endpoint branches (relu .. avg_pool2d but the
#: parameter layers 1 linear, 6 layer_norm and 12 conv2d); every other code, and every new one, needs a fixture
CORE_NO_FIXTURE = frozenset((2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14))
_CODE = re.compile(r"\bcode == (LAYER_[A-Z0-9_]+)")
_HEADER = re.compile(r"^\s*(?:el)?if\b.*\bcode == LAYER_")
_RANGE = re.compile(r"^(\s*)for _ in range\((\d+)\)")


def constants(text):
    """{name: value} of the module's `comptime LAYER_* = <int>` lines, and BLOCK_FIRST / BLOCK_LAST."""
    out = {}
    for m in re.finditer(r"^comptime ((?:LAYER|BLOCK)_[A-Z0-9_]+) = (\d+)\b", text, re.M):
        out[m.group(1)] = int(m.group(2))
    return out


def function_body(text, name, required=True):
    """The lines of `def name(...)` up to the next top-level statement."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"def {name}(") or line.startswith(f"def {name}["):
            body = []
            for nxt in lines[i + 1:]:
                # a multi-line signature closes with `) raises -> T:` at column 0
                if nxt and not nxt[0].isspace() and not nxt.startswith(("#", ")")):
                    break
                body.append(nxt)
            return body
    if required:
        raise SystemExit(f"block_tables_check: {name} not found in the source")
    return None


def branches(body):
    """[(codes, appends)] per `if/elif code == LAYER_*` branch of a table function, in order. A branch ends at the
    next line indented at or left of its header (another header, an else, the return)."""
    out = []
    i = 0
    while i < len(body):
        line = body[i]
        if not _HEADER.match(line):
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        codes = _CODE.findall(line)
        appends = 0
        loops = []  # (indent, count) of enclosing `for _ in range(K)`
        j = i + 1
        while j < len(body):
            nxt = body[j]
            stripped = nxt.strip()
            if stripped and not stripped.startswith("#"):
                ind = len(nxt) - len(nxt.lstrip())
                if ind <= indent:
                    break
                while loops and ind <= loops[-1][0]:
                    loops.pop()
                m = _RANGE.match(nxt)
                if m:
                    loops.append((len(m.group(1)), int(m.group(2))))
                elif "out.append(" in nxt:
                    k = 1
                    for _, n in loops:
                        k *= n
                    appends += k
            j += 1
        out.append((codes, appends))
        i = j
    return out


def unused_codes(values):
    """The codes BLOCK_UNUSED_FIRST..BLOCK_UNUSED_LAST (never assigned, not layers)."""
    lo, hi = values.get("BLOCK_UNUSED_FIRST"), values.get("BLOCK_UNUSED_LAST")
    return set(range(lo, hi + 1)) if lo is not None and hi is not None else set()


def delegated_values(read=None):
    """The codes the DELEGATES modules define: {value: name}."""
    read = read or reader()
    out = {}
    for rel, _, _ in DELEGATES:
        text = read(rel)
        if text is not None:
            for k, v in constants(text).items():
                if k.startswith("LAYER_"):
                    out[v] = k
    return out


def check_text(text, delegated=None):
    """check() over model_blocks.mojo's TEXT; DELEGATED {value: name} are the codes the lanes' own modules define
    (None: read them from the tree)."""
    if delegated is None:
        delegated = delegated_values()
    values = constants(text)
    first, last = values.get("BLOCK_FIRST"), values.get("BLOCK_LAST")
    if first is None or last is None:
        return ["BLOCK_FIRST / BLOCK_LAST not found"], {}
    unused = unused_codes(values)
    names = {v: k for k, v in values.items() if k.startswith("LAYER_") and first <= v <= last}
    problems = []
    for k, v in values.items():
        if k.startswith("LAYER_") and v > last:
            problems.append(f"{k} = {v} lies above BLOCK_LAST = {last} (is_block refuses it)")
    for v, k in delegated.items():
        if v > last:
            problems.append(f"{k} = {v} (a delegated module's) lies above BLOCK_LAST = {last} (is_block refuses it)")
    for v in range(first, last + 1):
        here, there = v in names, v in delegated
        if v in unused:
            if here or there:
                problems.append(f"code {v}: lies in BLOCK_UNUSED_FIRST..BLOCK_UNUSED_LAST but is "
                                f"{names.get(v) or delegated.get(v)}")
        elif here and there:
            problems.append(f"code {v}: {names[v]} here and {delegated[v]} in a delegated module")
        elif not here and not there:
            problems.append(f"code {v}: no LAYER_ constant between BLOCK_FIRST and BLOCK_LAST (here, in a "
                            f"delegated module, or BLOCK_UNUSED)")
    counts = {name: [0, 0, 0, 0] for name in names.values()}
    seen = {name: [0, 0, 0, 0] for name in names.values()}
    for t, table in enumerate(TABLES):
        for codes, appends in branches(function_body(text, table)):
            for code in codes:
                if code not in counts:
                    problems.append(f"{table}: {code} is not a block layer code")
                    continue
                seen[code][t] += 1
                counts[code][t] = appends
                if appends == 0:
                    problems.append(f"{table}: the {code} branch is empty")
    for code, (n_sizes, n_names, n_inits, n_dims) in counts.items():
        for t, table in enumerate(TABLES):
            if seen[code][t] > 1:
                problems.append(f"{table}: {code} has {seen[code][t]} branches")
        if not (n_sizes == n_names == n_inits):
            problems.append(f"{code}: {n_sizes} parameter sizes, {n_names} names and {n_inits} inits")
        if (n_sizes > 0) != (seen[code][3] > 0):
            problems.append(f"{code}: {n_sizes} parameter tensors but "
                            f"{'a' if seen[code][3] else 'no'} block_tensor_dims branch")
    for fn in NAMED:
        named = set(_CODE.findall("\n".join(function_body(text, fn))))
        for code in counts:
            if code not in named:
                problems.append(f"{fn}: no rule for {code}")
    return problems, {c: tuple(n) for c, n in counts.items()}


def check(path=SOURCE):
    """The list of disagreements of model_blocks.mojo's four tables (empty: consistent) and the per-code table
    {code name: (sizes, names, inits, dims)}: the tensor counts of the first three tables, and the appends of the
    code's block_tensor_dims branch (the axes it can give a shape; 0 without a branch)."""
    return check_text(Path(path).read_text(encoding="utf-8"))


def _cond_codes(text, cond, consts):
    """The LAYER_* names a branch condition admits: named in it, or in the body of a helper it calls."""
    found = set(re.findall(r"\b(LAYER_[A-Z0-9_]+)\b", cond))
    for helper in re.findall(r"\b(_?[a-z][a-z0-9_]*)\(code\b", cond):
        body = function_body(text, helper, required=False) or []
        found |= set(re.findall(r"\b(LAYER_[A-Z0-9_]+)\b", "\n".join(body)))
        joined = "\n".join(body)
        m = re.search(r"code >= ([A-Z0-9_]+) and code <= ([A-Z0-9_]+)", joined)
        allc = {k.group(1): int(k.group(2)) for k in re.finditer(r"^comptime ([A-Z0-9_]+) = (\d+)\b", text, re.M)}
        if m and m.group(1) in allc and m.group(2) in allc:
            lo, hi = allc[m.group(1)], allc[m.group(2)]
            found |= {k for k, v in consts.items() if k.startswith("LAYER_") and lo <= v <= hi}
    return {c for c in found if c in consts}


def _delegate_branches(text, fn, consts):
    """[(codes, appends)] per top-level `if` branch of a delegated table function (any admitting condition)."""
    body = function_body(text, fn, required=False)
    if body is None:
        return None
    out = []
    i = 0
    while i < len(body):
        line = body[i]
        m = re.match(r"^(\s{4})(?:el)?if\b(.*):\s*(#.*)?$", line)
        if not m:
            i += 1
            continue
        codes = _cond_codes(text, m.group(2), consts)
        appends, loops, j = 0, [], i + 1
        while j < len(body):
            nxt = body[j]
            stripped = nxt.strip()
            if stripped and not stripped.startswith("#"):
                ind = len(nxt) - len(nxt.lstrip())
                if ind <= 4:
                    break
                while loops and ind <= loops[-1][0]:
                    loops.pop()
                r = _RANGE.match(nxt)
                if r:
                    loops.append((len(r.group(1)), int(r.group(2))))
                elif "out.append(" in nxt:
                    k = 1
                    for _, n in loops:
                        k *= n
                    appends += k
            j += 1
        out.append((codes, appends))
        i = j
    return out


def check_delegate(path, text, admit, tables, blocks_text):
    """(problems, {code name: (sizes, names, inits, dims)}, {code name: value}) of one delegated module."""
    consts = {k: v for k, v in constants(text).items() if k.startswith("LAYER_")}
    problems = []
    if not consts:
        return [f"{path}: no LAYER_ constants"], {}, {}
    admitted = _cond_codes(text, f"{admit}(code)", consts)
    for c in sorted(set(consts) - admitted):
        problems.append(f"{path}: {c} is not admitted by {admit}")
    bvals = constants(blocks_text)
    first, last, unused = bvals.get("BLOCK_FIRST"), bvals.get("BLOCK_LAST"), unused_codes(bvals)
    for c in sorted(admitted):
        v = consts[c]
        if first is None or not (first <= v <= last) or v in unused:
            problems.append(f"{path}: {c} = {v} lies outside model_blocks' BLOCK_FIRST..BLOCK_LAST (is_block refuses it)")
    counts = {c: [0, 0, 0, 0] for c in admitted}
    seen = {c: [0, 0, 0, 0] for c in admitted}
    for t, table in enumerate(TABLES):
        fn = tables.get(table)
        if fn is None:
            continue
        if fn + "(" not in "\n".join(function_body(blocks_text, table, required=False) or []):
            problems.append(f"{BLOCKS} {table}: does not call {fn}")
        branches_ = _delegate_branches(text, fn, consts)
        if branches_ is None:
            problems.append(f"{path}: {fn} not found")
            continue
        for codes, appends in branches_:
            for code in codes:
                if code in counts:
                    seen[code][t] += 1
                    counts[code][t] = appends
                    if appends == 0:
                        problems.append(f"{path} {fn}: the {code} branch is empty")
    for code, (n_sizes, n_names, n_inits, _) in counts.items():
        if not (n_sizes == n_names == n_inits):
            problems.append(f"{path} {code}: {n_sizes} parameter sizes, {n_names} names and {n_inits} inits")
        if (n_sizes > 0) != (seen[code][3] > 0):
            problems.append(f"{path} {code}: {n_sizes} parameter tensors but "
                            f"{'a' if seen[code][3] else 'no'} tensor_dims branch")
    for table in NAMED:
        fn = tables.get(table)
        if fn is None:
            continue
        body = function_body(text, fn, required=False)
        if body is None:
            problems.append(f"{path}: {fn} not found")
            continue
        joined = "\n".join(body)
        code_lines = [l.strip() for l in body if l.strip() and not l.strip().startswith("#")]
        if table == "block_shape" and code_lines and code_lines[-1] == "return cur":
            continue  # every code the module admits keeps its input's shape unless a branch says otherwise
        named = set(re.findall(r"\b(LAYER_[A-Z0-9_]+)\b", joined))
        for line in body:
            m = re.match(r"^\s*(?:el)?if\b(.*):", line)
            if m:
                named |= _cond_codes(text, m.group(1), consts)
        for code in counts:
            if code not in named:
                problems.append(f"{path} {fn}: no rule for {code}")
    return problems, {c: tuple(n) for c, n in counts.items()}, {c: consts[c] for c in admitted}


# ------------------------------------------------------------------ the other files
def _literal(tree, name):
    """The literal value assigned to module-level NAME, or (for a dict-valued key "layer_names" etc.) None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    return None


def _dict_item(node, key):
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
    return None


def _python_tables(text):
    """(_LAYER {name: code}, layer_names tuple, Sequential method names, the add() docstring) of _x_sequence_model.py."""
    tree = ast.parse(text)
    layer = ast.literal_eval(_literal(tree, "_LAYER"))
    names_node = _dict_item(_literal(tree, "_SEQUENTIAL_CONTEXT"), "layer_names")
    layer_names = ast.literal_eval(names_node) if names_node is not None else None
    methods, add_doc = set(), ""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Sequential":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.add(item.name)
                    if item.name == "add":
                        add_doc = ast.get_docstring(item) or ""
    return layer, layer_names, methods, add_doc


def _fixture_names(text):
    node = _literal(ast.parse(text), "SEQUENTIAL_LAYER_FIXTURES")
    if not isinstance(node, ast.Dict):
        return None
    return [k.value for k in node.keys if isinstance(k, ast.Constant)]


def check_files(read):
    """The disagreements across every place of the module docstring. READ(relative path) -> text or None."""
    problems = []
    texts = {}
    for rel in (BLOCKS, MODEL, CHECKPOINT, IMPORT, WEIGHTS, DESCRIPTION, PYMODEL, FIXTURES):
        texts[rel] = read(rel)
        if texts[rel] is None:
            problems.append(f"{rel}: missing")
    if texts[BLOCKS] is None or texts[MODEL] is None or texts[PYMODEL] is None:
        return problems, {}
    block_problems, table = check_text(texts[BLOCKS], delegated_values(read))
    problems += [f"{BLOCKS}: {p}" for p in block_problems]
    bvals, mvals = constants(texts[BLOCKS]), constants(texts[MODEL])
    first, last = bvals.get("BLOCK_FIRST"), bvals.get("BLOCK_LAST")
    codes = {}
    for rel, vals in ((MODEL, mvals), (BLOCKS, bvals)):
        for name, v in vals.items():
            if not name.startswith("LAYER_"):
                continue
            if v in codes and codes[v] != name:
                problems.append(f"{rel}: {name} = {v} is also {codes[v]}")
            codes[v] = name
            if rel == MODEL and first is not None and v >= first:
                problems.append(f"{MODEL}: core {name} = {v} lies in the block range {first}..{last}")
    for rel, admit, tables in DELEGATES:
        dtext = read(rel)
        if dtext is None:
            problems.append(f"{rel}: missing")
            continue
        dprob, dtable, dvals = check_delegate(rel, dtext, admit, tables, texts[BLOCKS])
        problems += dprob
        for name, v in dvals.items():
            if v in codes and codes[v] != name:
                problems.append(f"{rel}: {name} = {v} is also {codes[v]}")
            bvals.setdefault(name, v)
            codes[v] = name
        table.update(dtable)
    with_params = {bvals[c] for c, n in table.items() if n[0] > 0 and c in bvals}

    # generic consumers: no per-code line, the model_blocks tables instead
    for rel, fn, calls in GENERIC:
        text = texts.get(rel)
        if text is None:
            continue
        body = text if fn is None else "\n".join(function_body(text, fn, required=False) or [])
        where = f"{rel}" + (f" {fn}" if fn else "")
        if fn is not None and not body:
            problems.append(f"{where}: function not found")
            continue
        for call in calls:
            if call not in body:
                problems.append(f"{where}: does not call {call[:-1]} (a layer's tables are read through it)")
        if first is not None:
            for m in re.finditer(r"\bcode\s*(==|>=|>|<=|<)\s*(\d+)", body):
                v = int(m.group(2))
                if v >= first or (m.group(1) in (">=", ">") and v >= first - 1):
                    problems.append(f"{where}: compares a code with the literal {v}; read block codes through "
                                    f"is_block and the model_blocks tables")

    # python/mojolearn/_x_sequence_model.py
    layer, layer_names, methods, add_doc = _python_tables(texts[PYMODEL])
    named_codes = {}
    for name, v in layer.items():
        if v == 0:
            continue
        if v not in codes:
            problems.append(f"{PYMODEL} _LAYER: {name!r} -> {v} is no LAYER_ code")
        named_codes.setdefault(v, []).append(name)
    for v, cname in sorted(codes.items()):
        if v not in named_codes:
            problems.append(f"{PYMODEL} _LAYER: no name for {cname} ({v})")
    for name in layer:
        if name not in methods:
            problems.append(f"{PYMODEL} Sequential: no method {name}()")
        if f"``'{name}'``" not in add_doc:
            problems.append(f"{PYMODEL} Sequential.add docstring: {name!r} is not listed")
    if layer_names is None:
        problems.append(f"{PYMODEL} _SEQUENTIAL_CONTEXT: no layer_names tuple")
    else:
        top = max(codes) if codes else 0
        if len(layer_names) != top + 1:
            problems.append(f"{PYMODEL} layer_names: {len(layer_names)} entries for codes 0..{top}")
        for v, cname in sorted(codes.items()):
            if v >= len(layer_names) or not layer_names[v]:
                problems.append(f"{PYMODEL} layer_names: no name for {cname} ({v})")
        for v, n in enumerate(layer_names):
            if n and v not in codes:
                problems.append(f"{PYMODEL} layer_names: {n!r} at {v}, which is no LAYER_ code")
        given = [n for n in layer_names if n]
        for n in sorted({n for n in given if given.count(n) > 1}):
            problems.append(f"{PYMODEL} layer_names: {n!r} names more than one code")

    # bindings/sequential_description_pack.mojo
    text = texts.get(DESCRIPTION)
    if text is not None:
        add_body = "\n".join(function_body(text, "sequential_add", required=False) or [])
        kinds = set(re.findall(r'\bkind == "([a-z0-9_]+)"', add_body))
        for name in sorted(layer):
            if name not in NO_ARGUMENTS and name not in kinds:
                problems.append(f"{DESCRIPTION} sequential_add: no branch for {name!r}")
        for name in sorted(kinds - set(layer)):
            problems.append(f"{DESCRIPTION} sequential_add: branch {name!r} is not in _LAYER")
        local = {m.group(1): int(m.group(2)) for m in re.finditer(r"^comptime _CODE_([A-Z0-9_]+) = (\d+)", text, re.M)}
        for name, v in local.items():
            if codes.get(v) != f"LAYER_{name}":
                problems.append(f"{DESCRIPTION}: _CODE_{name} = {v} but {codes.get(v, 'no LAYER_ code')} has it")
        emit = "\n".join(function_body(text, "_emit", required=False) or [])
        renumbered = {local[n] for n in re.findall(r"\bcode == _CODE_([A-Z0-9_]+)", emit) if n in local}
        reads = set(_CODE.findall("\n".join(function_body(texts[BLOCKS], "block_reads", required=False) or [])))
        for cname in sorted(reads):
            if cname in bvals and bvals[cname] not in renumbered:
                problems.append(f"{DESCRIPTION} _emit: {cname} reads an earlier layer but its reference is not "
                                f"renumbered in a container")

    # python/mojolearn/_verify_api_neural.py: the constructor table of the boundary test
    fixtures = _fixture_names(texts[FIXTURES]) if texts.get(FIXTURES) is not None else None
    if fixtures is None:
        problems.append(f"{FIXTURES}: no SEQUENTIAL_LAYER_FIXTURES")
    else:
        covered = {layer.get(n) for n in fixtures}
        for n in fixtures:
            if n not in layer:
                problems.append(f"{FIXTURES} SEQUENTIAL_LAYER_FIXTURES: {n!r} is not a Sequential layer name")
        for v, cname in sorted(codes.items()):
            if v not in CORE_NO_FIXTURE and v not in covered:
                problems.append(f"{FIXTURES} SEQUENTIAL_LAYER_FIXTURES: no constructor for {cname} ({v}"
                                f"{', has parameters' if v in with_params else ''}; names "
                                f"{named_codes.get(v, [])})")
    return problems, table


def reader(root=None, tree=None):
    """READ(relative path) -> text or None, from directory ROOT or git commit TREE."""
    if tree is not None:
        def read(rel):
            p = subprocess.run(["git", "show", f"{tree}:{rel}"], cwd=root or ROOT, capture_output=True)
            return p.stdout.decode("utf-8") if p.returncode == 0 else None
        return read
    base = Path(root or ROOT)

    def read(rel):
        path = base / rel
        return path.read_text(encoding="utf-8") if path.is_file() else None
    return read


def main(argv):
    args = argv[1:]
    if args and not args[0].startswith("--"):
        problems, table = check(Path(args[0]))
    else:
        root = tree = None
        if "--root" in args:
            root = args[args.index("--root") + 1]
        if "--tree" in args:
            tree = args[args.index("--tree") + 1]
        problems, table = check_files(reader(root, tree))
    for code, (s, n, i, d) in sorted(table.items()):
        print(f"{code:<26} sizes {s} names {n} inits {i} dims {'yes' if d else 'no'}")
    for p in problems:
        print("DISAGREE", p)
    print(f"BLOCK_TABLES {'consistent' if not problems else 'INCONSISTENT'}: {len(table)} block codes, "
          f"{len(problems)} problems")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
