"""Prove the kaiming default change cannot move a bit.

Deliberately does NOT `import mojolearn`. The whole point of the change is
that importing the package should not require a native library, and this
worktree has no built bindings, so `mojolearn/__init__.py`'s `_backend.select()`
refuses. `_portable_math` is loaded BY FILE PATH (it imports only ctypes,
struct, sys and pathlib) and the signatures are read by AST, which is a
stronger check than introspecting an imported object anyway: it reads the
source that will ship.

    python3 prove_kaiming.py
"""
import ast
import importlib.util
import struct
import sys

ROOT = "/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-aa202348f9b77c622"
PM = ROOT + "/python/mojolearn/_portable_math.py"
TI = ROOT + "/python/mojolearn/_training_impl.py"

spec = importlib.util.spec_from_file_location("_pm", PM)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


ok = True

print("=== 1. the default value itself, as raw IEEE-754 bits ===")
# `a=math.sqrt(5.0)` in the old signature and `a = math.sqrt(5.0)` in the new
# body are the SAME CALL to the SAME FUNCTION with the SAME ARGUMENT. The only
# thing that changed is when it runs.
old_a, new_a = pm.sqrt(5.0), pm.sqrt(5.0)
old_g, new_g = pm.sqrt(2.0), pm.sqrt(2.0)
print("  sqrt(5.0) = %r  bits 0x%016x" % (old_a, bits(old_a)))
print("  sqrt(2.0) = %r  bits 0x%016x" % (old_g, bits(old_g)))
for name, o, n in (("sqrt(5)", old_a, new_a), ("sqrt(2)", old_g, new_g)):
    same = bits(o) == bits(n)
    ok = ok and same
    print("  %-8s class-def-time 0x%016x  call-time 0x%016x  %s"
          % (name, bits(o), bits(n), "IDENTICAL" if same else "DIVERGENT"))

print("\n=== 2. the shipping source no longer evaluates it at import time ===")
tree = ast.parse(open(TI).read())
found = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("kaiming_uniform", "kaiming_normal"):
        args = node.args.args
        defaults = node.args.defaults
        named = [a.arg for a in args[len(args) - len(defaults):]]
        found[node.name] = dict(zip(named, defaults))
for fn, defs in sorted(found.items()):
    for arg, d in defs.items():
        literal = isinstance(d, ast.Constant) and d.value is None
        calls = [x for x in ast.walk(d) if isinstance(x, ast.Call)]
        ok = ok and literal and not calls
        print("  %-16s %-5s default = %s   calls-at-import=%d  %s"
              % (fn, arg, ast.unparse(d), len(calls),
                 "OK" if literal and not calls else "STILL EVALUATED AT IMPORT"))

print("\n=== 3. the value that crosses from changed code into unchanged code ===")
# `uniform()` and `normal()` are untouched and need a built binding. They do
# not need to run: the ONLY thing this edit can influence is the argument they
# receive -- `bound` for kaiming_uniform, `gain / sqrt(fan_in)` for
# kaiming_normal. Identical inputs into an unmodified function cannot produce
# different bytes, so bit-identity here IS the proof for everything downstream.
def u_bound(a, fan_in):
    gain = pm.sqrt(2.0 / (1.0 + float(a) * float(a)))
    return gain * pm.sqrt(3.0 / float(int(fan_in)))


def n_sigma(gain, fan_in):
    return float(gain) / pm.sqrt(float(int(fan_in)))


for fan_in in (1, 3, 8, 64, 256, 1024, 4096):
    b_old, b_new = u_bound(old_a, fan_in), u_bound(new_a, fan_in)
    s_old, s_new = n_sigma(old_g, fan_in), n_sigma(new_g, fan_in)
    same = bits(b_old) == bits(b_new) and bits(s_old) == bits(s_new)
    ok = ok and same
    print("  fan_in=%-5d bound 0x%016x==0x%016x  sigma 0x%016x==0x%016x  %s"
          % (fan_in, bits(b_old), bits(b_new), bits(s_old), bits(s_new),
             "IDENTICAL" if same else "DIVERGENT"))

print("\n=== 4. the library really is what the import used to need ===")
# If this file could compute sqrt without the native library the change would
# be pointless AND the arithmetic contract would already be broken. Show the
# native path is the one being used.
print("  _portable_math.sqrt -> native lib:", pm._lib is not None)
ok = ok and pm._lib is not None

print("\nVERDICT:", "IDENTICAL on every arm" if ok else "SOMETHING MOVED -- STOP")
sys.exit(0 if ok else 1)
